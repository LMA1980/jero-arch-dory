//! Wave 4 zero-allocation `112`-virtual-layer decode and row-parallel `M = 1` GEMV tests:
//! - Global `CountingAllocator` enforcing `(0, 0)` heap allocations across full `40`-physical /
//!   `112`-virtual-layer `"S-S-S-*"` (`n_loops = 4`) `DoryKvCacheMode::PerLoop` and `LastLoop` decode
//!   with pre-sized `ScratchWorkspace`, tied & untied LM heads.
//! - Bit-exact (`0.0` `max_abs_diff`) determinism and zero heap allocations across `1`, `4`, and `8`
//!   worker partitions (`DoryRecipe::with_worker_partitions`) across `Q5_K`, `Q6_K`, `Q8_0`,
//!   `Q4_K_M`, `Q4_0`, `Q5_0`, `F32`, `F16`, and `BF16`.
//! - `O(1)` `decode_scratch_bytes` / `peak_used_bytes` invariance across `n_loops = 1..=4`,
//!   fail-closed `ScratchWorkspace` bounds checking, and sequential `decode_step` vs `prefill_step`
//!   parity under `< 1e-4`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};

use jero_arch_dory::{DoryKvCacheMode, DoryRecipe, DorySubLayerKind};
use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, ScratchWorkspace, TensorView,
};

struct CountingAllocator;

static ACTIVE_EPOCH: AtomicU64 = AtomicU64::new(0);
static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TEST_SERIAL_LOCK: Mutex<()> = Mutex::new(());

thread_local! {
    static THREAD_EPOCH: Cell<u64> = const { Cell::new(0) };
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAllocator = CountingAllocator;

// SAFETY: Delegates directly to `std::alloc::System` while atomically counting allocations on the
// calling test thread and active pool workers tagged with the current `ACTIVE_EPOCH`.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let active = ACTIVE_EPOCH.load(Ordering::Relaxed);
        if active != 0 && THREAD_EPOCH.try_with(Cell::get).unwrap_or(0) == active {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Forwards the caller's validated `Layout` directly to `System::alloc`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` were originally allocated by `System::alloc` via `Self::alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let active = ACTIVE_EPOCH.load(Ordering::Relaxed);
        if active != 0 && THREAD_EPOCH.try_with(Cell::get).unwrap_or(0) == active {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: `ptr` and `layout` were originally allocated by `System` and `new_size` preserves alignment invariants.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn measure_allocs_in_pool<F, R>(pool: Option<&rayon::ThreadPool>, op: F) -> (R, usize, usize)
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    let epoch = NEXT_EPOCH.fetch_add(1, Ordering::SeqCst);
    THREAD_EPOCH.with(|cell| cell.set(epoch));
    if let Some(p) = pool {
        let n = p.current_num_threads();
        let barrier = Barrier::new(n);
        p.broadcast(|_| {
            THREAD_EPOCH.with(|cell| cell.set(epoch));
            barrier.wait();
        });
    }
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    ACTIVE_EPOCH.store(epoch, Ordering::SeqCst);
    let out = match pool {
        Some(p) => p.install(op),
        None => op(),
    };
    ACTIVE_EPOCH.store(0, Ordering::SeqCst);
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    (out, count, bytes)
}

fn measure_allocs<F, R>(op: F) -> (R, usize, usize)
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    measure_allocs_in_pool(None, op)
}

fn warm_pool(pool: &rayon::ThreadPool) {
    use rayon::prelude::*;

    let n = pool.current_num_threads();
    let barrier_start = Barrier::new(n);
    pool.broadcast(|_| {
        // Materialize each worker's OS thread parker/TLS handle before entering measured windows.
        std::thread::park_timeout(std::time::Duration::ZERO);
        barrier_start.wait();
    });
    let warm: usize = pool.install(|| (0..n * 64).into_par_iter().map(|i| i + 1).sum());
    let total = n * 64;
    assert_eq!(warm, total * (total + 1) / 2);
    let barrier_done = Barrier::new(n);
    pool.broadcast(|_| {
        barrier_done.wait();
    });
}

struct TensorEntry {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct OwnedWeights {
    entries: Vec<TensorEntry>,
}

impl OwnedWeights {
    const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn upsert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.name == name) {
            slot.quant = quant;
            slot.shape = shape;
            slot.bytes = bytes;
        } else {
            self.entries.push(TensorEntry {
                name: name.to_owned(),
                quant,
                shape,
                bytes,
            });
        }
    }
}

impl ModelWeights for OwnedWeights {
    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        let entry = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!("missing tensor '{name}'"),
            })?;
        TensorView::from_bytes(&entry.name, entry.quant, &entry.shape, 0, &entry.bytes)
    }

    fn tensor_count(&self) -> usize {
        self.entries.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.bytes.len()).sum()
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "deterministic IEEE 754 binary16 helper for synthetic test tensors"
)]
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 0 {
        return sign;
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 31 {
        return sign | 0x7C00;
    }
    if half_exp <= 0 {
        return sign;
    }
    sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16)
}

const fn f32_to_bf16_bits(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

fn encode_scale_vector(quant: QuantType, len: usize, base: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 4);
    for i in 0..len {
        #[expect(clippy::cast_precision_loss, reason = "small scale dimension")]
        let val = base * (1.0 + ((i % 7) as f32 - 3.0) * 0.06);
        match quant {
            QuantType::F32 => out.extend_from_slice(&val.to_le_bytes()),
            QuantType::F16 => out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes()),
            QuantType::BF16 => out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes()),
            other => unreachable!("unsupported 1D scale quant {other}"),
        }
    }
    out
}

fn encode_float_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let cap = if quant == QuantType::F32 {
        rows * cols * 4
    } else {
        rows * cols * 2
    };
    let mut out = Vec::with_capacity(cap);
    for r in 0..rows {
        for c in 0..cols {
            #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
            let val = seed * (((r * 11 + c * 7) % 17) as f32 - 8.0) * 0.05;
            match quant {
                QuantType::F32 => out.extend_from_slice(&val.to_le_bytes()),
                QuantType::F16 => out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes()),
                QuantType::BF16 => out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes()),
                other => unreachable!("non-float quant {other}"),
            }
        }
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "deterministic block-quantized test matrix encodings"
)]
fn encode_q32_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let blocks = cols / 32;
    match quant {
        QuantType::Q4_0 => {
            let mut out = Vec::with_capacity(rows * blocks * 18);
            for r in 0..rows {
                for b in 0..blocks {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let d = seed * (1.0 + ((r + b) % 4) as f32 * 0.2);
                    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    for i in 0..16 {
                        let q0 = ((r + i + b) % 15) as u8;
                        let q1 = ((r * 3 + i + b) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
            }
            out
        }
        QuantType::Q5_0 => {
            let mut out = Vec::with_capacity(rows * blocks * 22);
            for r in 0..rows {
                for b in 0..blocks {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let d = seed * (1.0 + ((r + b) % 4) as f32 * 0.18);
                    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    let qh = ((r as u32).wrapping_mul(0x0101_0101)
                        ^ (b as u32).wrapping_mul(0x5555_5555))
                    .to_le_bytes();
                    out.extend_from_slice(&qh);
                    for i in 0..16 {
                        let q0 = ((r + i + b) % 15) as u8;
                        let q1 = ((r * 3 + i + b) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
            }
            out
        }
        QuantType::Q8_0 => {
            let mut out = Vec::with_capacity(rows * blocks * 34);
            for r in 0..rows {
                for b in 0..blocks {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let d = seed * (1.0 + ((r + b) % 4) as f32 * 0.15);
                    out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    for i in 0..32 {
                        let q = (((r + i + b) % 21) as i8) - 10;
                        out.push(q.to_ne_bytes()[0]);
                    }
                }
            }
            out
        }
        other => unreachable!("non-32-block quant {other}"),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "deterministic K-quant test matrix encodings"
)]
fn encode_k_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let blocks = cols / 256;
    match quant {
        QuantType::Q4_K_M => {
            let mut out = Vec::with_capacity(rows * blocks * 144);
            for r in 0..rows {
                for _b in 0..blocks {
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.1).to_le_bytes());
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.02).to_le_bytes());
                    for s in 0..12 {
                        out.push(((r + s) % 31) as u8 + 1);
                    }
                    for j in 0..128 {
                        let lo = ((r + j) % 15) as u8;
                        let hi = ((r * 3 + j) % 15) as u8;
                        out.push((lo & 0x0F) | ((hi & 0x0F) << 4));
                    }
                }
            }
            out
        }
        QuantType::Q5_K => {
            let mut out = Vec::with_capacity(rows * blocks * 176);
            for r in 0..rows {
                for _b in 0..blocks {
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.08).to_le_bytes());
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.015).to_le_bytes());
                    for s in 0..12 {
                        out.push(((r + s) % 31) as u8 + 1);
                    }
                    for j in 0..32 {
                        out.push(((r + j * 3) & 0xFF) as u8);
                    }
                    for j in 0..128 {
                        let lo = ((r + j) % 15) as u8;
                        let hi = ((r * 5 + j) % 15) as u8;
                        out.push((lo & 0x0F) | ((hi & 0x0F) << 4));
                    }
                }
            }
            out
        }
        QuantType::Q6_K => {
            let mut out = Vec::with_capacity(rows * blocks * 210);
            for r in 0..rows {
                for _b in 0..blocks {
                    for j in 0..128 {
                        let lo = ((r + j) % 15) as u8;
                        let hi = ((r * 2 + j) % 15) as u8;
                        out.push((lo & 0x0F) | ((hi & 0x0F) << 4));
                    }
                    for j in 0..64 {
                        out.push(((r + j * 7) & 0xFF) as u8);
                    }
                    for s in 0..16 {
                        let sc = (((r + s) % 5) as i8) + 1;
                        out.push(sc.to_ne_bytes()[0]);
                    }
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.06).to_le_bytes());
                }
            }
            out
        }
        other => unreachable!("unsupported K-quant matrix {other}"),
    }
}

fn encode_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            encode_float_matrix(quant, rows, cols, seed)
        }
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            encode_q32_matrix(quant, rows, cols, seed)
        }
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            encode_k_quant_matrix(quant, rows, cols, seed)
        }
        other => unreachable!("unsupported test matrix quant {other}"),
    }
}

fn populate_model_weights(
    weights: &mut OwnedWeights,
    recipe: &DoryRecipe,
    mat_quant: QuantType,
    scale_quant: QuantType,
    include_output: bool,
) {
    let vocab_size = recipe.vocab_size;
    let hidden = recipe.hidden_dim as usize;
    let ffn = recipe.ffn_dim as usize;
    let q_dim = (recipe.num_heads as usize) * (recipe.head_dim as usize);
    let kv_dim = (recipe.num_kv_heads as usize) * (recipe.head_dim as usize);

    weights.upsert(
        "token_embd.weight",
        mat_quant,
        vec![hidden, vocab_size],
        encode_quant_matrix(mat_quant, vocab_size, hidden, 0.2),
    );
    if include_output {
        weights.upsert(
            "output.weight",
            mat_quant,
            vec![hidden, vocab_size],
            encode_quant_matrix(mat_quant, vocab_size, hidden, 0.3),
        );
        weights.upsert(
            "output_scale.weight",
            scale_quant,
            vec![vocab_size],
            encode_scale_vector(scale_quant, vocab_size, 0.04),
        );
    }

    for p in 0..recipe.physical_layer_count() {
        let kind = recipe.sublayer_kind(p).expect("valid sublayer pattern");
        #[expect(clippy::cast_precision_loss, reason = "small physical layer index")]
        let seed = (p as f32).mul_add(0.015, 0.1);
        weights.upsert(
            &format!("blk.{p}.residual_alpha.weight"),
            scale_quant,
            vec![hidden],
            encode_scale_vector(scale_quant, hidden, recipe.alpha_init),
        );
        match kind {
            DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
                for (suffix, rows, cols, s) in [
                    ("attn_q.weight", q_dim, hidden, seed),
                    ("attn_k.weight", kv_dim, hidden, seed + 0.02),
                    ("attn_v.weight", kv_dim, hidden, seed + 0.04),
                    ("attn_gate.weight", q_dim, hidden, seed + 0.01),
                    ("attn_output.weight", hidden, q_dim, seed + 0.05),
                ] {
                    weights.upsert(
                        &format!("blk.{p}.{suffix}"),
                        mat_quant,
                        vec![cols, rows],
                        encode_quant_matrix(mat_quant, rows, cols, s),
                    );
                }
                weights.upsert(
                    &format!("blk.{p}.attn_qk_scale.weight"),
                    scale_quant,
                    vec![q_dim],
                    encode_scale_vector(scale_quant, q_dim, 0.25),
                );
                weights.upsert(
                    &format!("blk.{p}.attn_gate_scale.weight"),
                    scale_quant,
                    vec![q_dim],
                    encode_scale_vector(scale_quant, q_dim, 0.15),
                );
            }
            DorySubLayerKind::Mlp => {
                for (suffix, rows, cols, s) in [
                    ("ffn_up.weight", ffn, hidden, seed),
                    ("ffn_gate.weight", ffn, hidden, seed + 0.02),
                    ("ffn_down.weight", hidden, ffn, seed + 0.04),
                ] {
                    weights.upsert(
                        &format!("blk.{p}.{suffix}"),
                        mat_quant,
                        vec![cols, rows],
                        encode_quant_matrix(mat_quant, rows, cols, s),
                    );
                }
                weights.upsert(
                    &format!("blk.{p}.ffn_up_scale.weight"),
                    scale_quant,
                    vec![ffn],
                    encode_scale_vector(scale_quant, ffn, 0.2),
                );
                weights.upsert(
                    &format!("blk.{p}.ffn_gate_scale.weight"),
                    scale_quant,
                    vec![ffn],
                    encode_scale_vector(scale_quant, ffn, 0.18),
                );
            }
        }
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "slice length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn test_wave4_112_virtual_layer_zero_alloc_decode_per_loop_and_last_loop() {
    let _serial = TEST_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pattern_40 = "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";
    assert_eq!(pattern_40.len(), 40);

    let build_recipe = |n_loops: u32, mode: DoryKvCacheMode| -> DoryRecipe {
        let mut r = DoryRecipe::new()
            .with_layer_pattern(pattern_40)
            .with_kv_cache_mode(mode)
            .with_worker_partitions(Some(1));
        r.n_input_layers = 8;
        r.n_recurrent_layers = 24;
        r.n_output_layers = 8;
        r.n_loops = n_loops;
        r.hidden_dim = 64;
        r.ffn_dim = 64;
        r.num_heads = 4;
        r.num_kv_heads = 2;
        r.head_dim = 16;
        r.sliding_window = 4;
        r.vocab_size = 32;
        r
    };

    for mode in [DoryKvCacheMode::PerLoop, DoryKvCacheMode::LastLoop] {
        for include_output in [true, false] {
            let recipe_l1 = build_recipe(1, mode);
            let recipe_l4 = build_recipe(4, mode);
            assert_eq!(recipe_l4.virtual_layer_count(), 112);

            let req1 = recipe_l1.decode_scratch_bytes(16).unwrap();
            let req4 = recipe_l4.decode_scratch_bytes(16).unwrap();
            assert_eq!(
                req1, req4,
                "decode_scratch_bytes must stay O(1) across n_loops=1..=4 for {mode:?}"
            );

            let mut weights = OwnedWeights::new();
            populate_model_weights(
                &mut weights,
                &recipe_l4,
                QuantType::Q8_0,
                QuantType::F16,
                include_output,
            );
            let cache = Arc::new(recipe_l4.build_physical_cache(&weights).unwrap());
            let cached = recipe_l4.with_physical_cache(cache);

            let kv_len = cached.attention_kv_layer_count().unwrap() * 2 * 16 * 32;
            let mut kv = vec![0.0_f32; kv_len];
            let mut logits = vec![0.0_f32; 32];
            let mut ws = ScratchWorkspace::new(req4).unwrap();
            let mut ctx = ExecutionContext::new(&[], 16, &mut logits, &mut kv)
                .unwrap()
                .with_scratch(&mut ws);

            for (step, &tok) in [0_u32, 7, 15, 31].iter().enumerate() {
                ctx.prepare_decode(tok).unwrap();
                let (res, allocs, bytes) =
                    measure_allocs(|| cached.decode_step(&mut ctx, &weights));
                res.unwrap();
                assert_eq!(
                    (allocs, bytes),
                    (0, 0),
                    "112-layer {mode:?} include_output={include_output} step {step} allocated"
                );
                assert_eq!(ctx.scratch().unwrap().used_bytes(), 0);
                assert!(ctx.scratch().unwrap().peak_used_bytes() <= req4);
            }
        }
    }
}

#[test]
fn test_wave4_worker_partitions_1_4_8_bit_exact_and_zero_alloc() {
    use rayon::prelude::*;

    let _serial = TEST_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let pool_4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool_8 = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    // Barrier-synchronize all pool workers through startup/parker initialization before measuring.
    warm_pool(&pool_4);
    warm_pool(&pool_8);

    // Self-check that epoch-scoped tracking still captures genuine allocations on both the calling
    // thread and active pool workers.
    let (_, call_allocs, call_bytes) = measure_allocs(|| vec![1_u8; 64].len());
    assert!(
        call_allocs >= 1 && call_bytes >= 64,
        "calling-thread heap allocation must be captured ({call_allocs}, {call_bytes})"
    );
    let (_, pool_allocs, pool_bytes) = measure_allocs_in_pool(Some(&pool_4), || {
        (0..4_usize)
            .into_par_iter()
            .map(|_| vec![1_u8; 64].len())
            .sum::<usize>()
    });
    assert!(
        pool_allocs >= 4 && pool_bytes >= 256,
        "active pool worker heap allocations must be captured ({pool_allocs}, {pool_bytes})"
    );

    let configs: &[(QuantType, u32, usize)] = &[
        (QuantType::Q5_K, 256, 64),
        (QuantType::Q6_K, 256, 64),
        (QuantType::Q8_0, 256, 64),
        (QuantType::Q4_K_M, 256, 64),
        (QuantType::Q4_0, 256, 64),
        (QuantType::Q5_0, 64, 64),
        (QuantType::F32, 64, 64),
        (QuantType::F16, 64, 64),
        (QuantType::BF16, 64, 64),
    ];

    for &(quant, dim, vocab) in configs {
        let mut base = DoryRecipe::new().with_layer_pattern("S*-g");
        base.n_input_layers = 1;
        base.n_recurrent_layers = 2;
        base.n_output_layers = 1;
        base.n_loops = 2;
        base.hidden_dim = dim;
        base.ffn_dim = dim;
        base.num_heads = 4;
        base.num_kv_heads = 2;
        base.head_dim = dim / 4;
        base.sliding_window = 3;
        base.vocab_size = vocab;

        let mut weights = OwnedWeights::new();
        populate_model_weights(&mut weights, &base, quant, QuantType::F16, true);
        let cache = Arc::new(base.build_physical_cache(&weights).unwrap());
        let req = base.decode_scratch_bytes(8).unwrap();
        let kv_dim = (base.num_kv_heads * base.head_dim) as usize;
        let kv_len = base.attention_kv_layer_count().unwrap() * 2 * 8 * kv_dim;

        let mut ref_logits = Vec::new();
        let mut run_partition = |workers: usize, pool: Option<&rayon::ThreadPool>| {
            let recipe = base
                .clone()
                .with_worker_partitions(Some(workers))
                .with_physical_cache(Arc::clone(&cache));
            let mut ws = ScratchWorkspace::new(req).unwrap();
            let mut kv = vec![0.0_f32; kv_len];
            let mut logits = vec![0.0_f32; vocab];
            let mut ctx = ExecutionContext::new(&[], 8, &mut logits, &mut kv)
                .unwrap()
                .with_scratch(&mut ws);

            for tok in [1_u32, 17, 42] {
                ctx.prepare_decode(tok).unwrap();
                let (res, allocs, bytes) =
                    measure_allocs_in_pool(pool, || recipe.decode_step(&mut ctx, &weights));
                res.unwrap();
                assert_eq!(
                    (allocs, bytes),
                    (0, 0),
                    "{quant} workers={workers} allocated during decode_step"
                );
            }
            assert_eq!(ctx.scratch().unwrap().used_bytes(), 0);
            if ref_logits.is_empty() {
                ref_logits = logits;
            } else {
                assert_eq!(
                    max_abs_diff(&ref_logits, &logits),
                    0.0,
                    "{quant} workers={workers} must match 1-worker logits bit-exact"
                );
            }
        };

        run_partition(1, None);
        run_partition(4, Some(&pool_4));
        run_partition(8, Some(&pool_8));
    }
}

#[test]
fn test_wave4_256d_multiquant_decode_with_scratch_vs_prefill_parity() {
    let _serial = TEST_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokens = [0_u32, 3, 5, 2, 4];
    for quant in [
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
        QuantType::Q4_K_M,
        QuantType::Q4_0,
    ] {
        let mut recipe = DoryRecipe::new()
            .with_layer_pattern("S*g-")
            .with_worker_partitions(Some(1));
        recipe.n_input_layers = 1;
        recipe.n_recurrent_layers = 2;
        recipe.n_output_layers = 1;
        recipe.n_loops = 3;
        recipe.hidden_dim = 256;
        recipe.ffn_dim = 256;
        recipe.num_heads = 4;
        recipe.num_kv_heads = 2;
        recipe.head_dim = 64;
        recipe.sliding_window = 3;
        recipe.vocab_size = 8;

        let mut weights = OwnedWeights::new();
        populate_model_weights(&mut weights, &recipe, quant, QuantType::F16, true);
        let cache = Arc::new(recipe.build_physical_cache(&weights).unwrap());
        let cached = recipe.clone().with_physical_cache(cache);

        let n_ctx = 16;
        let kv_len = cached.attention_kv_layer_count().unwrap() * 2 * n_ctx * 128;

        let mut kv_batch = vec![0.0_f32; kv_len];
        let mut logits_batch = vec![0.0_f32; 8];
        let mut ctx_batch =
            ExecutionContext::new(&tokens, n_ctx, &mut logits_batch, &mut kv_batch).unwrap();
        cached.prefill_step(&mut ctx_batch, &weights).unwrap();

        let req = cached.decode_scratch_bytes(n_ctx).unwrap();
        let mut ws = ScratchWorkspace::new(req).unwrap();
        let mut kv_seq = vec![0.0_f32; kv_len];
        let mut logits_seq = vec![0.0_f32; 8];
        let mut ctx_seq = ExecutionContext::new(&[], n_ctx, &mut logits_seq, &mut kv_seq)
            .unwrap()
            .with_scratch(&mut ws);

        for &tok in &tokens {
            ctx_seq.prepare_decode(tok).unwrap();
            let (res, allocs, bytes) =
                measure_allocs(|| cached.decode_step(&mut ctx_seq, &weights));
            res.unwrap();
            assert_eq!((allocs, bytes), (0, 0), "{quant} decode allocated");
        }
        assert!(
            max_abs_diff(ctx_batch.logits(), ctx_seq.logits()) < 1e-4,
            "{quant} batched vs zero-alloc decode logits diff"
        );
        assert!(
            max_abs_diff(ctx_batch.kv_cache(), ctx_seq.kv_cache()) < 1e-4,
            "{quant} batched vs zero-alloc decode KV cache diff"
        );
    }
}
