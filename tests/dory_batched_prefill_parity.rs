//! Wave 3 batched prefill integration parity tests for `jero-arch-dory`:
//! - 40-physical-layer (`112`-virtual-layer) 3-segment batched `[M, hidden_dim]` prefill across
//!   `F32`, `F16`, and `BF16` weights/scales (`PerLoop` and chunked `start_position`).
//! - `hidden_dim = 256` multi-quant (`Q5_K`, `Q6_K`, `Q8_0`, `Q4_K_M`, `Q4_0`) `QuantizedActQ8` +
//!   `gemm_repacked_rows` + `gqa_causal_attention_prefill_f32` vs sequential `decode_step` parity.
//! - Sliding-window causal cutoff and final-row-only `populate_dory_logits` invariants.

use std::sync::Arc;

use jero_arch_dory::{DoryKvCacheMode, DoryRecipe, DorySubLayerKind};
use jero_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

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

#[expect(
    clippy::cast_possible_truncation,
    reason = "deterministic modular quantized test matrix encodings"
)]
fn encode_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    match quant {
        QuantType::F32 => {
            let mut out = Vec::with_capacity(rows * cols * 4);
            for r in 0..rows {
                for c in 0..cols {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let val = seed * (((r * 11 + c * 7) % 17) as f32 - 8.0) * 0.05;
                    out.extend_from_slice(&val.to_le_bytes());
                }
            }
            out
        }
        QuantType::F16 => {
            let mut out = Vec::with_capacity(rows * cols * 2);
            for r in 0..rows {
                for c in 0..cols {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let val = seed * (((r * 11 + c * 7) % 17) as f32 - 8.0) * 0.05;
                    out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
                }
            }
            out
        }
        QuantType::BF16 => {
            let mut out = Vec::with_capacity(rows * cols * 2);
            for r in 0..rows {
                for c in 0..cols {
                    #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
                    let val = seed * (((r * 11 + c * 7) % 17) as f32 - 8.0) * 0.05;
                    out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes());
                }
            }
            out
        }
        QuantType::Q4_0 => {
            let blocks = cols / 32;
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
        QuantType::Q8_0 => {
            let blocks = cols / 32;
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
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            encode_k_quant_matrix(quant, rows, cols, seed)
        }
        other => unreachable!("unsupported test matrix quant {other}"),
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

fn populate_model_weights(
    weights: &mut OwnedWeights,
    recipe: &DoryRecipe,
    mat_quant: QuantType,
    scale_quant: QuantType,
    vocab_size: usize,
) {
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

fn assert_batched_chunked_and_decode_parity(
    recipe: &DoryRecipe,
    weights: &OwnedWeights,
    tokens: &[u32],
    (split_a, split_b): (usize, usize),
    n_ctx: usize,
    next_decode_tok: u32,
    tag: &str,
) {
    let vocab = recipe.vocab_size;
    let kv_slots = recipe.attention_kv_layer_count().unwrap();
    let kv_dim = (recipe.num_kv_heads * recipe.head_dim) as usize;
    let kv_len = kv_slots * 2 * n_ctx * kv_dim;
    let cached_recipe = recipe
        .clone()
        .with_physical_cache(Arc::new(recipe.build_physical_cache(weights).unwrap()));

    // 1. Single batched prefill over tokens[0..M].
    let mut kv_batch = vec![0.0_f32; kv_len];
    let mut logits_batch = vec![0.0_f32; vocab];
    let mut ctx_batch =
        ExecutionContext::new(tokens, n_ctx, &mut logits_batch, &mut kv_batch).unwrap();
    cached_recipe.prefill_step(&mut ctx_batch, weights).unwrap();
    assert_eq!(ctx_batch.position(), tokens.len());

    // 2. Three-chunk start_position prefill across [0..split_a], [split_a..split_b], [split_b..M].
    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut logits_chunk = vec![0.0_f32; vocab];
    let mut ctx_c1 = ExecutionContext::new_prefill_chunk(
        &tokens[..split_a],
        n_ctx,
        0,
        false,
        &mut [],
        &mut kv_chunk,
    )
    .unwrap();
    cached_recipe.prefill_step(&mut ctx_c1, weights).unwrap();

    let mut ctx_c2 = ExecutionContext::new_prefill_chunk(
        &tokens[split_a..split_b],
        n_ctx,
        split_a,
        false,
        &mut [],
        &mut kv_chunk,
    )
    .unwrap();
    cached_recipe.prefill_step(&mut ctx_c2, weights).unwrap();

    let mut ctx_c3 = ExecutionContext::new_prefill_chunk(
        &tokens[split_b..],
        n_ctx,
        split_b,
        true,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .unwrap();
    cached_recipe.prefill_step(&mut ctx_c3, weights).unwrap();
    assert_eq!(ctx_c3.position(), tokens.len());

    // 3. Sequential token-by-token decode_step oracle.
    let mut kv_seq = vec![0.0_f32; kv_len];
    let mut logits_seq = vec![0.0_f32; vocab];
    let mut ctx_seq = ExecutionContext::new(&[], n_ctx, &mut logits_seq, &mut kv_seq).unwrap();
    for &tok in tokens {
        ctx_seq.prepare_decode(tok).unwrap();
        cached_recipe.decode_step(&mut ctx_seq, weights).unwrap();
    }

    assert!(
        max_abs_diff(ctx_batch.logits(), ctx_seq.logits()) < 1e-4,
        "{tag}: batched vs sequential logits diff"
    );
    assert!(
        max_abs_diff(ctx_c3.logits(), ctx_seq.logits()) < 1e-4,
        "{tag}: chunked vs sequential logits diff"
    );
    assert!(
        max_abs_diff(ctx_batch.kv_cache(), ctx_seq.kv_cache()) < 1e-4,
        "{tag}: batched vs sequential KV cache diff"
    );
    assert!(
        max_abs_diff(ctx_c3.kv_cache(), ctx_seq.kv_cache()) < 1e-4,
        "{tag}: chunked vs sequential KV cache diff"
    );

    // 4. Post-prefill decode step must also align < 1e-4.
    ctx_batch.prepare_decode(next_decode_tok).unwrap();
    ctx_seq.prepare_decode(next_decode_tok).unwrap();
    cached_recipe.decode_step(&mut ctx_batch, weights).unwrap();
    cached_recipe.decode_step(&mut ctx_seq, weights).unwrap();
    assert!(
        max_abs_diff(&logits_batch, &logits_seq) < 1e-4,
        "{tag}: post-prefill decode logits diff"
    );
}

#[test]
fn test_dory_40_layer_looped_batched_prefill_f32_f16_bf16_and_chunked_parity() {
    let pattern_40 = "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";
    assert_eq!(pattern_40.len(), 40);

    let mut recipe = DoryRecipe::new().with_layer_pattern(pattern_40);
    recipe.n_input_layers = 8;
    recipe.n_recurrent_layers = 24;
    recipe.n_output_layers = 8;
    recipe.n_loops = 4;
    recipe.hidden_dim = 16;
    recipe.ffn_dim = 16;
    recipe.num_heads = 4;
    recipe.num_kv_heads = 2;
    recipe.head_dim = 4;
    recipe.sliding_window = 4;
    recipe.vocab_size = 8;
    assert_eq!(recipe.physical_layer_count(), 40);
    assert_eq!(recipe.virtual_layer_count(), 112);

    let tokens = [0_u32, 3, 1, 5, 2, 7];
    for precision in [QuantType::F32, QuantType::F16, QuantType::BF16] {
        let mut weights = OwnedWeights::new();
        populate_model_weights(&mut weights, &recipe, precision, precision, 8);
        assert_batched_chunked_and_decode_parity(
            &recipe,
            &weights,
            &tokens,
            (2, 5),
            16,
            4,
            &format!("40L-{precision}"),
        );
    }
}

#[test]
fn test_dory_256d_five_quants_batched_gemm_vs_chunked_and_decode_parity() {
    let quants = [
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
        QuantType::Q4_K_M,
        QuantType::Q4_0,
    ];
    let tokens = [0_u32, 3, 5, 2, 4];

    for quant in quants {
        let mut recipe = DoryRecipe::new().with_layer_pattern("S*g-");
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
        populate_model_weights(&mut weights, &recipe, quant, QuantType::F16, 8);
        assert_batched_chunked_and_decode_parity(
            &recipe,
            &weights,
            &tokens,
            (2, 4),
            16,
            1,
            &format!("256d-{quant}"),
        );
    }
}

fn last_loop_256d_recipe(pattern: &'static str) -> DoryRecipe {
    let mut recipe = DoryRecipe::new()
        .with_layer_pattern(pattern)
        .with_kv_cache_mode(DoryKvCacheMode::LastLoop);
    recipe.n_input_layers = 1;
    recipe.n_recurrent_layers = 1;
    recipe.n_output_layers = 1;
    recipe.n_loops = 3;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.num_heads = 4;
    recipe.num_kv_heads = 2;
    recipe.head_dim = 64;
    recipe.sliding_window = 3;
    recipe.vocab_size = 8;
    recipe
}

#[test]
fn test_dory_256d_last_loop_prefix_and_final_slot_parity() {
    let tokens = [0_u32, 3, 5, 2, 4];
    let ll_mlp_recipe = last_loop_256d_recipe("S-g");

    let mut ll_mlp_weights = OwnedWeights::new();
    populate_model_weights(
        &mut ll_mlp_weights,
        &ll_mlp_recipe,
        QuantType::Q8_0,
        QuantType::BF16,
        8,
    );
    assert_batched_chunked_and_decode_parity(
        &ll_mlp_recipe,
        &ll_mlp_weights,
        &tokens,
        (2, 4),
        16,
        1,
        "256d-LastLoop-S-g",
    );

    let ll_recipe = last_loop_256d_recipe("S*g");

    let mut ll_weights = OwnedWeights::new();
    populate_model_weights(
        &mut ll_weights,
        &ll_recipe,
        QuantType::Q8_0,
        QuantType::BF16,
        8,
    );
    let ll_cached = ll_recipe.clone().with_physical_cache(Arc::new(
        ll_recipe.build_physical_cache(&ll_weights).unwrap(),
    ));

    let n_ctx = 16_usize;
    let kv_dim = 2 * 64;
    let slots = ll_recipe.attention_kv_layer_count().unwrap();
    let kv_len = slots * 2 * n_ctx * kv_dim;
    let ll_tokens = [0_u32, 3, 5, 2];

    let mut kv_single = vec![0.0_f32; kv_len];
    let mut logits_single = vec![0.0_f32; 8];
    let mut ctx_single =
        ExecutionContext::new(&ll_tokens, n_ctx, &mut logits_single, &mut kv_single).unwrap();
    ll_cached
        .prefill_step(&mut ctx_single, &ll_weights)
        .unwrap();

    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut c1 = ExecutionContext::new_prefill_chunk(
        &ll_tokens[..2],
        n_ctx,
        0,
        false,
        &mut [],
        &mut kv_chunk,
    )
    .unwrap();
    ll_cached.prefill_step(&mut c1, &ll_weights).unwrap();
    let prefix_snap = kv_chunk.clone();
    let mut logits_chunk = vec![0.0_f32; 8];
    let mut c2 = ExecutionContext::new_prefill_chunk(
        &ll_tokens[2..],
        n_ctx,
        2,
        true,
        &mut logits_chunk,
        &mut kv_chunk,
    )
    .unwrap();
    ll_cached.prefill_step(&mut c2, &ll_weights).unwrap();

    for slot in 0..slots {
        let base = slot * 2 * n_ctx * kv_dim;
        assert_eq!(
            &kv_chunk[base..base + 2 * kv_dim],
            &prefix_snap[base..base + 2 * kv_dim]
        );
        assert!(
            max_abs_diff(
                &kv_chunk[base..base + 2 * kv_dim],
                &kv_single[base..base + 2 * kv_dim]
            ) < 1e-4
        );
        let tail = &kv_chunk[base + 2 * kv_dim..base + 4 * kv_dim];
        assert!(tail.iter().all(|v| v.is_finite()) && tail.iter().any(|&v| v.abs() > 1e-4));
    }
    assert!(logits_chunk.iter().all(|v| v.is_finite()));
}

#[test]
fn test_dory_batched_prefill_swa_cutoff_and_final_row_logits_only() {
    let mut recipe = DoryRecipe::new().with_layer_pattern("S");
    recipe.n_input_layers = 1;
    recipe.n_recurrent_layers = 0;
    recipe.n_output_layers = 0;
    recipe.n_loops = 1;
    recipe.hidden_dim = 8;
    recipe.ffn_dim = 8;
    recipe.num_heads = 2;
    recipe.num_kv_heads = 1;
    recipe.head_dim = 4;
    recipe.sliding_window = 3;
    recipe.vocab_size = 8;

    let mut weights = OwnedWeights::new();
    populate_model_weights(&mut weights, &recipe, QuantType::F32, QuantType::F32, 8);
    let cached = recipe
        .clone()
        .with_physical_cache(Arc::new(recipe.build_physical_cache(&weights).unwrap()));

    let n_ctx = 16_usize;
    let kv_len = recipe.attention_kv_layer_count().unwrap() * 2 * n_ctx * 4;
    let prompt_a = [0_u32, 1, 2, 3, 4];
    let prompt_b = [7_u32, 1, 2, 3, 4];

    let mut kv_a = vec![0.0_f32; kv_len];
    let mut logits_a = vec![0.0_f32; 8];
    let mut ctx_a = ExecutionContext::new(&prompt_a, n_ctx, &mut logits_a, &mut kv_a).unwrap();
    cached.prefill_step(&mut ctx_a, &weights).unwrap();

    let mut kv_b = vec![0.0_f32; kv_len];
    let mut logits_b = vec![0.0_f32; 8];
    let mut ctx_b = ExecutionContext::new(&prompt_b, n_ctx, &mut logits_b, &mut kv_b).unwrap();
    cached.prefill_step(&mut ctx_b, &weights).unwrap();

    // Token at pos=4 attends strictly to [2, 3, 4] under SWA=3, so mutating pos=0 leaves logits identical.
    assert!(
        max_abs_diff(&logits_a, &logits_b) < 1e-6,
        "SWA batched prefill leaked pos=0 outside sliding_window=3"
    );
    assert!(
        max_abs_diff(&kv_a[..4], &kv_b[..4]) > 1e-3,
        "KV slot pos=0 must reflect token change even though pos=4 masks it"
    );

    // Chunk ending with only row M-1 ([4]) after [0, 1, 2, 3] produces identical logits.
    let mut kv_chunk = vec![0.0_f32; kv_len];
    let mut c1 = ExecutionContext::new_prefill_chunk(
        &prompt_a[..4],
        n_ctx,
        0,
        false,
        &mut [],
        &mut kv_chunk,
    )
    .unwrap();
    cached.prefill_step(&mut c1, &weights).unwrap();
    let mut logits_c2 = vec![0.0_f32; 8];
    let mut c2 = ExecutionContext::new_prefill_chunk(
        &prompt_a[4..],
        n_ctx,
        4,
        true,
        &mut logits_c2,
        &mut kv_chunk,
    )
    .unwrap();
    cached.prefill_step(&mut c2, &weights).unwrap();
    assert!(max_abs_diff(&logits_a, &logits_c2) < 1e-6);
}
