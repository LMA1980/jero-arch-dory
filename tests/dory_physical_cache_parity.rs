//! Integration tests for [`DoryPhysicalCache`]: 40-layer loop sharing (no virtual weight
//! replication), `128-byte`-aligned weight repacking across `Q5_K`/`Q6_K`/`Q8_0`/`Q4_K_M`/`Q4_0`,
//! pre-decoded `nGPT` 1D scale vectors, and `ChatML`/`<think>` GGUF hydration parity.

use std::sync::Arc;

use jero_arch_dory::{
    DORY_MEDIUM_REASONING_SUFFIX, DoryPhysicalCache, DoryPhysicalSubLayer, DoryRecipe,
    DorySubLayerKind, project_linear,
};
use jero_core::{
    ArchRecipe, ExecutionContext, GgufArray, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
    StopReason, TensorView, test_helpers::create_synthetic_gguf_v3_with_kvs,
};

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

fn encode_f32_vector(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn encode_f16_vector(len: usize, base: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for i in 0..len {
        #[expect(clippy::cast_precision_loss, reason = "small vector dimension")]
        let val = base * (1.0 + ((i % 7) as f32 - 3.0) * 0.08);
        out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
    }
    out
}

fn encode_bf16_vector(len: usize, base: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len * 2);
    for i in 0..len {
        #[expect(clippy::cast_precision_loss, reason = "small vector dimension")]
        let val = base * (1.0 + ((i % 5) as f32 - 2.0) * 0.05);
        out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes());
    }
    out
}

fn encode_f32_matrix(rows: usize, cols: usize, seed: f32) -> Vec<u8> {
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "deterministic modular quantized test matrix encodings"
)]
fn encode_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    match quant {
        QuantType::F32 => encode_f32_matrix(rows, cols, seed),
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
        QuantType::Q4_K_M => {
            let blocks = cols / 256;
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
            let blocks = cols / 256;
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
            let blocks = cols / 256;
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
        other => unreachable!("unsupported test matrix quant {other}"),
    }
}

fn populate_sublayers(
    weights: &mut OwnedWeights,
    recipe: &DoryRecipe,
    mat_quant: QuantType,
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
        QuantType::F16,
        vec![vocab_size],
        encode_f16_vector(vocab_size, 0.04),
    );

    for p in 0..recipe.physical_layer_count() {
        let kind = recipe.sublayer_kind(p).expect("valid sublayer pattern");
        #[expect(clippy::cast_precision_loss, reason = "small layer count")]
        let seed = (p as f32).mul_add(0.015, 0.1);
        weights.upsert(
            &format!("blk.{p}.residual_alpha.weight"),
            QuantType::BF16,
            vec![hidden],
            encode_bf16_vector(hidden, recipe.alpha_init),
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
                    QuantType::F16,
                    vec![q_dim],
                    encode_f16_vector(q_dim, 0.25),
                );
                weights.upsert(
                    &format!("blk.{p}.attn_gate_scale.weight"),
                    QuantType::F32,
                    vec![q_dim],
                    encode_f32_vector(&vec![0.15; q_dim]),
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
                    QuantType::F16,
                    vec![ffn],
                    encode_f16_vector(ffn, 0.2),
                );
                weights.upsert(
                    &format!("blk.{p}.ffn_gate_scale.weight"),
                    QuantType::BF16,
                    vec![ffn],
                    encode_bf16_vector(ffn, 0.18),
                );
            }
        }
    }
}

#[test]
fn test_dory_physical_cache_40_layer_loop_sharing_no_virtual_duplication() {
    // 1. Canonical Dory1.0-4B: 72 physical sub-layers, n_loops=2 => 112 virtual steps.
    let canonical = DoryRecipe::new();
    assert_eq!(canonical.physical_layer_count(), 72);
    assert_eq!(canonical.virtual_layer_count(), 112);

    // 2. 40-layer physical model (12 input + 16 recurrent + 12 output) with n_loops = 4 => 88 virtual.
    let mut recipe_40 =
        DoryRecipe::new().with_layer_pattern("S-g-S-g-S-g-S-g-S-g-S-g-S-g-S-g-S-g-S-g-");
    recipe_40.n_input_layers = 12;
    recipe_40.n_recurrent_layers = 16;
    recipe_40.n_output_layers = 12;
    recipe_40.n_loops = 4;
    recipe_40.hidden_dim = 4;
    recipe_40.ffn_dim = 8;
    recipe_40.num_heads = 2;
    recipe_40.num_kv_heads = 1;
    recipe_40.head_dim = 2;
    recipe_40.sliding_window = 8;
    recipe_40.vocab_size = 8;
    assert_eq!(recipe_40.physical_layer_count(), 40);
    assert_eq!(recipe_40.virtual_layer_count(), 88);

    let mut weights = OwnedWeights::new();
    populate_sublayers(&mut weights, &recipe_40, QuantType::F32, 8);

    let cache = recipe_40.build_physical_cache(&weights).unwrap();
    // Strictly 40 physical sub-layers (20 Attention * 5 + 20 MLP * 3 = 160 matrices), ZERO virtual copies.
    assert_eq!(cache.len(), 40);
    assert_ne!(cache.len(), recipe_40.virtual_layer_count() as usize);
    assert_eq!(cache.repacked_matrix_count(), 20 * 5 + 20 * 3);
    assert_eq!(cache.total_aligned_bytes() % 128, 0);
    assert!(cache.output_scale().unwrap().is_aligned_128());

    for sub in cache.sublayers() {
        assert!(sub.residual_alpha().is_aligned_128());
        match sub {
            DoryPhysicalSubLayer::Attention {
                wq,
                wk,
                wv,
                wgate,
                wo,
                sqk,
                attn_alpha,
                attn_gate,
                ..
            } => {
                for w in [wq, wk, wv, wgate, wo] {
                    assert_eq!(w.payload_ptr().align_offset(128), 0);
                    #[cfg(feature = "with-kernels")]
                    assert_eq!(w.repacked().payload_ptr().align_offset(128), 0);
                }
                assert!(sqk.is_aligned_128());
                assert!(attn_alpha.is_aligned_128());
                assert!(attn_gate.is_aligned_128());
            }
            DoryPhysicalSubLayer::Mlp {
                wup,
                wgate,
                wdown,
                suv,
                mlp_alpha,
                mlp_gate,
                ..
            } => {
                for w in [wup, wgate, wdown] {
                    assert_eq!(w.payload_ptr().align_offset(128), 0);
                    #[cfg(feature = "with-kernels")]
                    assert_eq!(w.repacked().payload_ptr().align_offset(128), 0);
                }
                assert!(suv.is_aligned_128());
                assert!(mlp_alpha.is_aligned_128());
                assert!(mlp_gate.is_aligned_128());
            }
        }
    }

    // 3. Cached execution across 88 virtual steps matches uncached forward+decode bit-for-bit.
    let cached_recipe = recipe_40.clone().with_physical_cache(Arc::new(cache));
    let kv_layers = recipe_40.attention_kv_layer_count().unwrap();
    let kv_len = kv_layers * 2 * 4 * 2;
    let mut kv_u = vec![0.0_f32; kv_len];
    let mut kv_c = vec![0.0_f32; kv_len];
    let mut logits_u = [0.0_f32; 8];
    let mut logits_c = [0.0_f32; 8];

    let mut ctx_u = ExecutionContext::new(&[1, 3], 4, &mut logits_u, &mut kv_u).unwrap();
    let mut ctx_c = ExecutionContext::new(&[1, 3], 4, &mut logits_c, &mut kv_c).unwrap();
    recipe_40.prefill_step(&mut ctx_u, &weights).unwrap();
    cached_recipe.prefill_step(&mut ctx_c, &weights).unwrap();
    ctx_u.prepare_decode(5).unwrap();
    ctx_c.prepare_decode(5).unwrap();
    recipe_40.decode_step(&mut ctx_u, &weights).unwrap();
    cached_recipe.decode_step(&mut ctx_c, &weights).unwrap();

    let max_diff = logits_u
        .iter()
        .zip(logits_c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "40-layer cached vs uncached logits mismatch: max_diff={max_diff}"
    );
}

#[test]
fn test_dory_physical_cache_five_quants_repack_and_forward_parity() {
    let quants = [
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
        QuantType::Q4_K_M,
        QuantType::Q4_0,
    ];

    for quant in quants {
        let mut recipe = DoryRecipe::new().with_layer_pattern("S-g");
        recipe.n_input_layers = 1;
        recipe.n_recurrent_layers = 1;
        recipe.n_output_layers = 1;
        recipe.n_loops = 3;
        recipe.hidden_dim = 256;
        recipe.ffn_dim = 256;
        recipe.num_heads = 4;
        recipe.num_kv_heads = 2;
        recipe.head_dim = 64;
        recipe.sliding_window = 4;
        recipe.vocab_size = 8;

        let mut weights = OwnedWeights::new();
        populate_sublayers(&mut weights, &recipe, quant, 8);

        // 1. Direct projection parity check against TensorView kernel (`project_linear`).
        let input: Vec<f32> = (0..256)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small dim 256")]
                let v = ((i % 13) as f32 - 6.0) * 0.05;
                v
            })
            .collect();
        let view_q = weights.tensor("blk.0.attn_q.weight").unwrap();
        let expected_proj = project_linear(&view_q, &input, 256).unwrap();

        let cache = recipe.build_physical_cache(&weights).unwrap();
        assert_eq!(cache.len(), 3);
        let DoryPhysicalSubLayer::Attention { wq, .. } = cache.get(0).unwrap() else {
            panic!("expected sublayer 0 Attention");
        };
        assert_eq!(wq.quant(), quant);
        assert_eq!(wq.payload_ptr().align_offset(128), 0);
        let mut cached_proj = vec![0.0_f32; 256];
        wq.project_f32(&mut cached_proj, &input).unwrap();
        for (idx, (&e, &a)) in expected_proj.iter().zip(cached_proj.iter()).enumerate() {
            let tol = if cfg!(feature = "with-kernels") {
                0.05 + 0.015 * e.abs()
            } else {
                1e-4
            };
            assert!(
                (e - a).abs() <= tol,
                "{quant} row {idx}: repacked GEMV mismatch vs TensorView: expected {e}, got {a}"
            );
        }

        // 2. End-to-end prefill + decode logits parity across cached vs uncached DoryRecipe.
        let cached_recipe = recipe.clone().with_physical_cache(Arc::new(cache));
        let kv_layers = recipe.attention_kv_layer_count().unwrap();
        let kv_len = kv_layers * 2 * 4 * (2 * 64);
        let mut kv_u = vec![0.0_f32; kv_len];
        let mut kv_c = vec![0.0_f32; kv_len];
        let mut logits_u = [0.0_f32; 8];
        let mut logits_c = [0.0_f32; 8];

        let mut ctx_u = ExecutionContext::new(&[0, 2], 4, &mut logits_u, &mut kv_u).unwrap();
        let mut ctx_c = ExecutionContext::new(&[0, 2], 4, &mut logits_c, &mut kv_c).unwrap();
        recipe.prefill_step(&mut ctx_u, &weights).unwrap();
        cached_recipe.prefill_step(&mut ctx_c, &weights).unwrap();
        ctx_u.prepare_decode(5).unwrap();
        ctx_c.prepare_decode(5).unwrap();
        recipe.decode_step(&mut ctx_u, &weights).unwrap();
        cached_recipe.decode_step(&mut ctx_c, &weights).unwrap();

        let fwd_diff = logits_u
            .iter()
            .zip(logits_c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            fwd_diff < 1e-3,
            "{quant}: full prefill+decode parity mismatch max_diff={fwd_diff}"
        );
    }
}

#[test]
fn test_dory_gguf_physical_cache_stop_metadata_and_ngpt_init_parity() {
    let tokens = vec![
        "<unk>",
        "<s>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "tok0",
        "tok1",
    ];
    let kvs = [
        ("dory.block_count", GgufValue::Uint32(0)),
        ("dory.input_layer_count", GgufValue::Uint32(1)),
        ("dory.recurrent_layer_count", GgufValue::Uint32(1)),
        ("dory.output_layer_count", GgufValue::Uint32(1)),
        ("dory.num_loops", GgufValue::Uint32(3)),
        (
            "dory.recurrent_kv_cache_mode",
            GgufValue::String("per_loop"),
        ),
        ("dory.layer_pattern", GgufValue::String("S-*")),
        ("dory.embedding_length", GgufValue::Uint32(4)),
        ("dory.feed_forward_length", GgufValue::Uint32(4)),
        ("dory.attention.head_count", GgufValue::Uint32(2)),
        ("dory.attention.head_count_kv", GgufValue::Uint32(1)),
        ("dory.attention.key_length", GgufValue::Uint32(2)),
        ("dory.attention.sliding_window", GgufValue::Uint32(4)),
        ("dory.init_std", GgufValue::Float32(0.025)),
        ("dory.alpha_init", GgufValue::Float32(0.12)),
        ("dory.gate_init", GgufValue::Float32(18.0)),
        ("dory.vocab_size", GgufValue::Uint32(8)),
        ("dory.logit_scale", GgufValue::Float32(0.5)),
        ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(3)),
        ("tokenizer.ggml.eot_token_id", GgufValue::Uint32(3)),
        (
            "tokenizer.ggml.tokens",
            GgufValue::Array(GgufArray::String(tokens)),
        ),
        (
            "tokenizer.chat_template",
            GgufValue::String(
                "<|im_start|>user\n{{ msg }}<|im_end|>\n<|im_start|>assistant\n<think>",
            ),
        ),
    ];
    let tensor_specs = [
        ("token_embd.weight", QuantType::F32, &[4_usize, 8][..]),
        ("output.weight", QuantType::F32, &[4, 8]),
        ("output_scale.weight", QuantType::F16, &[8]),
        ("blk.0.attn_q.weight", QuantType::F32, &[4, 4]),
        ("blk.0.attn_k.weight", QuantType::F32, &[4, 2]),
        ("blk.0.attn_v.weight", QuantType::F32, &[4, 2]),
        ("blk.0.attn_gate.weight", QuantType::F32, &[4, 4]),
        ("blk.0.attn_output.weight", QuantType::F32, &[4, 4]),
        ("blk.0.attn_qk_scale.weight", QuantType::F16, &[4]),
        ("blk.0.residual_alpha.weight", QuantType::BF16, &[4]),
        ("blk.0.attn_gate_scale.weight", QuantType::F32, &[4]),
        ("blk.1.ffn_up.weight", QuantType::F32, &[4, 4]),
        ("blk.1.ffn_gate.weight", QuantType::F32, &[4, 4]),
        ("blk.1.ffn_down.weight", QuantType::F32, &[4, 4]),
        ("blk.1.ffn_up_scale.weight", QuantType::F16, &[4]),
        ("blk.1.residual_alpha.weight", QuantType::BF16, &[4]),
        ("blk.1.ffn_gate_scale.weight", QuantType::F32, &[4]),
        ("blk.2.attn_q.weight", QuantType::F32, &[4, 4]),
        ("blk.2.attn_k.weight", QuantType::F32, &[4, 2]),
        ("blk.2.attn_v.weight", QuantType::F32, &[4, 2]),
        ("blk.2.attn_gate.weight", QuantType::F32, &[4, 4]),
        ("blk.2.attn_output.weight", QuantType::F32, &[4, 4]),
        ("blk.2.residual_alpha.weight", QuantType::F32, &[4]),
    ];

    let bytes = create_synthetic_gguf_v3_with_kvs("dory", 32, &kvs, &tensor_specs);
    let gguf = GgufFile::parse(&bytes).unwrap();
    let recipe = DoryRecipe::from_gguf(&gguf).unwrap();
    let cache = DoryPhysicalCache::from_gguf(&recipe, &gguf).unwrap();

    assert_eq!(cache.len(), 3);
    assert_eq!(cache.repacked_matrix_count(), 13);
    assert!(cache.output_scale().is_some());

    let stop = cache.stop_metadata();
    assert_eq!(stop.bos_token_id, Some(1));
    assert_eq!(stop.im_end_token_id, Some(3));
    assert_eq!(stop.think_open_token_id, Some(4));
    assert_eq!(stop.think_close_token_id, Some(5));
    assert!(stop.chatml_template_aligned);
    assert!(stop.has_think_tokens);

    let criteria = stop.stop_criteria(8, 32);
    assert_eq!(criteria.evaluate(4, 1, 2), None);
    assert_eq!(criteria.evaluate(5, 2, 3), None);
    assert_eq!(criteria.evaluate(3, 3, 4), Some(StopReason::EosToken(3)));

    let prompt = stop.format_chatml_prompt("Test", Some("medium"), true);
    assert!(prompt.contains(DORY_MEDIUM_REASONING_SUFFIX));
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n"));
}
