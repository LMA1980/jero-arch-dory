//! Native integration tests verifying 256-dim forward/decode parity across all 9 GGUF
//! quantization formats, explicit naive full-sequence causal attention (SWA vs Full), and
//! tied vs untied LM head + `output_scale.weight` scaling.

use jero_arch_dory::{
    DORY_L2_EPS, DoryRecipe, DorySubLayerKind, apply_neox_rope, f16_bits_to_f32_finite,
    l2_normalize_in_place, project_linear, run_dory_attention_sublayer,
};
use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, ModelWeights, QuantType,
    TensorView, test_helpers::create_synthetic_gguf_v3_with_kvs,
};

struct OwnedEntry {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

struct OwnedTestWeights {
    entries: Vec<OwnedEntry>,
}

impl OwnedTestWeights {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn upsert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.name == name) {
            existing.quant = quant;
            existing.shape = shape;
            existing.bytes = bytes;
        } else {
            self.entries.push(OwnedEntry {
                name: name.to_owned(),
                quant,
                shape,
                bytes,
            });
        }
    }

    fn to_tensor_specs(&self) -> Vec<(&str, QuantType, &[usize])> {
        self.entries
            .iter()
            .map(|e| (e.name.as_str(), e.quant, e.shape.as_slice()))
            .collect()
    }
}

impl ModelWeights for OwnedTestWeights {
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
                reason: format!("tensor '{name}' missing"),
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
    reason = "deterministic finite normal f32 to IEEE 754 binary16 test helper"
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

fn encode_dense_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows * cols * if quant == QuantType::F32 { 4 } else { 2 });
    for r in 0..rows {
        for c in 0..cols {
            #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
            let val = seed * (((r * 13 + c * 5) % 19) as f32 - 9.0) * 0.1;
            match quant {
                QuantType::F32 => out.extend_from_slice(&val.to_le_bytes()),
                QuantType::F16 => out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes()),
                QuantType::BF16 => out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes()),
                _ => unreachable!("non-dense quant dispatched to encode_dense_matrix"),
            }
        }
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "small modular nibble/mask test matrix generators"
)]
fn encode_legacy_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let blocks = cols / 32;
    let block_bytes = match quant {
        QuantType::Q4_0 => 18,
        QuantType::Q5_0 => 22,
        QuantType::Q8_0 => 34,
        _ => unreachable!(),
    };
    let mut out = Vec::with_capacity(rows * blocks * block_bytes);
    for r in 0..rows {
        for b in 0..blocks {
            #[expect(clippy::cast_precision_loss, reason = "small matrix dims")]
            let d = seed * (1.0 + ((r + b) % 4) as f32 * 0.25);
            out.extend_from_slice(&f32_to_f16_bits(d).to_le_bytes());
            match quant {
                QuantType::Q4_0 => {
                    for i in 0..16 {
                        let q0 = ((r + i + b) % 15) as u8;
                        let q1 = ((r * 3 + i + b) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
                QuantType::Q5_0 => {
                    let qh = ((r as u32).wrapping_mul(0x9E37)) ^ (b as u32);
                    out.extend_from_slice(&qh.to_le_bytes());
                    for i in 0..16 {
                        let q0 = ((r + i + 2) % 15) as u8;
                        let q1 = ((r * 2 + i + 5) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
                QuantType::Q8_0 => {
                    for i in 0..32 {
                        let q = (((r * 7 + i * 3 + b) % 21) as i8) - 10;
                        out.push(q.to_ne_bytes()[0]);
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    out
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "small modular K-quant test matrix generators"
)]
fn encode_k_quant_matrix(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    let blocks = cols / 256;
    let block_bytes = match quant {
        QuantType::Q4_K_M => 144,
        QuantType::Q5_K => 176,
        QuantType::Q6_K => 210,
        _ => unreachable!(),
    };
    let mut out = Vec::with_capacity(rows * blocks * block_bytes);
    for r in 0..rows {
        for b in 0..blocks {
            match quant {
                QuantType::Q4_K_M => {
                    out.extend_from_slice(&f32_to_f16_bits(seed).to_le_bytes());
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.25).to_le_bytes());
                    for j in 0..12 {
                        out.push(((r + b + j) % 7 + 1) as u8);
                    }
                    for j in 0..128 {
                        let q0 = ((r + b + j) % 13 + 1) as u8;
                        let q1 = ((r + b + j * 3) % 13 + 1) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
                QuantType::Q5_K => {
                    out.extend_from_slice(&f32_to_f16_bits(seed).to_le_bytes());
                    out.extend_from_slice(&f32_to_f16_bits(seed * 0.2).to_le_bytes());
                    for j in 0..12 {
                        out.push(((r + b + j) % 7 + 1) as u8);
                    }
                    for j in 0..32 {
                        out.push(((r * 5 + j + b) & 0x55) as u8);
                    }
                    for j in 0..128 {
                        let q0 = ((r + j + b) % 15) as u8;
                        let q1 = ((r * 2 + j + b) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                }
                QuantType::Q6_K => {
                    for j in 0..128 {
                        let q0 = ((r + j + b) % 15) as u8;
                        let q1 = ((r * 3 + j + b) % 15) as u8;
                        out.push((q0 & 0x0F) | ((q1 & 0x0F) << 4));
                    }
                    for j in 0..64 {
                        out.push(((r + j + b) & 0x55) as u8);
                    }
                    for j in 0..16 {
                        let sc = (((r + j + b) % 7) as i8) - 3;
                        out.push(sc.to_ne_bytes()[0]);
                    }
                    out.extend_from_slice(&f32_to_f16_bits(seed).to_le_bytes());
                }
                _ => unreachable!(),
            }
        }
    }
    out
}

fn encode_matrix_for_quant(quant: QuantType, rows: usize, cols: usize, seed: f32) -> Vec<u8> {
    match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            encode_dense_matrix(quant, rows, cols, seed)
        }
        QuantType::Q4_0 | QuantType::Q5_0 | QuantType::Q8_0 => {
            encode_legacy_quant_matrix(quant, rows, cols, seed)
        }
        QuantType::Q4_K_M | QuantType::Q5_K | QuantType::Q6_K => {
            encode_k_quant_matrix(quant, rows, cols, seed)
        }
        _ => panic!("unsupported test quant format: {quant}"),
    }
}

fn build_recipe_and_weights(
    pattern: &'static str,
    hidden: u32,
    n_heads: u32,
    head_dim: u32,
    embd_quant: QuantType,
    body_quant: QuantType,
    untied_head: bool,
) -> (DoryRecipe, OwnedTestWeights) {
    let mut recipe = DoryRecipe::new()
        .with_layer_pattern(pattern)
        .with_loops(1)
        .unwrap();
    recipe.n_input_layers = 1;
    recipe.n_recurrent_layers = 1;
    recipe.n_output_layers = 1;
    recipe.hidden_dim = hidden;
    recipe.ffn_dim = hidden;
    recipe.num_heads = n_heads;
    recipe.num_kv_heads = n_heads;
    recipe.head_dim = head_dim;
    recipe.sliding_window = 2;
    recipe.vocab_size = 4;

    let mut weights = OwnedTestWeights::new();
    let h = hidden as usize;
    let vocab = 4_usize;

    weights.upsert(
        "token_embd.weight",
        embd_quant,
        vec![h, vocab],
        encode_matrix_for_quant(embd_quant, vocab, h, 0.25),
    );
    if untied_head {
        weights.upsert(
            "output.weight",
            body_quant,
            vec![h, vocab],
            encode_matrix_for_quant(body_quant, vocab, h, 0.3),
        );
    }

    for phys_idx in 0..recipe.physical_layer_count() {
        #[expect(clippy::cast_precision_loss, reason = "layer index <= 72")]
        let seed = (phys_idx as f32).mul_add(0.03, 0.15);
        match recipe.sublayer_kind(phys_idx).unwrap() {
            DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
                for (suffix, delta) in [
                    ("attn_q.weight", 0.0),
                    ("attn_k.weight", 0.02),
                    ("attn_v.weight", 0.04),
                    ("attn_gate.weight", 0.01),
                    ("attn_output.weight", 0.05),
                ] {
                    weights.upsert(
                        &format!("blk.{phys_idx}.{suffix}"),
                        body_quant,
                        vec![h, h],
                        encode_matrix_for_quant(body_quant, h, h, seed + delta),
                    );
                }
            }
            DorySubLayerKind::Mlp => {
                for (suffix, delta) in [
                    ("ffn_gate.weight", 0.01),
                    ("ffn_up.weight", 0.03),
                    ("ffn_down.weight", 0.06),
                ] {
                    weights.upsert(
                        &format!("blk.{phys_idx}.{suffix}"),
                        body_quant,
                        vec![h, h],
                        encode_matrix_for_quant(body_quant, h, h, seed + delta),
                    );
                }
            }
        }
    }

    (recipe, weights)
}

#[test]
fn test_dory_256d_all_nine_quants_forward_prefill_decode() {
    let all_quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q5_0,
        QuantType::Q8_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
    ];

    for quant in all_quants {
        let (recipe, weights) = build_recipe_and_weights("S-g", 256, 4, 64, quant, quant, true);
        let tensor_specs = weights.to_tensor_specs();
        let meta_bytes = create_synthetic_gguf_v3_with_kvs(
            "dory",
            32,
            &[
                ("dory.block_count", GgufValue::Uint32(0)),
                ("dory.embedding_length", GgufValue::Uint32(256)),
                ("dory.feed_forward_length", GgufValue::Uint32(256)),
                ("dory.attention.head_count", GgufValue::Uint32(4)),
                ("dory.attention.head_count_kv", GgufValue::Uint32(4)),
                ("dory.attention.key_length", GgufValue::Uint32(64)),
                ("dory.attention.sliding_window", GgufValue::Uint32(2)),
                ("dory.vocab_size", GgufValue::Uint32(4)),
                ("dory.input_layer_count", GgufValue::Uint32(1)),
                ("dory.recurrent_layer_count", GgufValue::Uint32(1)),
                ("dory.output_layer_count", GgufValue::Uint32(1)),
                ("dory.num_loops", GgufValue::Uint32(1)),
                ("dory.layer_pattern", GgufValue::String("S-g")),
                (
                    "dory.recurrent_kv_cache_mode",
                    GgufValue::String("per_loop"),
                ),
            ],
            &tensor_specs,
        );
        let gguf = GgufFile::parse(&meta_bytes).unwrap();
        recipe.validate_model(gguf.metadata(), &weights).unwrap();

        let n_kv_slots = recipe.attention_kv_layer_count().unwrap();
        assert_eq!(n_kv_slots, 2);
        let kv_len = n_kv_slots * 2 * 6 * 256;
        let mut kv = vec![0.0_f32; kv_len];
        let mut logits = [0.0_f32; 4];

        let mut ctx = ExecutionContext::new(&[0, 1], 6, &mut logits, &mut kv).unwrap();
        recipe.prefill_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 2);
        let prefill_logits = ctx.logits().to_vec();
        assert!(
            prefill_logits.iter().all(|v| v.is_finite()),
            "{quant} prefill logits contain non-finite value"
        );
        assert!(
            prefill_logits.iter().any(|v| v.abs() > 1e-6),
            "{quant} prefill logits collapsed to zero"
        );

        ctx.prepare_decode(2).unwrap();
        recipe.decode_step(&mut ctx, &weights).unwrap();
        assert_eq!(ctx.position(), 3);
        let decode_logits = ctx.logits();
        assert!(decode_logits.iter().all(|v| v.is_finite()));
        assert_ne!(
            prefill_logits, decode_logits,
            "{quant} decode step must advance state"
        );

        // Verify GGML col-major [cols, rows] and row-major [rows, cols] projections produce identical output.
        let bytes = encode_matrix_for_quant(quant, 8, 256, 0.25);
        let col_major = TensorView::from_bytes("w_col", quant, &[256, 8], 0, &bytes).unwrap();
        let row_major = TensorView::from_bytes("w_row", quant, &[8, 256], 0, &bytes).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "small loop index <= 255")]
        let input: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let y_col = project_linear(&col_major, &input, 8).unwrap();
        let y_row = project_linear(&row_major, &input, 8).unwrap();
        for (a, b) in y_col.iter().zip(y_row.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "{quant} layout parity mismatch: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_dory_256d_gguf_hydration_roundtrip() {
    // Verify full GGUF v3 hydration roundtrip at 256-dim parameters
    let gguf_bytes = create_synthetic_gguf_v3_with_kvs(
        "dory",
        32,
        &[
            ("dory.block_count", GgufValue::Uint32(0)),
            ("dory.embedding_length", GgufValue::Uint32(256)),
            ("dory.feed_forward_length", GgufValue::Uint32(256)),
            ("dory.attention.head_count", GgufValue::Uint32(4)),
            ("dory.attention.head_count_kv", GgufValue::Uint32(4)),
            ("dory.attention.key_length", GgufValue::Uint32(64)),
            ("dory.attention.sliding_window", GgufValue::Uint32(2)),
            ("dory.vocab_size", GgufValue::Uint32(4)),
            ("dory.input_layer_count", GgufValue::Uint32(1)),
            ("dory.recurrent_layer_count", GgufValue::Uint32(1)),
            ("dory.output_layer_count", GgufValue::Uint32(1)),
            ("dory.num_loops", GgufValue::Uint32(2)),
            ("dory.layer_pattern", GgufValue::String("S-g")),
            (
                "dory.recurrent_kv_cache_mode",
                GgufValue::String("last_loop"),
            ),
            (
                "dory.rope.profile",
                GgufValue::String("dual_swa_10k_full_10m"),
            ),
        ],
        &[],
    );
    let gguf = GgufFile::parse(&gguf_bytes).unwrap();
    let hydrated = DoryRecipe::from_gguf(&gguf).unwrap();
    assert_eq!(hydrated.attention_kv_layer_count().unwrap(), 2);
}

#[expect(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "self-contained naive full-sequence causal reference verification"
)]
fn naive_causal_attention_reference(
    recipe: &DoryRecipe,
    inputs: &[Vec<f32>],
    kind: DorySubLayerKind,
    physical_idx: u32,
    weights: &dyn ModelWeights,
) -> Vec<Vec<f32>> {
    let hidden_dim = inputs[0].len();
    let num_heads = recipe.num_heads as usize;
    let num_kv_heads = recipe.num_kv_heads as usize;
    let head_dim = recipe.head_dim as usize;
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let kv_group_size = num_heads / num_kv_heads;
    let init_std = recipe.init_std.max(1e-6);
    let theta = if kind == DorySubLayerKind::FullAttention {
        recipe.rope_theta_full
    } else {
        recipe.rope_theta_swa
    };

    let q_w = weights
        .tensor(&format!("blk.{physical_idx}.attn_q.weight"))
        .unwrap();
    let k_w = weights
        .tensor(&format!("blk.{physical_idx}.attn_k.weight"))
        .unwrap();
    let v_w = weights
        .tensor(&format!("blk.{physical_idx}.attn_v.weight"))
        .unwrap();
    let g_w = weights
        .tensor(&format!("blk.{physical_idx}.attn_gate.weight"))
        .unwrap();
    let o_w = weights
        .tensor(&format!("blk.{physical_idx}.attn_output.weight"))
        .unwrap();

    let default_qk = recipe.init_std / (head_dim as f32).sqrt();
    let default_gs = recipe.init_std / recipe.gate_init.max(1e-6).sqrt();
    let sqrt_head = (head_dim as f32).sqrt();
    let sqrt_gate = recipe.gate_init.sqrt();

    let mut ks = Vec::with_capacity(inputs.len());
    let mut vs = Vec::with_capacity(inputs.len());
    let mut outputs = Vec::with_capacity(inputs.len());

    for (pos, x) in inputs.iter().enumerate() {
        let mut q = project_linear(&q_w, x, q_dim).unwrap();
        let mut k = project_linear(&k_w, x, kv_dim).unwrap();
        let v = project_linear(&v_w, x, kv_dim).unwrap();
        let g = project_linear(&g_w, x, q_dim).unwrap();

        apply_neox_rope(&mut q, num_heads, head_dim, pos, theta);
        apply_neox_rope(&mut k, num_kv_heads, head_dim, pos, theta);

        for h in 0..num_heads {
            let sl = &mut q[h * head_dim..(h + 1) * head_dim];
            l2_normalize_in_place(sl, DORY_L2_EPS);
            let sq = default_qk * sqrt_head / init_std;
            for val in sl.iter_mut() {
                *val *= sq * sq;
            }
        }
        for kh in 0..num_kv_heads {
            let sl = &mut k[kh * head_dim..(kh + 1) * head_dim];
            l2_normalize_in_place(sl, DORY_L2_EPS);
            let sq = default_qk * sqrt_head / init_std;
            for val in sl.iter_mut() {
                *val *= sq * sq;
            }
        }

        ks.push(k);
        vs.push(v);

        let valid_len = pos + 1;
        let window_start = if kind == DorySubLayerKind::SlidingAttention {
            valid_len.saturating_sub(recipe.sliding_window as usize)
        } else {
            0
        };

        let mut concat_heads = vec![0.0_f32; q_dim];
        for h in 0..num_heads {
            let kh = h / kv_group_size;
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::new();
            let mut max_s = f32::NEG_INFINITY;
            for k_hist in &ks[window_start..valid_len] {
                let kh_slice = &k_hist[kh * head_dim..(kh + 1) * head_dim];
                let dot: f32 = qh.iter().zip(kh_slice.iter()).map(|(a, b)| a * b).sum();
                let sc = dot * sqrt_head;
                if sc > max_s {
                    max_s = sc;
                }
                scores.push(sc);
            }
            let exps: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_e: f32 = exps.iter().sum();
            let out_h = &mut concat_heads[h * head_dim..(h + 1) * head_dim];
            for (prob, v_hist) in exps
                .iter()
                .map(|e| e / sum_e)
                .zip(&vs[window_start..valid_len])
            {
                let vh = &v_hist[kh * head_dim..(kh + 1) * head_dim];
                for (o, &vv) in out_h.iter_mut().zip(vh.iter()) {
                    *o += prob * vv;
                }
            }
        }

        let gs_sq = (default_gs * sqrt_gate / init_std).powi(2);
        for i in 0..q_dim {
            concat_heads[i] *= 1.0_f32 / (1.0_f32 + (-(g[i] * gs_sq)).exp());
        }
        l2_normalize_in_place(&mut concat_heads, DORY_L2_EPS);
        let mut projected = project_linear(&o_w, &concat_heads, hidden_dim).unwrap();
        l2_normalize_in_place(&mut projected, DORY_L2_EPS);
        outputs.push(projected);
    }

    outputs
}

#[test]
fn test_dory_naive_full_sequence_causal_attention_parity() {
    let mut recipe = DoryRecipe::new()
        .with_layer_pattern("S-g")
        .with_loops(1)
        .unwrap();
    recipe.n_input_layers = 1;
    recipe.n_recurrent_layers = 1;
    recipe.n_output_layers = 1;
    recipe.hidden_dim = 8;
    recipe.ffn_dim = 8;
    recipe.num_heads = 2;
    recipe.num_kv_heads = 1;
    recipe.head_dim = 4;
    recipe.sliding_window = 2;
    recipe.vocab_size = 4;
    recipe.rope_theta_swa = 10_000.0;
    recipe.rope_theta_full = 500_000.0;

    let mut weights = OwnedTestWeights::new();
    for (phys_idx, seed) in [(0_u32, 0.2_f32), (2_u32, 0.31_f32)] {
        for (suffix, dim_out, delta) in [
            ("attn_q.weight", 8, 0.01),
            ("attn_k.weight", 4, 0.03),
            ("attn_v.weight", 4, 0.05),
            ("attn_gate.weight", 8, 0.02),
            ("attn_output.weight", 8, 0.04),
        ] {
            let dim_in = 8_usize;
            weights.upsert(
                &format!("blk.{phys_idx}.{suffix}"),
                QuantType::F32,
                vec![dim_in, dim_out],
                encode_matrix_for_quant(QuantType::F32, dim_out, dim_in, seed + delta),
            );
        }
    }

    #[expect(clippy::cast_precision_loss, reason = "small test positions")]
    let seq_inputs: Vec<Vec<f32>> = (0..5)
        .map(|t| {
            let mut v = (0..8)
                .map(|d| ((t * 5 + d) as f32).sin() * 0.5 + 0.1)
                .collect::<Vec<_>>();
            l2_normalize_in_place(&mut v, DORY_L2_EPS);
            v
        })
        .collect();

    // Check SWA ('S', physical 0 -> compact slot 0) and Full ('g', physical 2 -> compact slot 1)
    for (kind, phys_idx, kv_slot) in [
        (DorySubLayerKind::SlidingAttention, 0_u32, 0_usize),
        (DorySubLayerKind::FullAttention, 2_u32, 1_usize),
    ] {
        let expected =
            naive_causal_attention_reference(&recipe, &seq_inputs, kind, phys_idx, &weights);
        let mut kv_buf = vec![0.0_f32; 2 * 2 * 5 * 4];
        for (pos, x) in seq_inputs.iter().enumerate() {
            let actual = run_dory_attention_sublayer(
                &recipe,
                x,
                pos,
                kind,
                phys_idx,
                kv_slot,
                &mut kv_buf,
                &weights,
            )
            .unwrap();
            for (a, e) in actual.iter().zip(expected[pos].iter()) {
                assert!(
                    (a - e).abs() < 1e-5,
                    "{kind:?} pos={pos} attention parity diff: actual={a}, expected={e}"
                );
            }
        }
    }
}

#[test]
fn test_dory_tied_vs_untied_lm_head_and_output_scale_parity() {
    let hidden = 32_usize;
    let vocab = 4_usize;
    let (recipe, tied_weights) =
        build_recipe_and_weights("S-*", 32, 2, 16, QuantType::Q8_0, QuantType::F32, false);

    // 1. Tied LM head default 1/sqrt(hidden_dim) scaling matches direct final hidden * embd * (1/sqrt(32))
    let mut kv_tied = vec![0.0_f32; 2 * 2 * 4 * 32];
    let final_hidden = recipe
        .execute_dory_virtual_schedule(1, 0, &mut kv_tied, &tied_weights)
        .unwrap();
    let embd_view = tied_weights.tensor("token_embd.weight").unwrap();
    let raw_dots = project_linear(&embd_view, &final_hidden, vocab).unwrap();

    let mut logits_tied = [0.0_f32; 4];
    let mut kv_prefill = vec![0.0_f32; 2 * 2 * 4 * 32];
    let mut ctx = ExecutionContext::new(&[1], 4, &mut logits_tied, &mut kv_prefill).unwrap();
    recipe.prefill_step(&mut ctx, &tied_weights).unwrap();
    let default_scale = 1.0_f32 / 32.0_f32.sqrt();
    for (v, (&got, &dot)) in logits_tied.iter().zip(raw_dots.iter()).enumerate() {
        let want = dot * default_scale;
        assert!(
            (got - want).abs() < 1e-4,
            "tied default vocab[{v}] mismatch: got={got}, want={want}"
        );
    }

    // 2. Untied output.weight overrides tied token_embd.weight AND output_scale.weight applies (s[v] / init_std)
    let mut untied_weights = tied_weights;
    untied_weights.upsert(
        "output.weight",
        QuantType::F16,
        vec![hidden, vocab],
        encode_matrix_for_quant(QuantType::F16, vocab, hidden, 0.35),
    );
    let out_scale_vals = [0.25_f32, 0.5, 0.75, 1.0];
    let mut out_scale_bytes = Vec::with_capacity(vocab * 2);
    for s in out_scale_vals {
        out_scale_bytes.extend_from_slice(&f32_to_f16_bits(s).to_le_bytes());
    }
    untied_weights.upsert(
        "output_scale.weight",
        QuantType::F16,
        vec![vocab],
        out_scale_bytes,
    );

    let untied_view = untied_weights.tensor("output.weight").unwrap();
    let raw_untied_dots = project_linear(&untied_view, &final_hidden, vocab).unwrap();
    let mut logits_untied = [0.0_f32; 4];
    let mut kv_untied = vec![0.0_f32; 2 * 2 * 4 * 32];
    let mut ctx_u = ExecutionContext::new(&[1], 4, &mut logits_untied, &mut kv_untied).unwrap();
    recipe.prefill_step(&mut ctx_u, &untied_weights).unwrap();

    let init_std = recipe.init_std.max(1e-6);
    for v in 0..vocab {
        let expected_s = f16_bits_to_f32_finite(f32_to_f16_bits(out_scale_vals[v]));
        let want = raw_untied_dots[v] * (expected_s / init_std);
        assert!(
            (logits_untied[v] - want).abs() < 1e-4,
            "untied + output_scale vocab[{v}] mismatch: got={}, want={want}",
            logits_untied[v]
        );
    }
}
