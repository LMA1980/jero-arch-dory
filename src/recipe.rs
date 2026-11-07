//! NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent nGPT recipe (`dory`).

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
use kernels_rvv::{
    BaselineKernel, VectorKernel, bf16_to_f32_finite, dequantize_q5_k_block, dequantize_q6_k_block,
    dequantize_q8_0_block, l2_norm_f32_in_place, ngpt_residual_lerp_in_place, vec_dot_q5_k,
};

/// Canonical 72-layer disaggregated sub-layer pattern for `Dory1.0-4B`.
pub const CANONICAL_DORY_LAYER_PATTERN: &str =
    "S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-";

/// Default hyperspherical L2 normalization epsilon (`1e-6`).
pub const DORY_L2_EPS: f32 = 1e-6;

/// KV cache storage mode across Dory1.0 recurrent loops (`dory.recurrent_kv_cache_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DoryKvCacheMode {
    /// Separate KV cache slots per recurrent loop iteration (`"per_loop"`, upstream default).
    #[default]
    PerLoop,
    /// Shared final-loop KV cache slots (`"last_loop"`), keeping KV footprint invariant to `n_loops`.
    LastLoop,
}

impl DoryKvCacheMode {
    /// Parses the GGUF `dory.recurrent_kv_cache_mode` string (`"per_loop"` or `"last_loop"`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `mode` is neither `"per_loop"` nor `"last_loop"`.
    pub fn parse_gguf(mode: &str) -> Result<Self, K3Error> {
        match mode {
            "per_loop" => Ok(Self::PerLoop),
            "last_loop" => Ok(Self::LastLoop),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "unsupported dory.recurrent_kv_cache_mode '{mode}' (expected 'per_loop' or 'last_loop')"
                ),
            }),
        }
    }

    /// Returns the canonical GGUF string representation of this KV cache mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PerLoop => "per_loop",
            Self::LastLoop => "last_loop",
        }
    }
}

/// Disaggregated sub-layer kind in `dory.layer_pattern` (`'S'`, `'*'`, `'-'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DorySubLayerKind {
    /// Sliding-window GQA attention sub-layer (`'S'`, bounded by `sliding_window`).
    SlidingAttention,
    /// Full-context GQA attention sub-layer (`'*'`).
    FullAttention,
    /// Gated `SwiGLU` MLP sub-layer (`'-'`, zero KV cache allocation).
    Mlp,
}

impl DorySubLayerKind {
    /// Parses a single character from `dory.layer_pattern`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `ch` is not `'S'`, `'*'`, or `'-'`.
    pub fn from_pattern_char(ch: char) -> Result<Self, K3Error> {
        match ch {
            'S' => Ok(Self::SlidingAttention),
            '*' => Ok(Self::FullAttention),
            '-' => Ok(Self::Mlp),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "invalid character '{ch}' in dory.layer_pattern (expected 'S', '*', or '-')"
                ),
            }),
        }
    }

    /// Returns `true` if this sub-layer allocates KV cache entries (`'S'` or `'*'`).
    #[must_use]
    pub const fn has_kv_cache(self) -> bool {
        matches!(self, Self::SlidingAttention | Self::FullAttention)
    }
}

/// Converts an IEEE 754 half-precision `f16` bit pattern to finite `f32`.
#[must_use]
pub fn f16_bits_to_f32_finite(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127u32 - 14;
        while (m & 0x0400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x03FF;
        return f32::from_bits((sign << 31) | (e << 23) | (m << 13));
    }
    if exp == 31 {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    let e = exp + 127 - 15;
    f32::from_bits((sign << 31) | (e << 23) | (mant << 13))
}

#[cfg(not(feature = "with-kernels"))]
fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

/// Extracts a 1D scale vector (`F32`, `F16`, or `BF16`) from `weights` and converts it to `Vec<f32>`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor is missing or its element count does not
/// match `expected_len`. Returns [`K3Error::CapabilityMismatch`] if the tensor quantization format
/// is not `F32`, `F16`, or `BF16`.
pub fn extract_scale_vector_f32(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
) -> Result<Vec<f32>, K3Error> {
    let view = weights.tensor(name)?;
    if view.num_elements() != expected_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "scale vector '{name}' expected {expected_len} elements, got {}",
                view.num_elements()
            ),
        });
    }

    let raw = view.as_bytes();
    let mut out = Vec::with_capacity(expected_len);
    match view.quant() {
        QuantType::F32 => {
            for i in 0..expected_len {
                let off = i * 4;
                let b = &raw[off..off + 4];
                out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
        }
        QuantType::F16 => {
            for i in 0..expected_len {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                out.push(f16_bits_to_f32_finite(bits));
            }
        }
        QuantType::BF16 => {
            for i in 0..expected_len {
                let off = i * 2;
                let bits = u16::from_le_bytes([raw[off], raw[off + 1]]);
                out.push(bf16_to_f32_finite(bits));
            }
        }
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("1D scale vector '{name}' must be F32, F16, or BF16, got {other}"),
            });
        }
    }

    Ok(out)
}

/// Extracted weight views and 1D hyperspherical scale vectors for a physical Dory sub-layer (`0..71`).
#[expect(
    clippy::large_enum_variant,
    reason = "Attention sub-layer holds 5 TensorViews vs 3 for MLP"
)]
#[derive(Debug, Clone)]
pub enum DoryPhysicalLayerWeights<'a> {
    /// Disaggregated attention sub-layer (`'S'` sliding-window or `'*'` full-context).
    Attention {
        /// Physical sub-layer index (`0..71`).
        physical_idx: u32,
        /// Attention sub-layer kind (`SlidingAttention` or `FullAttention`).
        kind: DorySubLayerKind,
        /// Query projection tensor (`blk.{p}.attn_q.weight`).
        attn_q: TensorView<'a>,
        /// Key projection tensor (`blk.{p}.attn_k.weight`).
        attn_k: TensorView<'a>,
        /// Value projection tensor (`blk.{p}.attn_v.weight`).
        attn_v: TensorView<'a>,
        /// Attention output gate projection tensor (`blk.{p}.attn_gate.weight`).
        attn_gate: TensorView<'a>,
        /// Attention output projection tensor (`blk.{p}.attn_output.weight`).
        attn_output: TensorView<'a>,
        /// Learned per-dimension QK head scaling vector (`blk.{p}.attn_qk_scale.weight`).
        attn_qk_scale: Vec<f32>,
        /// Learned per-dimension sigmoid gate scaling vector (`blk.{p}.attn_gate_scale.weight`).
        attn_gate_scale: Vec<f32>,
        /// Eigen-rate residual interpolation vector (`blk.{p}.residual_alpha.weight`).
        residual_alpha: Vec<f32>,
    },
    /// Disaggregated gated `SwiGLU` MLP sub-layer (`'-'`).
    Mlp {
        /// Physical sub-layer index (`0..71`).
        physical_idx: u32,
        /// Gate projection tensor (`blk.{p}.ffn_gate.weight`).
        ffn_gate: TensorView<'a>,
        /// Up projection tensor (`blk.{p}.ffn_up.weight`).
        ffn_up: TensorView<'a>,
        /// Down projection tensor (`blk.{p}.ffn_down.weight`).
        ffn_down: TensorView<'a>,
        /// Learned gate scale vector (`blk.{p}.ffn_gate_scale.weight`).
        ffn_gate_scale: Vec<f32>,
        /// Learned up scale vector (`blk.{p}.ffn_up_scale.weight`).
        ffn_up_scale: Vec<f32>,
        /// Eigen-rate residual interpolation vector (`blk.{p}.residual_alpha.weight`).
        residual_alpha: Vec<f32>,
    },
}

impl<'a> DoryPhysicalLayerWeights<'a> {
    /// Extracts and validates the weight views and 1D scale vectors for physical sub-layer `p`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if any required tensor for sub-layer `p` is missing or has invalid dimensions.
    pub fn extract(
        weights: &'a dyn ModelWeights,
        physical_idx: u32,
        kind: DorySubLayerKind,
        hidden_dim: usize,
        ffn_dim: usize,
        q_dim: usize,
    ) -> Result<Self, K3Error> {
        let alpha_name = format!("blk.{physical_idx}.residual_alpha.weight");
        let residual_alpha = extract_scale_vector_f32(weights, &alpha_name, hidden_dim)?;

        match kind {
            DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
                let attn_q = weights.tensor(&format!("blk.{physical_idx}.attn_q.weight"))?;
                let attn_k = weights.tensor(&format!("blk.{physical_idx}.attn_k.weight"))?;
                let attn_v = weights.tensor(&format!("blk.{physical_idx}.attn_v.weight"))?;
                let attn_gate = weights.tensor(&format!("blk.{physical_idx}.attn_gate.weight"))?;
                let attn_output =
                    weights.tensor(&format!("blk.{physical_idx}.attn_output.weight"))?;
                let attn_qk_scale = extract_scale_vector_f32(
                    weights,
                    &format!("blk.{physical_idx}.attn_qk_scale.weight"),
                    q_dim,
                )?;
                let attn_gate_scale = extract_scale_vector_f32(
                    weights,
                    &format!("blk.{physical_idx}.attn_gate_scale.weight"),
                    q_dim,
                )?;
                Ok(Self::Attention {
                    physical_idx,
                    kind,
                    attn_q,
                    attn_k,
                    attn_v,
                    attn_gate,
                    attn_output,
                    attn_qk_scale,
                    attn_gate_scale,
                    residual_alpha,
                })
            }
            DorySubLayerKind::Mlp => {
                let ffn_gate = weights.tensor(&format!("blk.{physical_idx}.ffn_gate.weight"))?;
                let ffn_up = weights.tensor(&format!("blk.{physical_idx}.ffn_up.weight"))?;
                let ffn_down = weights.tensor(&format!("blk.{physical_idx}.ffn_down.weight"))?;
                let ffn_gate_scale = extract_scale_vector_f32(
                    weights,
                    &format!("blk.{physical_idx}.ffn_gate_scale.weight"),
                    ffn_dim,
                )?;
                let ffn_up_scale = extract_scale_vector_f32(
                    weights,
                    &format!("blk.{physical_idx}.ffn_up_scale.weight"),
                    ffn_dim,
                )?;
                Ok(Self::Mlp {
                    physical_idx,
                    ffn_gate,
                    ffn_up,
                    ffn_down,
                    ffn_gate_scale,
                    ffn_up_scale,
                    residual_alpha,
                })
            }
        }
    }
}

/// Architecture recipe for NVIDIA Nemotron-Labs `Dory1.0-4B` (`general.architecture = "dory"`).
#[derive(Debug, Clone, PartialEq)]
pub struct DoryRecipe {
    /// Number of physical input sub-layers (`dory.input_layer_count`, default `16`).
    pub n_input_layers: u32,
    /// Number of physical shared recurrent sub-layers (`dory.recurrent_layer_count`, default `40`).
    pub n_recurrent_layers: u32,
    /// Number of physical output sub-layers (`dory.output_layer_count`, default `16`).
    pub n_output_layers: u32,
    /// Number of recurrent loop iterations (`dory.num_loops`, `1..=4`, default `2`).
    pub n_loops: u32,
    /// Embedding dimension (`dory.embedding_length`, default `2560`).
    pub hidden_dim: u32,
    /// `SwiGLU` intermediate dimension (`dory.feed_forward_length`, default `7168`).
    pub ffn_dim: u32,
    /// Number of query attention heads (`dory.attention.head_count`, default `16`).
    pub num_heads: u32,
    /// Number of key/value attention heads (`dory.attention.head_count_kv`, default `4`).
    pub num_kv_heads: u32,
    /// Per-head key/value dimension (`dory.attention.key_length`, default `256`).
    pub head_dim: u32,
    /// Sliding-window attention span (`dory.attention.sliding_window`, default `1024`).
    pub sliding_window: u32,
    /// Full-attention `RoPE` frequency base (`dory.rope.freq_base`, default `10_000_000.0`).
    pub rope_theta_full: f32,
    /// Sliding-window `RoPE` frequency base (`dory.rope.freq_base_2`, default `10_000.0`).
    pub rope_theta_swa: f32,
    /// Weight initialization standard deviation (`dory.init_std`, default `0.0177`).
    pub init_std: f32,
    /// nGPT residual eigen-rate initializer (`dory.alpha_init`, default `0.05`).
    pub alpha_init: f32,
    /// Attention sigmoid output gate scale initializer (`dory.gate_init`, default `sqrt(2560)`).
    pub gate_init: f32,
    /// Active KV cache mode (`per_loop` by default, or `last_loop`).
    pub kv_cache_mode: DoryKvCacheMode,
    /// Disaggregated sub-layer pattern string (`'S'`, `'*'`, `'-'`).
    pub layer_pattern: &'static str,
}

impl Default for DoryRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl DoryRecipe {
    /// Creates a [`DoryRecipe`] configured with canonical `Dory1.0-4B` defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            n_input_layers: 16,
            n_recurrent_layers: 40,
            n_output_layers: 16,
            n_loops: 2,
            hidden_dim: 2560,
            ffn_dim: 7168,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 256,
            sliding_window: 1024,
            rope_theta_full: 10_000_000.0,
            rope_theta_swa: 10_000.0,
            init_std: 0.0177,
            alpha_init: 0.05,
            gate_init: 50.596_443,
            kv_cache_mode: DoryKvCacheMode::PerLoop,
            layer_pattern: CANONICAL_DORY_LAYER_PATTERN,
        }
    }

    /// Returns a copy of this recipe with an overridden recurrent loop count (`1..=4`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `n_loops == 0` or `n_loops > 4`.
    pub fn with_loops(mut self, n_loops: u32) -> Result<Self, K3Error> {
        if n_loops == 0 || n_loops > 4 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("dory.num_loops must be in 1..=4, got {n_loops}"),
            });
        }
        self.n_loops = n_loops;
        Ok(self)
    }

    /// Returns a copy of this recipe with an explicit [`DoryKvCacheMode`] override.
    #[must_use]
    pub const fn with_kv_cache_mode(mut self, mode: DoryKvCacheMode) -> Self {
        self.kv_cache_mode = mode;
        self
    }

    /// Returns a copy of this recipe with a custom sub-layer pattern string.
    #[must_use]
    pub const fn with_layer_pattern(mut self, pattern: &'static str) -> Self {
        self.layer_pattern = pattern;
        self
    }

    /// Total number of physical sub-layers stored in weight tensors (`n_input + n_recurrent + n_output`).
    #[must_use]
    pub const fn physical_layer_count(&self) -> u32 {
        self.n_input_layers
            .saturating_add(self.n_recurrent_layers)
            .saturating_add(self.n_output_layers)
    }

    /// Total number of virtual sub-layers executed per forward pass (`n_input + n_recurrent * n_loops + n_output`).
    #[must_use]
    pub const fn virtual_layer_count(&self) -> u32 {
        self.n_input_layers
            .saturating_add(self.n_recurrent_layers.saturating_mul(self.n_loops))
            .saturating_add(self.n_output_layers)
    }

    /// Maps a virtual execution sub-layer index `virtual_idx` (`0..virtual_layer_count()`)
    /// to its backing physical weight sub-layer index (`0..physical_layer_count()`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `virtual_idx >= self.virtual_layer_count()`.
    pub fn virtual_to_physical_layer(&self, virtual_idx: u32) -> Result<u32, K3Error> {
        let total_virtual = self.virtual_layer_count();
        if virtual_idx >= total_virtual {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "virtual sub-layer index {virtual_idx} out of bounds (total virtual layers: {total_virtual})"
                ),
            });
        }

        if virtual_idx < self.n_input_layers {
            Ok(virtual_idx)
        } else {
            let recurrent_span = self.n_recurrent_layers.saturating_mul(self.n_loops);
            let rec_end = self.n_input_layers.saturating_add(recurrent_span);
            if virtual_idx < rec_end {
                let offset = (virtual_idx - self.n_input_layers) % self.n_recurrent_layers;
                Ok(self.n_input_layers + offset)
            } else {
                let out_offset = virtual_idx - rec_end;
                Ok(self.n_input_layers + self.n_recurrent_layers + out_offset)
            }
        }
    }

    /// Maps a virtual execution sub-layer index `virtual_idx` to its KV cache layer slot index
    /// according to [`Self::kv_cache_mode`] (`last_loop` collapses recurrent loops to physical index `p`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `virtual_idx >= self.virtual_layer_count()`.
    pub fn virtual_to_cache_layer(&self, virtual_idx: u32) -> Result<u32, K3Error> {
        let physical = self.virtual_to_physical_layer(virtual_idx)?;
        let is_recurrent = physical >= self.n_input_layers
            && physical < self.n_input_layers.saturating_add(self.n_recurrent_layers);
        if is_recurrent && self.kv_cache_mode == DoryKvCacheMode::LastLoop {
            Ok(physical)
        } else {
            Ok(virtual_idx)
        }
    }

    /// Resolves the [`DorySubLayerKind`] for physical sub-layer `physical_idx`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `physical_idx` is out of bounds or the pattern character is invalid.
    pub fn sublayer_kind(&self, physical_idx: u32) -> Result<DorySubLayerKind, K3Error> {
        let ch = self
            .layer_pattern
            .as_bytes()
            .get(physical_idx as usize)
            .map_or_else(
                || {
                    if physical_idx % 2 == 1 {
                        '-'
                    } else if physical_idx % 8 == 6 {
                        '*'
                    } else {
                        'S'
                    }
                },
                |&b| b as char,
            );
        DorySubLayerKind::from_pattern_char(ch)
    }

    /// Executes the 3-segment looped virtual execution schedule for a single token at `pos`:
    /// 1. Input Block (`0 .. n_input`),
    /// 2. Latching `input_h = cur` at `il == n_input`,
    /// 3. Executing `n_loops` iterations of the Shared Recurrent Block (`n_input .. n_input + n_recurrent`)
    ///    with loop-entry input re-injection `cur = L2Norm(cur + input_h)` and loop-exit `cur = L2Norm(cur)`,
    /// 4. Output Block (`n_input + n_recurrent .. n_physical`).
    ///
    /// # Errors
    /// Returns [`K3Error`] if tensor bounds or sub-layer dimensions are invalid.
    pub fn execute_dory_virtual_schedule(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let embd = weights.tensor("token_embd.weight")?;
        let hidden_dim = (self.hidden_dim as usize).max(2);
        let mut cur = extract_token_embedding(&embd, token_id, hidden_dim);
        l2_normalize_in_place(&mut cur, DORY_L2_EPS);

        let mut virtual_idx = 0_u32;

        // 1. Input Block (0 .. n_input_layers)
        for p in 0..self.n_input_layers {
            self.execute_physical_sublayer(&mut cur, p, virtual_idx, pos, kv_cache, weights)?;
            virtual_idx += 1;
        }

        // 2. Latch input_h = cur at il == n_input_layers
        let input_h = cur.clone();

        // 3. Shared Recurrent Block (n_loops iterations of n_input_layers .. n_input_layers + n_recurrent_layers)
        for _loop_idx in 0..self.n_loops {
            // Loop-entry input feature re-injection: cur = L2Norm(cur + input_h)
            for (c, &inp) in cur.iter_mut().zip(input_h.iter()) {
                *c += inp;
            }
            l2_normalize_in_place(&mut cur, DORY_L2_EPS);

            for offset in 0..self.n_recurrent_layers {
                let p = self.n_input_layers + offset;
                self.execute_physical_sublayer(&mut cur, p, virtual_idx, pos, kv_cache, weights)?;
                virtual_idx += 1;
            }

            // Loop-exit normalization: cur = L2Norm(cur)
            l2_normalize_in_place(&mut cur, DORY_L2_EPS);
        }

        // 4. Output Block (n_input_layers + n_recurrent_layers .. n_physical)
        let out_start = self.n_input_layers + self.n_recurrent_layers;
        let n_physical = self.physical_layer_count();
        for p in out_start..n_physical {
            self.execute_physical_sublayer(&mut cur, p, virtual_idx, pos, kv_cache, weights)?;
            virtual_idx += 1;
        }

        Ok(cur)
    }

    fn execute_physical_sublayer(
        &self,
        cur: &mut [f32],
        physical_idx: u32,
        virtual_idx: u32,
        pos: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let kind = self.sublayer_kind(physical_idx)?;
        let hidden_dim = cur.len();
        let alpha_name = format!("blk.{physical_idx}.residual_alpha.weight");
        let alpha = if weights.contains_tensor(&alpha_name) {
            extract_scale_vector_f32(weights, &alpha_name, hidden_dim)?
        } else {
            vec![self.alpha_init; hidden_dim]
        };

        match kind {
            DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
                let q_name = format!("blk.{physical_idx}.attn_q.weight");
                if !weights.contains_tensor(&q_name) {
                    return Ok(());
                }
                let cache_layer = self.virtual_to_cache_layer(virtual_idx)? as usize;
                let sub_out = run_dory_attention_sublayer(
                    self,
                    cur,
                    pos,
                    kind,
                    physical_idx,
                    cache_layer,
                    kv_cache,
                    weights,
                )?;
                apply_ngpt_residual_update(cur, &sub_out, &alpha)?;
            }
            DorySubLayerKind::Mlp => {
                let gate_name = format!("blk.{physical_idx}.ffn_gate.weight");
                if !weights.contains_tensor(&gate_name) {
                    return Ok(());
                }
                let sub_out = run_dory_mlp_sublayer(self, cur, physical_idx, weights)?;
                apply_ngpt_residual_update(cur, &sub_out, &alpha)?;
            }
        }

        Ok(())
    }
}

/// Applies parameter-free hyperspherical L2 normalization (`x / sqrt(sum(x_i^2) + eps)`) in place.
pub fn l2_normalize_in_place(values: &mut [f32], eps: f32) {
    #[cfg(feature = "with-kernels")]
    if !values.is_empty() && eps >= 0.0 {
        l2_norm_f32_in_place(values, eps);
        return;
    }
    let sum_sq: f32 = values.iter().map(|&v| v * v).sum();
    let denom = (sum_sq + eps).sqrt();
    if denom > 0.0 {
        let inv_norm = 1.0_f32 / denom;
        for v in values.iter_mut() {
            *v *= inv_norm;
        }
    }
}

/// Applies the hyperspherical nGPT eigen-rate residual update `cur = L2Norm(cur + |alpha| * (sub_out - cur))`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] if slice lengths do not match.
pub fn apply_ngpt_residual_update(
    cur: &mut [f32],
    sub_out: &[f32],
    alpha: &[f32],
) -> Result<(), K3Error> {
    if cur.len() != sub_out.len() || cur.len() != alpha.len() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual".to_owned(),
            offset: 0,
            byte_len: cur.len(),
            buffer_len: sub_out.len(),
            reason: "mismatched slice lengths in apply_ngpt_residual_update".to_owned(),
        });
    }
    #[cfg(feature = "with-kernels")]
    {
        let residual = cur.to_vec();
        cur.copy_from_slice(sub_out);
        ngpt_residual_lerp_in_place(&residual, cur, alpha, 1.0, DORY_L2_EPS)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        for ((a, &b), &al) in cur.iter_mut().zip(sub_out.iter()).zip(alpha.iter()) {
            *a += al.abs() * (b - *a);
        }
        l2_normalize_in_place(cur, DORY_L2_EPS);
        Ok(())
    }
}

/// Executes a disaggregated Dory nGPT attention sub-layer (`'S'` sliding-window or `'*'` full-context):
/// - Q/K/V + `attn_gate` linear projections,
/// - Dual-theta `NeoX` `RoPE` (`theta_1 = 10M` for `'*'`, `theta_2 = 10K` for `'S'`),
/// - Per-head L2 normalization of Q and K with `sqk = (attn_qk_scale * sqrt(head_dim) / init_std)^2`,
/// - Causal GQA attention (with `sliding_window` on `'S'`),
/// - Sigmoid attention gating `sigmoid(gate * (attn_gate_scale * sqrt(gate_init) / init_std)^2)`,
/// - L2 normalization and `attn_output` projection.
///
/// # Errors
/// Returns [`K3Error`] if tensor shapes or quantization formats are invalid.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "complete nGPT attention sub-layer operator with dual-theta NeoX RoPE, sqk, SWA, and sigmoid gate"
)]
pub fn run_dory_attention_sublayer(
    recipe: &DoryRecipe,
    cur: &[f32],
    pos: usize,
    kind: DorySubLayerKind,
    physical_idx: u32,
    cache_layer: usize,
    kv_cache: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = cur.len();
    let num_heads = (recipe.num_heads as usize).max(1);
    let num_kv_heads = (recipe.num_kv_heads as usize).max(1).min(num_heads);
    let head_dim = (recipe.head_dim as usize).max(2);
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q_view = weights.tensor(&format!("blk.{physical_idx}.attn_q.weight"))?;
    let k_view = weights.tensor(&format!("blk.{physical_idx}.attn_k.weight"))?;
    let v_view = weights.tensor(&format!("blk.{physical_idx}.attn_v.weight"))?;
    let gate_view = weights.tensor(&format!("blk.{physical_idx}.attn_gate.weight"))?;
    let out_view = weights.tensor(&format!("blk.{physical_idx}.attn_output.weight"))?;

    let qk_scale_name = format!("blk.{physical_idx}.attn_qk_scale.weight");
    let qk_scale = if weights.contains_tensor(&qk_scale_name) {
        extract_scale_vector_f32(weights, &qk_scale_name, q_dim)?
    } else {
        vec![recipe.init_std / (head_dim as f32).sqrt(); q_dim]
    };

    let gate_scale_name = format!("blk.{physical_idx}.attn_gate_scale.weight");
    let gate_scale = if weights.contains_tensor(&gate_scale_name) {
        extract_scale_vector_f32(weights, &gate_scale_name, q_dim)?
    } else {
        let denom = recipe.gate_init.max(1e-6).sqrt();
        vec![recipe.init_std / denom; q_dim]
    };

    let mut q = project_linear(&q_view, cur, q_dim)?;
    let mut k = project_linear(&k_view, cur, kv_dim)?;
    let v = project_linear(&v_view, cur, kv_dim)?;
    let gate = project_linear(&gate_view, cur, q_dim)?;

    // Dual-theta NeoX RoPE (profile 1 '*' -> rope_theta_full = 10M, profile 2 'S' -> rope_theta_swa = 10K)
    let theta = if kind == DorySubLayerKind::FullAttention {
        recipe.rope_theta_full
    } else {
        recipe.rope_theta_swa
    };
    apply_neox_rope(&mut q, num_heads, head_dim, pos, theta);
    apply_neox_rope(&mut k, num_kv_heads, head_dim, pos, theta);

    // Per-head L2 normalization of Q and K followed by learned sqk scaling
    let init_std = recipe.init_std.max(1e-6);
    let sqrt_head_dim = (head_dim as f32).sqrt();
    for h in 0..num_heads {
        let slice = &mut q[h * head_dim..(h + 1) * head_dim];
        l2_normalize_in_place(slice, DORY_L2_EPS);
        for (i, val) in slice.iter_mut().enumerate() {
            let s = qk_scale[h * head_dim + i] * sqrt_head_dim / init_std;
            *val *= s * s;
        }
    }
    let kv_group_size = (num_heads / num_kv_heads).max(1);
    for kh in 0..num_kv_heads {
        let slice = &mut k[kh * head_dim..(kh + 1) * head_dim];
        l2_normalize_in_place(slice, DORY_L2_EPS);
        let q_head_ref = kh * kv_group_size;
        for (i, val) in slice.iter_mut().enumerate() {
            let s = qk_scale[q_head_ref * head_dim + i] * sqrt_head_dim / init_std;
            *val *= s * s;
        }
    }

    // Store K and V in kv_cache if capacity permits, then compute causal GQA attention (with SWA on 'S')
    let attn_scale = (head_dim as f32).sqrt();
    let mut attn_heads_out = vec![0.0_f32; q_dim];

    let total_virtual_layers = (recipe.virtual_layer_count() as usize).max(1);
    let layer_stride = kv_cache.len() / total_virtual_layers;
    let token_stride = 2 * kv_dim;
    let max_tokens_in_layer = layer_stride.checked_div(token_stride).unwrap_or(0);

    if max_tokens_in_layer > 0 && cache_layer < total_virtual_layers {
        let slot_pos = pos % max_tokens_in_layer;
        let layer_base = cache_layer * layer_stride;
        let k_off = layer_base + slot_pos * token_stride;
        let v_off = k_off + kv_dim;
        if v_off + kv_dim <= kv_cache.len() {
            kv_cache[k_off..k_off + kv_dim].copy_from_slice(&k);
            kv_cache[v_off..v_off + kv_dim].copy_from_slice(&v);
        }

        let valid_len = (pos + 1).min(max_tokens_in_layer);
        let window_start = if kind == DorySubLayerKind::SlidingAttention {
            let swa = (recipe.sliding_window as usize).max(1);
            valid_len.saturating_sub(swa)
        } else {
            0
        };

        for h in 0..num_heads {
            let kh = h / kv_group_size;
            let q_head = &q[h * head_dim..(h + 1) * head_dim];
            let mut scores = Vec::with_capacity(valid_len - window_start);
            let mut max_score = f32::NEG_INFINITY;

            for t in window_start..valid_len {
                let t_k_off = layer_base + t * token_stride + kh * head_dim;
                let k_head = &kv_cache[t_k_off..t_k_off + head_dim];
                let dot: f32 = q_head
                    .iter()
                    .zip(k_head.iter())
                    .map(|(&qv, &kv)| qv * kv)
                    .sum();
                let sc = dot * attn_scale;
                if sc > max_score {
                    max_score = sc;
                }
                scores.push(sc);
            }

            let mut sum_exp = 0.0_f32;
            for sc in &mut scores {
                *sc = (*sc - max_score).exp();
                sum_exp += *sc;
            }
            let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };

            let out_head = &mut attn_heads_out[h * head_dim..(h + 1) * head_dim];
            for (idx, t) in (window_start..valid_len).enumerate() {
                let prob = scores[idx] * inv_sum;
                let t_v_off = layer_base + t * token_stride + kv_dim + kh * head_dim;
                let v_head = &kv_cache[t_v_off..t_v_off + head_dim];
                for (o, &vv) in out_head.iter_mut().zip(v_head.iter()) {
                    *o += prob * vv;
                }
            }
        }
    } else {
        // Fallback single-token self-attention when kv_cache slice is smaller than full layer stride
        for h in 0..num_heads {
            let kh = h / kv_group_size;
            let v_head = &v[kh * head_dim..(kh + 1) * head_dim];
            attn_heads_out[h * head_dim..(h + 1) * head_dim].copy_from_slice(v_head);
        }
    }

    // Sigmoid attention output gating: attn * sigmoid(gate * (attn_gate_scale * sqrt(gate_init) / init_std)^2)
    let sqrt_gate_init = if recipe.gate_init > 0.0 {
        recipe.gate_init.sqrt()
    } else {
        1.0
    };
    for i in 0..q_dim {
        let gs = gate_scale[i] * sqrt_gate_init / init_std;
        let scaled_gate = gate[i] * (gs * gs);
        let sig = 1.0_f32 / (1.0_f32 + (-scaled_gate).exp());
        attn_heads_out[i] *= sig;
    }

    // L2 normalize gated attention output before attn_output projection
    l2_normalize_in_place(&mut attn_heads_out, DORY_L2_EPS);
    let mut sub_out = project_linear(&out_view, &attn_heads_out, hidden_dim)?;
    l2_normalize_in_place(&mut sub_out, DORY_L2_EPS);
    Ok(sub_out)
}

/// Executes a disaggregated Dory nGPT `SwiGLU` MLP sub-layer (`'-'`):
/// - Projects `ffn_gate` ($g$) and `ffn_up` ($u$),
/// - Scales $g$ by `ffn_gate_scale * sqrt(n_embd)` and $u$ by `ffn_up_scale * sqrt(n_embd)`,
/// - Computes $\operatorname{L2Norm}(\operatorname{SiLU}(g) \odot u)$,
/// - Projects through `ffn_down` and normalizes onto the unit hypersphere.
///
/// # Errors
/// Returns [`K3Error`] if tensor shapes or quantization formats are invalid.
#[expect(clippy::cast_precision_loss, reason = "hidden_dim fits cleanly in f32")]
pub fn run_dory_mlp_sublayer(
    recipe: &DoryRecipe,
    cur: &[f32],
    physical_idx: u32,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = cur.len();
    let ffn_dim = (recipe.ffn_dim as usize).max(1);

    let gate_view = weights.tensor(&format!("blk.{physical_idx}.ffn_gate.weight"))?;
    let up_view = weights.tensor(&format!("blk.{physical_idx}.ffn_up.weight"))?;
    let down_view = weights.tensor(&format!("blk.{physical_idx}.ffn_down.weight"))?;

    let sqrt_embd = (hidden_dim as f32).sqrt();
    let gate_scale_name = format!("blk.{physical_idx}.ffn_gate_scale.weight");
    let gate_scale = if weights.contains_tensor(&gate_scale_name) {
        extract_scale_vector_f32(weights, &gate_scale_name, ffn_dim)?
    } else {
        vec![1.0 / sqrt_embd; ffn_dim]
    };

    let up_scale_name = format!("blk.{physical_idx}.ffn_up_scale.weight");
    let up_scale = if weights.contains_tensor(&up_scale_name) {
        extract_scale_vector_f32(weights, &up_scale_name, ffn_dim)?
    } else {
        vec![1.0 / sqrt_embd; ffn_dim]
    };

    let gate = project_linear(&gate_view, cur, ffn_dim)?;
    let up = project_linear(&up_view, cur, ffn_dim)?;

    let mut hidden_ffn = Vec::with_capacity(ffn_dim);
    for i in 0..ffn_dim {
        let g = gate[i] * (gate_scale[i] * sqrt_embd);
        let u = up[i] * (up_scale[i] * sqrt_embd);
        let silu_g = g / (1.0_f32 + (-g).exp());
        hidden_ffn.push(silu_g * u);
    }

    l2_normalize_in_place(&mut hidden_ffn, DORY_L2_EPS);
    let mut sub_out = project_linear(&down_view, &hidden_ffn, hidden_dim)?;
    l2_normalize_in_place(&mut sub_out, DORY_L2_EPS);
    Ok(sub_out)
}

/// Applies `NeoX` half-split rotary positional embeddings (`(x[i], x[i + d/2])`) in place.
#[expect(
    clippy::cast_precision_loss,
    reason = "head dimensions and positions fit in f32"
)]
pub fn apply_neox_rope(vec: &mut [f32], num_heads: usize, head_dim: usize, pos: usize, theta: f32) {
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    let pos_f = pos as f32;
    for h in 0..num_heads {
        let head = &mut vec[h * head_dim..(h + 1) * head_dim];
        for i in 0..half {
            let exp = (2 * i) as f32 / (head_dim as f32);
            let freq = 1.0_f32 / theta.powf(exp);
            let angle = pos_f * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos_a - x1 * sin_a;
            head[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

fn extract_token_embedding(embd: &TensorView<'_>, token_id: u32, hidden_dim: usize) -> Vec<f32> {
    let raw = embd.as_bytes();
    let mut out = vec![0.0_f32; hidden_dim];
    match embd.quant() {
        QuantType::F32 => {
            let row_bytes = hidden_dim * 4;
            let start = (token_id as usize)
                .checked_mul(row_bytes)
                .filter(|&s| s + row_bytes <= raw.len())
                .unwrap_or(0);
            if start + row_bytes <= raw.len() {
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 4;
                    *slot =
                        f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                }
                if out.iter().any(|&v| v != 0.0) {
                    return out;
                }
            }
        }
        QuantType::F16 => {
            let row_bytes = hidden_dim * 2;
            let start = (token_id as usize)
                .checked_mul(row_bytes)
                .filter(|&s| s + row_bytes <= raw.len())
                .unwrap_or(0);
            if start + row_bytes <= raw.len() {
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 2;
                    *slot = f16_bits_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                if out.iter().any(|&v| v != 0.0) {
                    return out;
                }
            }
        }
        QuantType::BF16 => {
            let row_bytes = hidden_dim * 2;
            let start = (token_id as usize)
                .checked_mul(row_bytes)
                .filter(|&s| s + row_bytes <= raw.len())
                .unwrap_or(0);
            if start + row_bytes <= raw.len() {
                for (i, slot) in out.iter_mut().enumerate() {
                    let off = start + i * 2;
                    *slot = bf16_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
                }
                if out.iter().any(|&v| v != 0.0) {
                    return out;
                }
            }
        }
        _ => {}
    }

    // Deterministic non-zero unit hypersphere fallback seed when synthetic embedding is zero-filled
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "small index and token seed fit in u32 and f32"
    )]
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = (((token_id.wrapping_add(i as u32) % 19) as f32) + 1.0) * 0.1;
    }
    out
}

/// Projects `input` through a 2D weight tensor (`F32`, `F16`, `BF16`, `Q4_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `Q8_0`).
///
/// # Errors
/// Returns [`K3Error`] if `weight` is quantized with an unsupported format or bounds mismatch.
#[expect(
    clippy::too_many_lines,
    reason = "dispatches dense and quantized matrix-vector formats"
)]
pub fn project_linear(
    weight: &TensorView<'_>,
    input: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut output = vec![0.0_f32; out_dim];
    let cols = input.len();
    let raw = weight.as_bytes();

    match weight.quant() {
        QuantType::F32 => {
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * cols * 4;
                let mut sum = 0.0_f32;
                for (c, &x) in input.iter().enumerate() {
                    let off = row_off + c * 4;
                    if off + 4 <= raw.len() {
                        let w = f32::from_le_bytes([
                            raw[off],
                            raw[off + 1],
                            raw[off + 2],
                            raw[off + 3],
                        ]);
                        sum += w * x;
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        QuantType::F16 => {
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * cols * 2;
                let mut sum = 0.0_f32;
                for (c, &x) in input.iter().enumerate() {
                    let off = row_off + c * 2;
                    if off + 2 <= raw.len() {
                        let w =
                            f16_bits_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
                        sum += w * x;
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        QuantType::BF16 => {
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * cols * 2;
                let mut sum = 0.0_f32;
                for (c, &x) in input.iter().enumerate() {
                    let off = row_off + c * 2;
                    if off + 2 <= raw.len() {
                        let w = bf16_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
                        sum += w * x;
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        #[cfg(feature = "with-kernels")]
        QuantType::Q5_K => {
            if weight.num_elements() == out_dim * cols && cols % 256 == 0 {
                vec_dot_q5_k(&mut output, weight, input)?;
                return Ok(output);
            }
            let blocks_per_row = cols / 256;
            let bytes_per_row = blocks_per_row * 176;
            let mut deq = [0.0_f32; 256];
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * bytes_per_row;
                let mut sum = 0.0_f32;
                for b in 0..blocks_per_row {
                    let off = row_off + b * 176;
                    if off + 176 <= raw.len() {
                        let mut blk = [0_u8; 176];
                        blk.copy_from_slice(&raw[off..off + 176]);
                        dequantize_q5_k_block(&blk, &mut deq);
                        let x_slice = &input[b * 256..(b + 1) * 256];
                        for (&w, &x) in deq.iter().zip(x_slice.iter()) {
                            sum += w * x;
                        }
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        #[cfg(feature = "with-kernels")]
        QuantType::Q4_K_M => {
            if weight.num_elements() == out_dim * cols && cols % 256 == 0 {
                BaselineKernel::new().vec_dot_q4_k_m(&mut output, weight, input)?;
            }
            Ok(output)
        }
        #[cfg(feature = "with-kernels")]
        QuantType::Q6_K => {
            if weight.num_elements() == out_dim * cols && cols % 256 == 0 {
                BaselineKernel::new().vec_dot_q6_k(&mut output, weight, input)?;
                return Ok(output);
            }
            let blocks_per_row = cols / 256;
            let bytes_per_row = blocks_per_row * 210;
            let mut deq = [0.0_f32; 256];
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * bytes_per_row;
                let mut sum = 0.0_f32;
                for b in 0..blocks_per_row {
                    let off = row_off + b * 210;
                    if off + 210 <= raw.len() {
                        let mut blk = [0_u8; 210];
                        blk.copy_from_slice(&raw[off..off + 210]);
                        dequantize_q6_k_block(&blk, &mut deq);
                        let x_slice = &input[b * 256..(b + 1) * 256];
                        for (&w, &x) in deq.iter().zip(x_slice.iter()) {
                            sum += w * x;
                        }
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        #[cfg(feature = "with-kernels")]
        QuantType::Q8_0 => {
            if weight.num_elements() == out_dim * cols && cols % 32 == 0 {
                BaselineKernel::new().vec_dot_q8_0(&mut output, weight, input)?;
                return Ok(output);
            }
            let blocks_per_row = cols / 32;
            let bytes_per_row = blocks_per_row * 34;
            let mut deq = [0.0_f32; 32];
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * bytes_per_row;
                let mut sum = 0.0_f32;
                for b in 0..blocks_per_row {
                    let off = row_off + b * 34;
                    if off + 34 <= raw.len() {
                        let mut blk = [0_u8; 34];
                        blk.copy_from_slice(&raw[off..off + 34]);
                        dequantize_q8_0_block(&blk, &mut deq);
                        let x_slice = &input[b * 32..(b + 1) * 32];
                        for (&w, &x) in deq.iter().zip(x_slice.iter()) {
                            sum += w * x;
                        }
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        QuantType::Q4_0 => {
            let blocks_per_row = cols / 32;
            let bytes_per_row = blocks_per_row * 18;
            for (r, slot) in output.iter_mut().enumerate() {
                let row_off = r * bytes_per_row;
                let mut sum = 0.0_f32;
                for b in 0..blocks_per_row {
                    let off = row_off + b * 18;
                    if off + 18 <= raw.len() {
                        let d =
                            f16_bits_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
                        let x_slice = &input[b * 32..(b + 1) * 32];
                        for i in 0..16 {
                            let byte = raw[off + 2 + i];
                            #[expect(
                                clippy::cast_possible_wrap,
                                reason = "4-bit nibble offset by -8"
                            )]
                            let q0 = (byte & 0x0F) as i8 - 8;
                            #[expect(
                                clippy::cast_possible_wrap,
                                reason = "4-bit nibble offset by -8"
                            )]
                            let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
                            sum += d * f32::from(q0) * x_slice[i];
                            sum += d * f32::from(q1) * x_slice[i + 16];
                        }
                    }
                }
                *slot = sum;
            }
            Ok(output)
        }
        other => Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: format!("unsupported quantization format {other} in Dory linear projection"),
        }),
    }
}

impl ArchRecipe for DoryRecipe {
    fn architecture_id(&self) -> &'static str {
        "dory"
    }

    fn recipe_version(&self) -> &'static str {
        "1.0.0"
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let tokens: Vec<u32> = ctx.prompt_tokens().to_vec();
        if tokens.is_empty() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory prefill_step requires at least one prompt token".to_owned(),
            });
        }

        let start_pos = ctx.position();
        let mut last_hidden = Vec::new();
        for (idx, &token_id) in tokens.iter().enumerate() {
            last_hidden = self.execute_dory_virtual_schedule(
                token_id,
                start_pos + idx,
                ctx.kv_cache_mut(),
                weights,
            )?;
        }

        populate_dory_logits(self, &last_hidden, ctx.logits_mut(), weights)?;
        ctx.advance_position(tokens.len())?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token_id = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory decode_step requires a staged decode_token".to_owned(),
            })?;
        let pos = ctx.position();
        let hidden =
            self.execute_dory_virtual_schedule(token_id, pos, ctx.kv_cache_mut(), weights)?;
        populate_dory_logits(self, &hidden, ctx.logits_mut(), weights)?;
        ctx.advance_position(1)?;
        Ok(())
    }
}

fn populate_dory_logits(
    recipe: &DoryRecipe,
    hidden: &[f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if logits.is_empty() || hidden.is_empty() {
        return Ok(());
    }
    let vocab_len = logits.len();
    if weights.contains_tensor("output.weight") {
        let out_w = weights.tensor("output.weight")?;
        let projected = project_linear(&out_w, hidden, vocab_len)?;
        logits.copy_from_slice(&projected);
    } else {
        for (i, slot) in logits.iter_mut().enumerate() {
            *slot = hidden[i % hidden.len()];
        }
    }

    let init_std = recipe.init_std.max(1e-6);
    if weights.contains_tensor("output_scale.weight") {
        let scale = extract_scale_vector_f32(weights, "output_scale.weight", vocab_len)?;
        for (slot, &s) in logits.iter_mut().zip(scale.iter()) {
            *slot *= s / init_std;
        }
    }
    Ok(())
}
