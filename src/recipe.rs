//! NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent nGPT recipe (`dory`).

use std::borrow::Cow;

use jero_core::{
    ArchRecipe, ExecutionContext, GgufFile, GgufValue, K3Error, KvCacheView, KvGeometry,
    ModelMetadata, ModelWeights, QuantType, TensorView,
};

#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{l2_norm_f32_in_place, ngpt_residual_lerp_in_place};

use crate::quant;

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
    /// Converts from `jero_core::DoryKvCacheMode` to `DoryKvCacheMode`.
    #[must_use]
    pub const fn from_core(mode: jero_core::DoryKvCacheMode) -> Self {
        match mode {
            jero_core::DoryKvCacheMode::PerLoop => Self::PerLoop,
            jero_core::DoryKvCacheMode::LastLoop => Self::LastLoop,
        }
    }

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

/// Disaggregated sub-layer kind in `dory.layer_pattern` (`'S'`, `'*'` / `'g'`, `'-'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DorySubLayerKind {
    /// Sliding-window GQA attention sub-layer (`'S'`, bounded by `sliding_window`).
    SlidingAttention,
    /// Full-context GQA attention sub-layer (`'*'` or `'g'`).
    FullAttention,
    /// Gated `SwiGLU` MLP sub-layer (`'-'`, zero KV cache allocation).
    Mlp,
}

impl DorySubLayerKind {
    /// Parses a single character from `dory.layer_pattern`.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `ch` is not `'S'`, `'*'`, `'g'`, or `'-'`.
    pub fn from_pattern_char(ch: char) -> Result<Self, K3Error> {
        match ch {
            'S' => Ok(Self::SlidingAttention),
            '*' | 'g' => Ok(Self::FullAttention),
            '-' => Ok(Self::Mlp),
            _ => Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "invalid character '{ch}' in dory.layer_pattern (expected 'S', '*', 'g', or '-')"
                ),
            }),
        }
    }

    /// Returns `true` if this sub-layer allocates KV cache entries (`'S'`, `'*'`, or `'g'`).
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

/// Extracts a 1D scale vector (`F32`, `F16`, or `BF16`) from `weights` and converts it to `Vec<f32>`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if the tensor is missing, truncated, or its element count
/// does not match `expected_len`. Returns [`K3Error::CapabilityMismatch`] if the tensor quantization
/// format is not `F32`, `F16`, or `BF16`.
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
    let elem_bytes = match view.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("1D scale vector '{name}' must be F32, F16, or BF16, got {other}"),
            });
        }
    };
    let required_bytes = expected_len.saturating_mul(elem_bytes);
    if raw.len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: view.offset(),
            byte_len: required_bytes,
            buffer_len: raw.len(),
            reason: format!(
                "scale vector '{name}' byte length {} smaller than required {required_bytes}",
                raw.len()
            ),
        });
    }

    let mut out = Vec::with_capacity(expected_len);
    match view.quant() {
        QuantType::F32 => {
            for b in raw[..required_bytes].chunks_exact(4) {
                let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                out.push(if val.is_finite() { val } else { 0.0 });
            }
        }
        QuantType::F16 => {
            for b in raw[..required_bytes].chunks_exact(2) {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                out.push(f16_bits_to_f32_finite(bits));
            }
        }
        QuantType::BF16 => {
            for b in raw[..required_bytes].chunks_exact(2) {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                out.push(quant::bf16_to_f32_finite(bits));
            }
        }
        _ => unreachable!(),
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

/// Canonical vocabulary size for `Dory1.0-4B` (`131,072`).
pub const DORY_DEFAULT_VOCAB_SIZE: usize = 131_072;

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
    /// Disaggregated sub-layer pattern string (`'S'`, `'*'`, `'g'`, `'-'`).
    pub layer_pattern: Cow<'static, str>,
    /// `RoPE` profile identifier parsed from `dory.rope.profile` (default `"dual_theta"`).
    pub rope_profile: Cow<'static, str>,
    /// Model vocabulary size (`dory.vocab_size` or `token_embd.weight` rows, default `131_072`).
    pub vocab_size: usize,
    /// Optional explicit final logit multiplier (`dory.logit_scale`).
    pub logit_scale: Option<f32>,
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
            layer_pattern: Cow::Borrowed(CANONICAL_DORY_LAYER_PATTERN),
            rope_profile: Cow::Borrowed("dual_theta"),
            vocab_size: DORY_DEFAULT_VOCAB_SIZE,
            logit_scale: None,
        }
    }

    /// Parses `dory.rope.profile` across string, unsigned/non-negative signed integer (`u32`/`i32`),
    /// returning a canonical profile identifier (`"dual_theta"`, `"neox"`, `"profile_{n}"`, etc.).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `val` is empty or neither a string nor a
    /// non-negative integer.
    pub fn parse_rope_profile(val: &GgufValue<'_>) -> Result<Cow<'static, str>, K3Error> {
        if let Some(s) = val.as_str() {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "dory.rope.profile string must not be empty".to_owned(),
                });
            }
            return Ok(Cow::Owned(trimmed.to_owned()));
        }
        if let Some(u) = val.as_u32() {
            return Ok(Cow::Owned(format!("profile_{u}")));
        }
        Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: "dory.rope.profile must be a string or non-negative integer".to_owned(),
        })
    }

    /// Hydrates a [`DoryRecipe`] from parsed [`ModelMetadata`].
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if the metadata architecture is not `"dory"` or if
    /// hyperparameters / `dory.layer_pattern` are malformed.
    pub fn from_metadata(metadata: &ModelMetadata) -> Result<Self, K3Error> {
        if metadata.architecture() != "dory" {
            return Err(K3Error::CapabilityMismatch {
                architecture: metadata.architecture().to_owned(),
                reason: format!(
                    "expected architecture 'dory', got '{}'",
                    metadata.architecture()
                ),
            });
        }

        let mut recipe = Self::new();
        if let Some(dory) = metadata.dory() {
            recipe.n_input_layers = dory.input_layer_count();
            recipe.n_recurrent_layers = dory.recurrent_layer_count();
            recipe.n_output_layers = dory.output_layer_count();
            recipe.n_loops = dory.num_loops();
            recipe.kv_cache_mode = DoryKvCacheMode::from_core(dory.recurrent_kv_cache_mode());
            recipe.layer_pattern = Cow::Owned(dory.layer_pattern().to_owned());
            let profile = dory.rope_profile().trim();
            if profile.is_empty() {
                return Err(K3Error::CapabilityMismatch {
                    architecture: "dory".to_owned(),
                    reason: "dory.rope.profile must not be empty".to_owned(),
                });
            }
            recipe.rope_profile = Cow::Owned(profile.to_owned());
            recipe.rope_theta_swa = dory.rope_freq_base_2();
            recipe.init_std = dory.init_std();
            recipe.alpha_init = dory.alpha_init();
            recipe.gate_init = dory.gate_init();
        }

        if let Ok(dim) = u32::try_from(metadata.embedding_dim())
            && dim > 0
        {
            recipe.hidden_dim = dim;
        }
        if let Some(ffn) = metadata
            .feed_forward_length()
            .and_then(|v| u32::try_from(v).ok())
            && ffn > 0
        {
            recipe.ffn_dim = ffn;
        }
        if let Ok(nh) = u32::try_from(metadata.head_count())
            && nh > 0
        {
            recipe.num_heads = nh;
        }
        if let Ok(nkv) = u32::try_from(metadata.head_count_kv())
            && nkv > 0
        {
            recipe.num_kv_heads = nkv;
        }
        if let Some(kl) = metadata
            .key_length()
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
        {
            recipe.head_dim = kl;
        } else if recipe.hidden_dim != 2560 && recipe.num_heads > 0 {
            recipe.head_dim = (recipe.hidden_dim / recipe.num_heads).max(2);
        }
        if let Some(sw) = metadata
            .sliding_window()
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
        {
            recipe.sliding_window = sw;
        }
        if let Some(freq) = metadata
            .rope_freq_base()
            .filter(|v| v.is_finite() && *v > 0.0)
        {
            recipe.rope_theta_full = freq;
        }
        if let Some(vs) = metadata.vocab_size().filter(|&v| v > 0) {
            recipe.vocab_size = vs;
        } else if let Some(embd) = metadata.find_tensor("token_embd.weight")
            && embd.shape().len() == 2
        {
            let hidden = recipe.hidden_dim as usize;
            let (d0, d1) = (embd.shape()[0], embd.shape()[1]);
            if d0 == hidden && d1 > 0 {
                recipe.vocab_size = d1;
            } else if d1 == hidden && d0 > 0 {
                recipe.vocab_size = d0;
            }
        }
        if let Some(ls) = metadata.logit_scale().filter(|v| v.is_finite() && *v > 0.0) {
            recipe.logit_scale = Some(ls);
        }

        recipe.validate_hyperparams()?;
        Ok(recipe)
    }

    /// Hydrates a [`DoryRecipe`] directly from a parsed [`GgufFile`], honoring string/integer/default
    /// `dory.rope.profile` encodings, dual-theta SWA/global `RoPE` frequencies, dynamic
    /// `dory.layer_pattern`, and `n_loops` / `kv_sharing` overrides.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if metadata keys or hyperparameters are invalid.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut recipe = Self::from_metadata(gguf.metadata())?;

        if let Some(val) = gguf.metadata_value("dory.rope.profile") {
            recipe.rope_profile = Self::parse_rope_profile(val)?;
        }
        if let Some(val) = gguf.metadata_value("dory.rope.freq_base_2") {
            let f = val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.rope.freq_base_2 must be a float".to_owned(),
            })?;
            recipe.rope_theta_swa = f;
        }
        if let Some(val) = gguf.metadata_value("dory.rope.freq_base") {
            let f = val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.rope.freq_base must be a float".to_owned(),
            })?;
            recipe.rope_theta_full = f;
        }
        if let Some(val) = gguf.metadata_value("dory.attention.sliding_window")
            && let Some(sw) = val.as_u32()
        {
            recipe.sliding_window = sw;
        }
        if let Some(val) = gguf.metadata_value("dory.n_loops") {
            let loops = val.as_u32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.n_loops must be an integer in 1..=4".to_owned(),
            })?;
            recipe = recipe.with_loops(loops)?;
        }
        if let Some(val) = gguf
            .metadata_value("dory.kv_sharing")
            .or_else(|| gguf.metadata_value("dory.recurrent_kv_cache_mode"))
        {
            let mode_str = val.as_str().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory KV cache sharing mode must be a string".to_owned(),
            })?;
            recipe.kv_cache_mode = DoryKvCacheMode::parse_gguf(mode_str)?;
        }
        if let Some(val) = gguf.metadata_value("dory.logit_scale") {
            let ls = val.as_f32().ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory.logit_scale must be a float".to_owned(),
            })?;
            recipe.logit_scale = Some(ls);
        }

        recipe.validate_hyperparams()?;
        Ok(recipe)
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

    /// Returns a copy of this recipe with a static sub-layer pattern string.
    #[must_use]
    pub fn with_layer_pattern(mut self, pattern: &'static str) -> Self {
        self.layer_pattern = Cow::Borrowed(pattern);
        self
    }

    /// Returns a copy of this recipe with a borrowed or heap-owned dynamic sub-layer pattern string.
    #[must_use]
    pub fn with_layer_pattern_owned(mut self, pattern: impl Into<Cow<'static, str>>) -> Self {
        self.layer_pattern = pattern.into();
        self
    }

    /// Returns a copy of this recipe with an explicit `RoPE` profile identifier.
    #[must_use]
    pub fn with_rope_profile(mut self, profile: impl Into<Cow<'static, str>>) -> Self {
        self.rope_profile = profile.into();
        self
    }

    /// Validates recipe hyperparameters and `layer_pattern` consistency.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if any dimension, loop count, `RoPE` parameter, or
    /// pattern entry is invalid.
    pub fn validate_hyperparams(&self) -> Result<(), K3Error> {
        if self.n_loops == 0 || self.n_loops > 4 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("dory.num_loops must be in 1..=4, got {}", self.n_loops),
            });
        }
        if self.hidden_dim < 2
            || self.ffn_dim == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.num_heads % self.num_kv_heads != 0
            || self.head_dim < 2
            || self.sliding_window == 0
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory architecture dimensions must be valid positive multiples".to_owned(),
            });
        }
        if !self.rope_theta_full.is_finite()
            || self.rope_theta_full <= 0.0
            || !self.rope_theta_swa.is_finite()
            || self.rope_theta_swa <= 0.0
            || !self.init_std.is_finite()
            || self.init_std <= 0.0
            || !self.alpha_init.is_finite()
            || !self.gate_init.is_finite()
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory RoPE/initializer constants must be positive finite values".to_owned(),
            });
        }
        if let Some(ls) = self.logit_scale
            && (!ls.is_finite() || ls <= 0.0)
        {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("dory.logit_scale must be positive and finite, got {ls}"),
            });
        }
        self.validate_layer_pattern()
    }

    /// Validates that `layer_pattern` covers every physical sub-layer `0..physical_layer_count()`
    /// using only valid tokens (`'S'`, `'*'`, `'g'`, `'-'`).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `physical_layer_count() == 0`, `layer_pattern`
    /// is shorter than `physical_layer_count()`, or any character is unrecognized.
    pub fn validate_layer_pattern(&self) -> Result<(), K3Error> {
        let physical = self.physical_layer_count() as usize;
        if physical == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "dory physical layer count must be non-zero".to_owned(),
            });
        }
        let pattern_chars: Vec<char> = self.layer_pattern.chars().collect();
        if pattern_chars.len() < physical {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "dory.layer_pattern length ({}) is shorter than physical layer count ({physical})",
                    pattern_chars.len()
                ),
            });
        }
        for (p, &ch) in pattern_chars.iter().take(physical).enumerate() {
            DorySubLayerKind::from_pattern_char(ch).map_err(|_| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "invalid character '{ch}' at physical sub-layer {p} in dory.layer_pattern"
                ),
            })?;
        }
        Ok(())
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

    /// Counts how many physical attention sub-layers (`'S'`, `'*'`, `'g'`) appear within `[start..end)`.
    fn count_attention_in_physical_range(&self, start: u32, end: u32) -> Result<usize, K3Error> {
        let mut count = 0_usize;
        for p in start..end {
            if self.sublayer_kind(p)?.has_kv_cache() {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Returns the compact count of KV cache slots allocated strictly for attention sub-layers
    /// (`'S'`, `'*'`, `'g'`) across the configured loops and [`DoryKvCacheMode`], excluding all
    /// `'-'` MLP sub-layers (and sharing recurrent attention slots in [`DoryKvCacheMode::LastLoop`]).
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `layer_pattern` is malformed or truncated.
    pub fn attention_kv_layer_count(&self) -> Result<usize, K3Error> {
        let rec_start = self.n_input_layers;
        let out_start = rec_start.saturating_add(self.n_recurrent_layers);
        let n_physical = self.physical_layer_count();

        let in_attn = self.count_attention_in_physical_range(0, rec_start)?;
        let rec_attn = self.count_attention_in_physical_range(rec_start, out_start)?;
        let out_attn = self.count_attention_in_physical_range(out_start, n_physical)?;

        let rec_slots = match self.kv_cache_mode {
            DoryKvCacheMode::PerLoop => rec_attn.saturating_mul(self.n_loops as usize),
            DoryKvCacheMode::LastLoop => rec_attn,
        };
        Ok(in_attn.saturating_add(rec_slots).saturating_add(out_attn))
    }

    /// Maps a virtual execution sub-layer index (`0..virtual_layer_count()`) to its compact
    /// attention-only KV cache slot index (`0..attention_kv_layer_count()`), or `None` if the
    /// underlying sub-layer is an MLP (`'-'`) with no KV state.
    ///
    /// # Errors
    /// Returns [`K3Error::CapabilityMismatch`] if `virtual_idx >= self.virtual_layer_count()` or
    /// `layer_pattern` is invalid.
    pub fn virtual_to_attention_kv_slot(&self, virtual_idx: u32) -> Result<Option<usize>, K3Error> {
        let physical = self.virtual_to_physical_layer(virtual_idx)?;
        if !self.sublayer_kind(physical)?.has_kv_cache() {
            return Ok(None);
        }

        let rec_start = self.n_input_layers;
        let out_phys_start = rec_start.saturating_add(self.n_recurrent_layers);
        let in_attn = self.count_attention_in_physical_range(0, rec_start)?;
        let rec_attn = self.count_attention_in_physical_range(rec_start, out_phys_start)?;

        if virtual_idx < self.n_input_layers {
            let offset = self.count_attention_in_physical_range(0, physical)?;
            return Ok(Some(offset));
        }

        let recurrent_span = self.n_recurrent_layers.saturating_mul(self.n_loops);
        let rec_virt_end = self.n_input_layers.saturating_add(recurrent_span);
        if virtual_idx < rec_virt_end {
            let rec_offset = self.count_attention_in_physical_range(rec_start, physical)?;
            let slot = match self.kv_cache_mode {
                DoryKvCacheMode::PerLoop => {
                    let loop_idx =
                        ((virtual_idx - self.n_input_layers) / self.n_recurrent_layers) as usize;
                    in_attn
                        .saturating_add(loop_idx.saturating_mul(rec_attn))
                        .saturating_add(rec_offset)
                }
                DoryKvCacheMode::LastLoop => in_attn.saturating_add(rec_offset),
            };
            return Ok(Some(slot));
        }

        let rec_slots = match self.kv_cache_mode {
            DoryKvCacheMode::PerLoop => rec_attn.saturating_mul(self.n_loops as usize),
            DoryKvCacheMode::LastLoop => rec_attn,
        };
        let out_offset = self.count_attention_in_physical_range(out_phys_start, physical)?;
        Ok(Some(
            in_attn.saturating_add(rec_slots).saturating_add(out_offset),
        ))
    }

    /// Builds the compact attention-only [`KvGeometry`] for a sequence capacity of `seq_capacity`.
    ///
    /// # Errors
    /// Returns [`K3Error`] if `attention_kv_layer_count() == 0`, dimensions are invalid, or buffer
    /// sizing overflows `usize`.
    pub fn kv_geometry(&self, seq_capacity: usize) -> Result<KvGeometry, K3Error> {
        let num_layers = self.attention_kv_layer_count()?;
        let sliding = (self.sliding_window > 0).then_some(self.sliding_window as usize);
        KvGeometry::new(
            num_layers,
            self.num_kv_heads as usize,
            self.head_dim as usize,
            seq_capacity,
            sliding,
        )
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
        let n_physical = self.physical_layer_count();
        if physical_idx >= n_physical {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "physical sub-layer index {physical_idx} out of bounds (physical_layer_count={n_physical})"
                ),
            });
        }
        let ch = self
            .layer_pattern
            .chars()
            .nth(physical_idx as usize)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "dory.layer_pattern has no entry at physical sub-layer {physical_idx}"
                ),
            })?;
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
    /// Returns [`K3Error`] if tensor bounds, KV cache capacity, or sub-layer dimensions are invalid.
    pub fn execute_dory_virtual_schedule(
        &self,
        token_id: u32,
        pos: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let num_kv_slots = self.attention_kv_layer_count()?;
        let mut kv_view = if num_kv_slots > 0 {
            let kv_dim = (self.num_kv_heads as usize)
                .max(1)
                .saturating_mul((self.head_dim as usize).max(2));
            let slot_elems = num_kv_slots.saturating_mul(2).saturating_mul(kv_dim);
            let seq_capacity = kv_cache.len().checked_div(slot_elems).unwrap_or(0);
            if seq_capacity <= pos {
                if weights.tensor_count() > 0 {
                    return if seq_capacity == 0 {
                        Err(K3Error::TensorBoundsError {
                            tensor_name: "kv_cache".to_owned(),
                            offset: 0,
                            byte_len: slot_elems
                                .saturating_mul(pos + 1)
                                .saturating_mul(size_of::<f32>()),
                            buffer_len: size_of_val(kv_cache),
                            reason: format!(
                                "kv_cache buffer length {} cannot hold compact Dory attention slot count {num_kv_slots} at pos {pos}",
                                kv_cache.len()
                            ),
                        })
                    } else {
                        Err(K3Error::ContextLengthExceeded {
                            requested_tokens: pos + 1,
                            max_context_tokens: seq_capacity,
                        })
                    };
                }
                None
            } else {
                let geom = self.kv_geometry(seq_capacity)?;
                Some(KvCacheView::new(kv_cache, geom)?)
            }
        } else {
            None
        };

        self.execute_dory_schedule_with_kv(token_id, pos, kv_view.as_mut(), weights)
    }

    fn execute_dory_schedule_with_kv(
        &self,
        token_id: u32,
        pos: usize,
        mut kv_view: Option<&mut KvCacheView<'_>>,
        weights: &dyn ModelWeights,
    ) -> Result<Vec<f32>, K3Error> {
        let hidden_dim = (self.hidden_dim as usize).max(2);
        let mut cur = match weights.tensor("token_embd.weight") {
            Ok(embd) => extract_token_embedding(&embd, token_id, hidden_dim)?,
            Err(err) => {
                if weights.tensor_count() == 0 {
                    synthetic_unit_embedding(token_id, hidden_dim)
                } else {
                    return Err(err);
                }
            }
        };
        l2_normalize_in_place(&mut cur, DORY_L2_EPS);

        let mut virtual_idx = 0_u32;

        // 1. Input Block (0 .. n_input_layers)
        for p in 0..self.n_input_layers {
            self.execute_physical_sublayer_kv(
                &mut cur,
                p,
                virtual_idx,
                pos,
                kv_view.as_deref_mut(),
                weights,
            )?;
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
                self.execute_physical_sublayer_kv(
                    &mut cur,
                    p,
                    virtual_idx,
                    pos,
                    kv_view.as_deref_mut(),
                    weights,
                )?;
                virtual_idx += 1;
            }

            // Loop-exit normalization: cur = L2Norm(cur)
            l2_normalize_in_place(&mut cur, DORY_L2_EPS);
        }

        // 4. Output Block (n_input_layers + n_recurrent_layers .. n_physical)
        let out_start = self.n_input_layers + self.n_recurrent_layers;
        let n_physical = self.physical_layer_count();
        for p in out_start..n_physical {
            self.execute_physical_sublayer_kv(
                &mut cur,
                p,
                virtual_idx,
                pos,
                kv_view.as_deref_mut(),
                weights,
            )?;
            virtual_idx += 1;
        }

        Ok(cur)
    }

    fn execute_physical_sublayer_kv(
        &self,
        cur: &mut [f32],
        physical_idx: u32,
        virtual_idx: u32,
        pos: usize,
        kv_view: Option<&mut KvCacheView<'_>>,
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
                if weights.tensor_count() == 0 {
                    let q_name = format!("blk.{physical_idx}.attn_q.weight");
                    if !weights.contains_tensor(&q_name) {
                        return Ok(());
                    }
                }
                let kv_slot = self.virtual_to_attention_kv_slot(virtual_idx)?.ok_or_else(|| {
                    K3Error::CapabilityMismatch {
                        architecture: "dory".to_owned(),
                        reason: format!(
                            "attention sub-layer virtual={virtual_idx} physical={physical_idx} has no KV slot"
                        ),
                    }
                })?;
                let kv = kv_view.ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: "kv_cache".to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: 0,
                    reason: format!(
                        "missing KvCacheView for attention sub-layer virtual={virtual_idx} slot={kv_slot}"
                    ),
                })?;
                let sub_out = run_dory_attention_with_kv_view(
                    self,
                    cur,
                    pos,
                    kind,
                    physical_idx,
                    kv_slot,
                    kv,
                    weights,
                )?;
                apply_ngpt_residual_update(cur, &sub_out, &alpha)?;
            }
            DorySubLayerKind::Mlp => {
                if weights.tensor_count() == 0 {
                    let gate_name = format!("blk.{physical_idx}.ffn_gate.weight");
                    if !weights.contains_tensor(&gate_name) {
                        return Ok(());
                    }
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

/// Executes a disaggregated Dory nGPT attention sub-layer (`'S'` sliding-window or `'*'`/`'g'` full-context):
/// - Q/K/V + `attn_gate` linear projections,
/// - Dual-theta `NeoX` `RoPE` (`theta_1 = 10M` for `'*'`/`'g'`, `theta_2 = 10K` for `'S'`),
/// - Per-head L2 normalization of Q and K with `sqk = (attn_qk_scale * sqrt(head_dim) / init_std)^2`,
/// - Compact `KvCacheView` causal GQA attention (with `sliding_window` on `'S'`),
/// - Sigmoid attention gating `sigmoid(gate * (attn_gate_scale * sqrt(gate_init) / init_std)^2)`,
/// - L2 normalization and `attn_output` projection.
///
/// # Errors
/// Returns [`K3Error`] if tensor shapes, KV cache bounds, or quantization formats are invalid.
#[expect(
    clippy::too_many_arguments,
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
    let num_kv_slots = recipe.attention_kv_layer_count()?;
    if num_kv_slots == 0 || cache_layer >= num_kv_slots {
        return Err(K3Error::TensorBoundsError {
            tensor_name: format!("kv_cache.layer.{cache_layer}"),
            offset: 0,
            byte_len: 0,
            buffer_len: size_of_val(kv_cache),
            reason: format!(
                "attention KV slot {cache_layer} out of bounds for compact attention slot count {num_kv_slots}"
            ),
        });
    }
    let num_kv_heads = (recipe.num_kv_heads as usize).max(1);
    let head_dim = (recipe.head_dim as usize).max(2);
    let kv_dim = num_kv_heads.saturating_mul(head_dim);
    let slot_elems = num_kv_slots.saturating_mul(2).saturating_mul(kv_dim);
    let seq_capacity = kv_cache.len().checked_div(slot_elems).unwrap_or(0);
    if seq_capacity == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_cache".to_owned(),
            offset: 0,
            byte_len: slot_elems
                .saturating_mul(pos + 1)
                .saturating_mul(size_of::<f32>()),
            buffer_len: size_of_val(kv_cache),
            reason: format!(
                "kv_cache length {} smaller than single-token compact attention requirement {slot_elems}",
                kv_cache.len()
            ),
        });
    }
    let geom = recipe.kv_geometry(seq_capacity)?;
    let mut kv_view = KvCacheView::new(kv_cache, geom)?;
    run_dory_attention_with_kv_view(
        recipe,
        cur,
        pos,
        kind,
        physical_idx,
        cache_layer,
        &mut kv_view,
        weights,
    )
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "complete nGPT attention sub-layer operator backed by bounds-checked KvCacheView"
)]
fn run_dory_attention_with_kv_view(
    recipe: &DoryRecipe,
    cur: &[f32],
    pos: usize,
    kind: DorySubLayerKind,
    physical_idx: u32,
    kv_slot: usize,
    kv_view: &mut KvCacheView<'_>,
    weights: &dyn ModelWeights,
) -> Result<Vec<f32>, K3Error> {
    let hidden_dim = cur.len();
    let num_heads = (recipe.num_heads as usize).max(1);
    let num_kv_heads = (recipe.num_kv_heads as usize).max(1).min(num_heads);
    let head_dim = (recipe.head_dim as usize).max(2);
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q_view = weights.tensor(&format!("blk.{physical_idx}.attn_q.weight"))?;
    let key_weight = weights.tensor(&format!("blk.{physical_idx}.attn_k.weight"))?;
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
    let mut k = project_linear(&key_weight, cur, kv_dim)?;
    let v = project_linear(&v_view, cur, kv_dim)?;
    let gate = project_linear(&gate_view, cur, q_dim)?;

    // Dual-theta NeoX RoPE (profile 1 '*'/g -> rope_theta_full = 10M, profile 2 'S' -> rope_theta_swa = 10K)
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

    // Write K and V into compact attention-only KvCacheView (fails closed on overflow/OOB)
    {
        let (key_dst, val_dst) = kv_view.write_pos_range(kv_slot, pos)?;
        key_dst.copy_from_slice(&k);
        val_dst.copy_from_slice(&v);
    }

    let (k_plane, v_plane) = kv_view.layer_kv(kv_slot)?;
    let valid_len = pos + 1;
    let window_start = if kind == DorySubLayerKind::SlidingAttention {
        let swa = (recipe.sliding_window as usize).max(1);
        valid_len.saturating_sub(swa)
    } else {
        0
    };

    let attn_scale = (head_dim as f32).sqrt();
    let mut attn_heads_out = vec![0.0_f32; q_dim];

    for h in 0..num_heads {
        let kh = h / kv_group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = Vec::with_capacity(valid_len - window_start);
        let mut max_score = f32::NEG_INFINITY;

        for t in window_start..valid_len {
            let t_k_off = t * kv_dim + kh * head_dim;
            let k_head = &k_plane[t_k_off..t_k_off + head_dim];
            let dot = quant::dot_product_f32(q_head, k_head);
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
        let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
            1.0 / sum_exp
        } else {
            0.0
        };

        let out_head = &mut attn_heads_out[h * head_dim..(h + 1) * head_dim];
        for (idx, t) in (window_start..valid_len).enumerate() {
            let prob = scores[idx] * inv_sum;
            let t_v_off = t * kv_dim + kh * head_dim;
            let v_head = &v_plane[t_v_off..t_v_off + head_dim];
            for (o, &vv) in out_head.iter_mut().zip(v_head.iter()) {
                *o += prob * vv;
            }
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

fn synthetic_unit_embedding(token_id: u32, hidden_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; hidden_dim];
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

fn extract_token_embedding(
    embd: &TensorView<'_>,
    token_id: u32,
    hidden_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut out = vec![0.0_f32; hidden_dim];
    quant::dequantize_row_to_f32(embd, token_id as usize, hidden_dim, &mut out)?;
    Ok(out)
}

/// Projects `input` through a 2D weight tensor (`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`,
/// `Q4_K_M`, `Q5_K`, `Q6_K`) without silent synthetic fallbacks.
///
/// # Errors
/// Returns [`K3Error`] if `weight` is quantized with an unsupported format or bounds mismatch.
pub fn project_linear(
    weight: &TensorView<'_>,
    input: &[f32],
    out_dim: usize,
) -> Result<Vec<f32>, K3Error> {
    let mut output = vec![0.0_f32; out_dim];
    quant::project_matvec(&mut output, weight, input)?;
    Ok(output)
}

fn validate_1d_scale_if_present(
    weights: &dyn ModelWeights,
    name: &str,
    expected_len: usize,
) -> Result<(), K3Error> {
    if !weights.contains_tensor(name) {
        return Ok(());
    }
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
    let elem_bytes = match view.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!("1D scale vector '{name}' must be F32, F16, or BF16, got {other}"),
            });
        }
    };
    let required = expected_len.saturating_mul(elem_bytes);
    if view.as_bytes().len() < required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: view.offset(),
            byte_len: required,
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "scale vector '{name}' byte length {} smaller than required {required}",
                view.as_bytes().len()
            ),
        });
    }
    Ok(())
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

    fn requires_kv_cache(&self) -> bool {
        self.attention_kv_layer_count().unwrap_or(1) > 0
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q5_0
                | QuantType::Q8_0
                | QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q6_K
        )
    }

    fn validate_model(
        &self,
        metadata: &ModelMetadata,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        for desc in metadata.tensors() {
            if !self.supports_quant(desc.quant()) {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    desc.name(),
                    desc.quant(),
                ));
            }
            if weights.tensor_count() > 0
                && let Ok(view) = weights.tensor(desc.name())
                && !self.supports_quant(view.quant())
            {
                return Err(K3Error::unsupported_tensor_quant(
                    metadata.architecture(),
                    view.name(),
                    view.quant(),
                ));
            }
        }

        let effective = if metadata.dory().is_some()
            || (metadata.embedding_dim() > 0 && metadata.layer_count() > 0)
        {
            Cow::Owned(Self::from_metadata(metadata)?)
        } else {
            Cow::Borrowed(self)
        };
        effective.validate_hyperparams()?;

        if weights.tensor_count() == 0 {
            return Ok(());
        }

        let hidden_dim = (effective.hidden_dim as usize).max(2);
        let ffn_dim = (effective.ffn_dim as usize).max(1);
        let num_heads = (effective.num_heads as usize).max(1);
        let num_kv_heads = (effective.num_kv_heads as usize).max(1).min(num_heads);
        let head_dim = (effective.head_dim as usize).max(2);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        validate_global_weights(&effective, weights, hidden_dim)?;

        let n_physical = effective.physical_layer_count();
        for p in 0..n_physical {
            let kind = effective.sublayer_kind(p)?;
            validate_physical_layer_weights(
                &effective, p, kind, weights, hidden_dim, q_dim, kv_dim, ffn_dim,
            )?;
        }

        Ok(())
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

        let n_ctx = ctx.n_ctx();
        let start_pos = ctx.position();
        let end_pos =
            start_pos
                .checked_add(tokens.len())
                .ok_or(K3Error::ContextLengthExceeded {
                    requested_tokens: usize::MAX,
                    max_context_tokens: n_ctx,
                })?;
        if end_pos > n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: end_pos,
                max_context_tokens: n_ctx,
            });
        }

        let num_kv_slots = self.attention_kv_layer_count()?;
        let (logits_buf, kv_buf) = ctx.buffers_mut();
        let mut kv_view = if num_kv_slots > 0 {
            let geom = self.kv_geometry(n_ctx)?;
            if weights.tensor_count() > 0 {
                Some(KvCacheView::new(kv_buf, geom)?)
            } else {
                KvCacheView::new(kv_buf, geom).ok()
            }
        } else {
            None
        };

        let mut last_hidden = Vec::new();
        for (idx, &token_id) in tokens.iter().enumerate() {
            last_hidden = self.execute_dory_schedule_with_kv(
                token_id,
                start_pos + idx,
                kv_view.as_mut(),
                weights,
            )?;
        }

        populate_dory_logits(self, &last_hidden, logits_buf, weights)?;
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
        let n_ctx = ctx.n_ctx();
        let pos = ctx.position();
        if pos >= n_ctx {
            return Err(K3Error::ContextLengthExceeded {
                requested_tokens: pos + 1,
                max_context_tokens: n_ctx,
            });
        }

        let num_kv_slots = self.attention_kv_layer_count()?;
        let (logits_buf, kv_buf) = ctx.buffers_mut();
        let mut kv_view = if num_kv_slots > 0 {
            let geom = self.kv_geometry(n_ctx)?;
            if weights.tensor_count() > 0 {
                Some(KvCacheView::new(kv_buf, geom)?)
            } else {
                KvCacheView::new(kv_buf, geom).ok()
            }
        } else {
            None
        };

        let hidden =
            self.execute_dory_schedule_with_kv(token_id, pos, kv_view.as_mut(), weights)?;
        populate_dory_logits(self, &hidden, logits_buf, weights)?;
        ctx.advance_position(1)?;
        Ok(())
    }
}

fn validate_vocab_matrix_rows(
    recipe: &DoryRecipe,
    weights: &dyn ModelWeights,
    name: &str,
    hidden_dim: usize,
) -> Result<usize, K3Error> {
    let view = weights.tensor(name)?;
    if !recipe.supports_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "dory",
            view.name(),
            view.quant(),
        ));
    }
    let (rows, _) = quant::resolve_2d_dims(&view, hidden_dim)?;
    Ok(rows)
}

fn validate_global_weights(
    recipe: &DoryRecipe,
    weights: &dyn ModelWeights,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    let embd_vocab_rows =
        validate_vocab_matrix_rows(recipe, weights, "token_embd.weight", hidden_dim)?;
    let lm_vocab_rows = if weights.contains_tensor("output.weight") {
        validate_vocab_matrix_rows(recipe, weights, "output.weight", hidden_dim)?
    } else {
        embd_vocab_rows
    };
    validate_1d_scale_if_present(weights, "output_scale.weight", lm_vocab_rows)
}

fn validate_layer_matrix(
    recipe: &DoryRecipe,
    weights: &dyn ModelWeights,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let view = weights.tensor(name)?;
    if !recipe.supports_quant(view.quant()) {
        return Err(K3Error::unsupported_tensor_quant(
            "dory",
            name,
            view.quant(),
        ));
    }
    quant::validate_2d_dims(&view, rows, cols)
}

#[expect(
    clippy::too_many_arguments,
    reason = "per-layer dimension bounds check across attention and MLP projections"
)]
fn validate_physical_layer_weights(
    recipe: &DoryRecipe,
    phys_idx: u32,
    kind: DorySubLayerKind,
    weights: &dyn ModelWeights,
    hidden_dim: usize,
    q_dim: usize,
    kv_dim: usize,
    ffn_dim: usize,
) -> Result<(), K3Error> {
    validate_1d_scale_if_present(
        weights,
        &format!("blk.{phys_idx}.residual_alpha.weight"),
        hidden_dim,
    )?;

    match kind {
        DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
            for (suffix, rows, cols) in [
                ("attn_q.weight", q_dim, hidden_dim),
                ("attn_k.weight", kv_dim, hidden_dim),
                ("attn_v.weight", kv_dim, hidden_dim),
                ("attn_gate.weight", q_dim, hidden_dim),
                ("attn_output.weight", hidden_dim, q_dim),
            ] {
                validate_layer_matrix(
                    recipe,
                    weights,
                    &format!("blk.{phys_idx}.{suffix}"),
                    rows,
                    cols,
                )?;
            }
            validate_1d_scale_if_present(
                weights,
                &format!("blk.{phys_idx}.attn_qk_scale.weight"),
                q_dim,
            )?;
            validate_1d_scale_if_present(
                weights,
                &format!("blk.{phys_idx}.attn_gate_scale.weight"),
                q_dim,
            )?;
        }
        DorySubLayerKind::Mlp => {
            for (suffix, rows, cols) in [
                ("ffn_gate.weight", ffn_dim, hidden_dim),
                ("ffn_up.weight", ffn_dim, hidden_dim),
                ("ffn_down.weight", hidden_dim, ffn_dim),
            ] {
                validate_layer_matrix(
                    recipe,
                    weights,
                    &format!("blk.{phys_idx}.{suffix}"),
                    rows,
                    cols,
                )?;
            }
            validate_1d_scale_if_present(
                weights,
                &format!("blk.{phys_idx}.ffn_gate_scale.weight"),
                ffn_dim,
            )?;
            validate_1d_scale_if_present(
                weights,
                &format!("blk.{phys_idx}.ffn_up_scale.weight"),
                ffn_dim,
            )?;
        }
    }
    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "hidden dimension fits cleanly in f32 scalar"
)]
fn populate_dory_logits(
    recipe: &DoryRecipe,
    hidden: &[f32],
    logits: &mut [f32],
    weights: &dyn ModelWeights,
) -> Result<(), K3Error> {
    if hidden.is_empty() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "dory".to_owned(),
            reason: "Dory final hidden state must not be empty".to_owned(),
        });
    }
    if logits.is_empty() {
        if weights.tensor_count() == 0 {
            return Ok(());
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: "logits".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "logits output buffer must not be empty".to_owned(),
        });
    }

    let lm_head = if weights.contains_tensor("output.weight") {
        Some(weights.tensor("output.weight")?)
    } else if weights.contains_tensor("token_embd.weight") {
        // Tied LM head fallback when output.weight is omitted in GGUF.
        Some(weights.tensor("token_embd.weight")?)
    } else if weights.tensor_count() == 0 {
        None
    } else {
        return Err(weights.tensor("output.weight").unwrap_err());
    };

    let Some(lm_w) = lm_head else {
        for (i, slot) in logits.iter_mut().enumerate() {
            *slot = hidden[i % hidden.len()];
        }
        return Ok(());
    };

    let (vocab_rows, _) = quant::resolve_2d_dims(&lm_w, hidden.len())?;
    let proj_len = logits.len().min(vocab_rows);
    quant::project_matvec_prefix(&mut logits[..proj_len], &lm_w, hidden)?;
    if proj_len < logits.len() {
        logits[proj_len..].fill(0.0);
    }

    let init_std = recipe.init_std.max(1e-6);
    if weights.contains_tensor("output_scale.weight") {
        let scale = extract_scale_vector_f32(weights, "output_scale.weight", vocab_rows)?;
        let extra = recipe.logit_scale.unwrap_or(1.0);
        for (slot, &s) in logits[..proj_len].iter_mut().zip(scale[..proj_len].iter()) {
            *slot *= (s / init_std) * extra;
        }
    } else {
        let scale = recipe
            .logit_scale
            .unwrap_or_else(|| 1.0_f32 / (hidden.len() as f32).sqrt());
        for slot in &mut logits[..proj_len] {
            *slot *= scale;
        }
    }
    Ok(())
}
