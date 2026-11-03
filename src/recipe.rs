//! NVIDIA Nemotron-Labs Dory1.0 looped latent-recurrent nGPT recipe (`dory`).

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

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
}

/// Applies parameter-free hyperspherical L2 normalization (`x / sqrt(sum(x_i^2) + eps)`) in place.
pub fn l2_normalize_in_place(values: &mut [f32], eps: f32) {
    let sum_sq: f32 = values.iter().map(|&v| v * v).sum();
    let inv_norm = 1.0_f32 / (sum_sq + eps).sqrt();
    for v in values.iter_mut() {
        *v *= inv_norm;
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
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_len = ctx.prompt_tokens().len();
        if prompt_len == 0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory prefill_step requires at least one prompt token".to_owned(),
            });
        }
        let _ = weights.tensor("token_embd.weight")?;
        ctx.advance_position(prompt_len)?;
        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let _token_id = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: "Dory decode_step requires a staged decode_token".to_owned(),
            })?;
        let _ = weights.tensor("token_embd.weight")?;
        ctx.advance_position(1)?;
        Ok(())
    }
}
