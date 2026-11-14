//! Load-time `128-byte`-aligned physical sub-layer weight cache, pre-decoded `nGPT` 1D scale
//! vectors, and `ChatML` / `<think>` stop-token metadata alignment for `Dory1.0`.
//!
//! `Dory1.0-4B` reuses its physical recurrent sub-layers across `n_loops` iterations (`1..=4`,
//! `112` virtual sub-layers over `72` or `40` physical sub-layers). [`DoryPhysicalCache`] repacks
//! each physical 2D projection matrix once at model load time into [`DoryCachedWeight`]
//! (wrapping `jero_kernels_rvv::RepackedWeight` when `with-kernels` is enabled) and decodes all
//! 1D `nGPT` vectors (`sqk`, `suv`, `attn_alpha`, `mlp_alpha`, `attn_gate`, `mlp_gate`, and
//! optional `output_scale`) once into `128-byte`-aligned `f32` storage ([`AlignedF32Vec`]).

use std::borrow::Cow;
use std::ops::Deref;

use jero_core::{GgufArray, GgufFile, K3Error, ModelWeights, QuantType, StopCriteria, TensorView};
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    QuantizedActQ8, RepackedWeight, gemm_repacked_f32, gemm_repacked_rows, gemv_repacked_f32,
    gemv_repacked_f32_rows,
};

use crate::quant;
use crate::recipe::{DoryRecipe, DorySubLayerKind, f16_bits_to_f32_finite};

/// Canonical Dory1.0 beginning-of-sequence token string (`"<s>"`).
pub const DORY_BOS_TOKEN: &str = "<s>";

/// Canonical Dory1.0 end-of-turn / end-of-sequence `ChatML` stop token string (`"<|im_end|>"`).
pub const DORY_EOS_TOKEN: &str = "<|im_end|>";

/// Canonical `ChatML` turn-start token string (`"<|im_start|>"`).
pub const DORY_IM_START_TOKEN: &str = "<|im_start|>";

/// Opening delimiter for Dory1.0 reasoning spans (`"<think>"`).
pub const DORY_THINK_OPEN_TOKEN: &str = "<think>";

/// Closing delimiter for Dory1.0 reasoning spans (`"</think>"`).
pub const DORY_THINK_CLOSE_TOKEN: &str = "</think>";

/// Canonical suffix appended under `"medium"` reasoning effort in Dory1.0 `ChatML` formatting.
pub const DORY_MEDIUM_REASONING_SUFFIX: &str = " {reasoning effort: efficient}";

/// Number of `f32` scalars contained in a single `128-byte`-aligned block (`128 / 4 = 32`).
const F32_PER_ALIGNED128: usize = 32;

/// A `128-byte`-aligned block of `32` `f32` scalars (`128` bytes total, zero padding).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlignedF32Block(pub [f32; F32_PER_ALIGNED128]);

impl Default for AlignedF32Block {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl AlignedF32Block {
    /// Returns a zero-initialized `128-byte`-aligned `f32` block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self([0.0_f32; F32_PER_ALIGNED128])
    }
}

const _: () = {
    assert!(size_of::<AlignedF32Block>() == 128);
    assert!(align_of::<AlignedF32Block>() == 128);
};

#[cfg(not(feature = "with-kernels"))]
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
struct RawAligned128([u8; 128]);

#[cfg(not(feature = "with-kernels"))]
impl Default for RawAligned128 {
    fn default() -> Self {
        Self([0_u8; 128])
    }
}

/// Pre-decoded 1D `nGPT` scale vector backed by `128-byte`-aligned [`AlignedF32Block`] storage.
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedF32Vec {
    len: usize,
    blocks: Vec<AlignedF32Block>,
}

impl Default for AlignedF32Vec {
    fn default() -> Self {
        Self::empty()
    }
}

impl AlignedF32Vec {
    /// Returns an empty `128-byte`-aligned `f32` vector.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            len: 0,
            blocks: Vec::new(),
        }
    }

    /// Allocates and copies `values` into `128-byte`-aligned storage.
    #[must_use]
    pub fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::empty();
        }
        let mut out = Self::with_zeroed_len(values.len());
        out.as_mut_f32_capacity()[..values.len()].copy_from_slice(values);
        out
    }

    /// Allocates a `128-byte`-aligned vector of length `len` filled with finite `value`
    /// (`0.0` if non-finite).
    #[must_use]
    pub fn splat(value: f32, len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let finite = if value.is_finite() { value } else { 0.0_f32 };
        let mut out = Self::with_zeroed_len(len);
        out.as_mut_f32_capacity()[..len].fill(finite);
        out
    }

    /// Validates and decodes a 1D `F32`/`F16`/`BF16` scale tensor from `weights` once into
    /// `128-byte`-aligned `f32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on missing tensors, element count mismatch, or
    /// truncated backing bytes, or [`K3Error::CapabilityMismatch`] if `quant` is not
    /// `F32`, `F16`, or `BF16`.
    pub fn decode_tensor(
        weights: &dyn ModelWeights,
        name: &str,
        expected_len: usize,
    ) -> Result<Self, K3Error> {
        let view = weights.tensor(name)?;
        let required_bytes = validate_1d_scale_view(&view, expected_len)?;
        if expected_len == 0 {
            return Ok(Self::empty());
        }

        let mut out = Self::with_zeroed_len(expected_len);
        let dst = &mut out.as_mut_f32_capacity()[..expected_len];
        let raw = &view.as_bytes()[..required_bytes];

        match view.quant() {
            QuantType::F32 => {
                for (slot, chunk) in dst.iter_mut().zip(raw.chunks_exact(4)) {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    *slot = if val.is_finite() { val } else { 0.0_f32 };
                }
            }
            QuantType::F16 => {
                for (slot, chunk) in dst.iter_mut().zip(raw.chunks_exact(2)) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    *slot = f16_bits_to_f32_finite(bits);
                }
            }
            QuantType::BF16 => {
                for (slot, chunk) in dst.iter_mut().zip(raw.chunks_exact(2)) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    *slot = quant::bf16_to_f32_finite(bits);
                }
            }
            _ => unreachable!("validate_1d_scale_view permits only F32, F16, and BF16"),
        }

        Ok(out)
    }

    /// Decodes `name` from `weights` if present, otherwise returns `Self::splat(default_val, expected_len)`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `name` exists in `weights` but has invalid dimensions, format, or bytes.
    pub fn decode_tensor_or_default(
        weights: &dyn ModelWeights,
        name: &str,
        expected_len: usize,
        default_val: f32,
    ) -> Result<Self, K3Error> {
        if weights.contains_tensor(name) {
            Self::decode_tensor(weights, name, expected_len)
        } else {
            Ok(Self::splat(default_val, expected_len))
        }
    }

    /// Returns the number of `f32` elements stored in this vector.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains zero elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the required byte alignment (`128`).
    #[must_use]
    pub const fn alignment_bytes(&self) -> usize {
        align_of::<AlignedF32Block>()
    }

    /// Returns the total byte capacity occupied by the backing `128-byte`-aligned blocks.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        self.blocks
            .len()
            .saturating_mul(size_of::<AlignedF32Block>())
    }

    /// Returns `true` when the base pointer satisfies `ptr % 128 == 0`.
    #[must_use]
    pub fn is_aligned_128(&self) -> bool {
        (self.as_slice().as_ptr() as usize) % 128 == 0
    }

    /// Borrows the decoded `f32` elements as a `128-byte`-aligned slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        if self.len == 0 {
            // SAFETY: `NonNull::<AlignedF32Block>::dangling()` is `128-byte`-aligned (>= `4`)
            // and reading zero `f32` elements creates a valid empty aligned slice.
            return unsafe {
                core::slice::from_raw_parts(
                    core::ptr::NonNull::<AlignedF32Block>::dangling()
                        .as_ptr()
                        .cast::<f32>(),
                    0,
                )
            };
        }
        // SAFETY: `AlignedF32Block` is `#[repr(C, align(128))]` wrapping `[f32; 32]` with no
        // padding; `self.len <= self.blocks.len() * 32` is invariant-enforced in constructors.
        unsafe { core::slice::from_raw_parts(self.blocks.as_ptr().cast::<f32>(), self.len) }
    }

    fn with_zeroed_len(len: usize) -> Self {
        let block_count = len.div_ceil(F32_PER_ALIGNED128);
        Self {
            len,
            blocks: vec![AlignedF32Block::zeroed(); block_count],
        }
    }

    fn as_mut_f32_capacity(&mut self) -> &mut [f32] {
        let cap = self.blocks.len() * F32_PER_ALIGNED128;
        // SAFETY: `AlignedF32Block` is `#[repr(C, align(128))]` wrapping `[f32; 32]` with zero
        // padding, and `cap == self.blocks.len() * 32`.
        unsafe { core::slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<f32>(), cap) }
    }
}

impl Deref for AlignedF32Vec {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[f32]> for AlignedF32Vec {
    fn as_ref(&self) -> &[f32] {
        self.as_slice()
    }
}

/// Validates that a 1D scale tensor `view` has `expected_len` elements and valid `F32`/`F16`/`BF16`
/// byte storage, returning the required byte count.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on shape/byte mismatches or [`K3Error::CapabilityMismatch`]
/// on unsupported quantization types.
pub(crate) fn validate_1d_scale_view(
    view: &TensorView<'_>,
    expected_len: usize,
) -> Result<usize, K3Error> {
    if view.num_elements() != expected_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "scale vector '{}' expected {expected_len} elements, got {}",
                view.name(),
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
                reason: format!(
                    "1D scale vector '{}' must be F32, F16, or BF16, got {other}",
                    view.name()
                ),
            });
        }
    };
    let required = expected_len.saturating_mul(elem_bytes);
    if view.as_bytes().len() < required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: required,
            buffer_len: view.as_bytes().len(),
            reason: format!(
                "scale vector '{}' byte length {} smaller than required {required}",
                view.name(),
                view.as_bytes().len()
            ),
        });
    }
    Ok(required)
}

/// Load-time `128-byte`-aligned physical weight matrix stored inside [`DoryPhysicalSubLayer`].
///
/// When the `with-kernels` feature is active (default), this wraps `jero_kernels_rvv::RepackedWeight`
/// directly so quantized projections (`Q5_K`, `Q6_K`, `Q8_0`, `Q4_K_M`, `Q4_0`, `Q5_0`, `F32`,
/// `F16`, `BF16`) execute without repeated per-token block decoding across recurrent loops. When built
/// with `--no-default-features`, it copies the tensor bytes into `128-byte`-aligned storage once at
/// load time and evaluates through the scalar fallback kernels.
#[derive(Debug, Clone)]
pub struct DoryCachedWeight {
    #[cfg(feature = "with-kernels")]
    inner: RepackedWeight,
    #[cfg(not(feature = "with-kernels"))]
    tensor_name: String,
    #[cfg(not(feature = "with-kernels"))]
    quant: QuantType,
    #[cfg(not(feature = "with-kernels"))]
    rows: usize,
    #[cfg(not(feature = "with-kernels"))]
    cols: usize,
    #[cfg(not(feature = "with-kernels"))]
    byte_len: usize,
    #[cfg(not(feature = "with-kernels"))]
    blocks: Vec<RawAligned128>,
}

impl DoryCachedWeight {
    /// Repacks a 2D weight tensor view with expected `(expected_rows, expected_cols)` into a
    /// `128-byte`-aligned matrix cache entry.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `view.quant()` is unsupported by `DoryRecipe`,
    /// or [`K3Error::TensorBoundsError`] on dimension or backing byte mismatches.
    pub fn from_tensor_view(
        view: &TensorView<'_>,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        if !DoryRecipe::is_supported_quant(view.quant()) {
            return Err(K3Error::unsupported_tensor_quant(
                "dory",
                view.name(),
                view.quant(),
            ));
        }
        quant::validate_2d_dims(view, expected_rows, expected_cols)?;

        #[cfg(feature = "with-kernels")]
        {
            let inner = RepackedWeight::from_tensor(view, expected_rows, expected_cols)?;
            Ok(Self { inner })
        }

        #[cfg(not(feature = "with-kernels"))]
        {
            let raw = view.as_bytes();
            let byte_len = raw.len();
            let mut blocks = vec![RawAligned128::default(); byte_len.div_ceil(128).max(1)];
            // SAFETY: `RawAligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with zero
            // padding; `byte_len <= blocks.len() * 128`.
            let dst = unsafe {
                core::slice::from_raw_parts_mut(blocks.as_mut_ptr().cast::<u8>(), byte_len)
            };
            dst.copy_from_slice(raw);
            Ok(Self {
                tensor_name: view.name().to_owned(),
                quant: view.quant(),
                rows: expected_rows,
                cols: expected_cols,
                byte_len,
                blocks,
            })
        }
    }

    /// Loads `tensor_name` from `weights` and repacks it for `(expected_rows, expected_cols)`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `tensor_name` is missing or invalid.
    pub fn load_from_weights(
        weights: &dyn ModelWeights,
        tensor_name: &str,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        let view = weights.tensor(tensor_name)?;
        Self::from_tensor_view(&view, expected_rows, expected_cols)
    }

    /// Borrows the underlying [`jero_kernels_rvv::RepackedWeight`] (`with-kernels` feature).
    #[cfg(feature = "with-kernels")]
    #[must_use]
    pub const fn repacked(&self) -> &RepackedWeight {
        &self.inner
    }

    /// Number of output rows `N`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.rows()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.rows
        }
    }

    /// Number of logical input columns `K`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.cols()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.cols
        }
    }

    /// Source GGUF [`QuantType`] prior to load-time repacking.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.quant()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.quant
        }
    }

    /// Total bytes occupied by this cached matrix's `128-byte`-aligned buffer.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.aligned_bytes()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.blocks.len().saturating_mul(size_of::<RawAligned128>())
        }
    }

    /// Base pointer to the `128-byte`-aligned payload (`ptr as usize % 128 == 0`).
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        #[cfg(feature = "with-kernels")]
        {
            self.inner.payload_ptr()
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.blocks.as_ptr().cast::<u8>()
        }
    }

    /// Evaluates single-token matrix-vector projection `output = self * input` (`output.len() == self.rows()`,
    /// `input.len() == self.cols()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if vector lengths mismatch.
    pub fn project_f32(&self, output: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            gemv_repacked_f32(output, &self.inner, input)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            // SAFETY: `RawAligned128` has no padding and `self.byte_len <= self.blocks.len() * 128`.
            let payload = unsafe {
                core::slice::from_raw_parts(self.blocks.as_ptr().cast::<u8>(), self.byte_len)
            };
            let dims = [self.cols, self.rows];
            let view = TensorView::from_bytes(&self.tensor_name, self.quant, &dims, 0, payload)?;
            quant::project_matvec(output, &view, input)
        }
    }

    /// Evaluates a leading-row prefix GEMV (`0 < output.len() <= self.rows()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on empty/oversized `output` or mismatched `input`.
    pub fn project_prefix_f32(&self, output: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            gemv_repacked_f32_rows(output, &self.inner, input, 0..output.len())
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            // SAFETY: `RawAligned128` has no padding and `self.byte_len <= self.blocks.len() * 128`.
            let payload = unsafe {
                core::slice::from_raw_parts(self.blocks.as_ptr().cast::<u8>(), self.byte_len)
            };
            let dims = [self.cols, self.rows];
            let view = TensorView::from_bytes(&self.tensor_name, self.quant, &dims, 0, payload)?;
            quant::project_matvec_prefix(output, &view, input)
        }
    }

    /// Evaluates batched multi-token GEMM (`[m, cols] x [rows, cols]^T -> [m, rows]`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    pub fn project_batch_f32(
        &self,
        output: &mut [f32],
        input: &[f32],
        m: usize,
    ) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            gemm_repacked_f32(output, &self.inner, input, m)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            if m == 0 || input.len() != m.saturating_mul(self.cols) {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: self.tensor_name.clone(),
                    offset: 0,
                    byte_len: size_of_val(input),
                    buffer_len: m.saturating_mul(self.cols).saturating_mul(size_of::<f32>()),
                    reason: format!(
                        "batched input len {} != m ({m}) * cols ({})",
                        input.len(),
                        self.cols
                    ),
                });
            }
            if output.len() != m.saturating_mul(self.rows) {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: self.tensor_name.clone(),
                    offset: 0,
                    byte_len: size_of_val(output),
                    buffer_len: m.saturating_mul(self.rows).saturating_mul(size_of::<f32>()),
                    reason: format!(
                        "batched output len {} != m ({m}) * rows ({})",
                        output.len(),
                        self.rows
                    ),
                });
            }
            for r in 0..m {
                let x_row = &input[r * self.cols..(r + 1) * self.cols];
                let y_row = &mut output[r * self.rows..(r + 1) * self.rows];
                self.project_f32(y_row, x_row)?;
            }
            Ok(())
        }
    }

    /// Evaluates batched multi-token GEMM (`[M, cols] x [rows, cols]^T -> [M, rows]`) from a
    /// pre-quantized `QuantizedActQ8` activation buffer via `gemm_repacked_rows`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    #[cfg(feature = "with-kernels")]
    pub fn project_batch_act(
        &self,
        output: &mut [f32],
        act: &QuantizedActQ8,
    ) -> Result<(), K3Error> {
        gemm_repacked_rows(output, &self.inner, act, 0..self.inner.rows())
    }

    /// Evaluates batched multi-token GEMM (`[M, cols] x [rows, cols]^T -> [M, rows]`) from a
    /// prepared [`DoryBatchAct`] scratch buffer, reusing activation quantization across sibling
    /// projections (`Wq`/`Wk`/`Wv`/`Wgate` and `Wup`/`Wgate`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions mismatch.
    pub fn project_batch_prepared(
        &self,
        output: &mut [f32],
        act: &DoryBatchAct,
    ) -> Result<(), K3Error> {
        #[cfg(feature = "with-kernels")]
        {
            self.project_batch_act(output, &act.inner)
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.project_batch_f32(output, &act.raw, act.m)
        }
    }
}

/// Reusable batched `[M, K]` activation scratch container shared across sibling GEMM projections.
///
/// Under `with-kernels`, wraps `QuantizedActQ8` (`quantize_into`) so sibling projections (`Wq`, `Wk`,
/// `Wv`, `Wgate` or `Wup`, `Wgate`) quantize activations once per sub-layer while reusing heap
/// allocations across layers. Under `--no-default-features`, retains validated row-major `f32`
/// activations for uniform batched execution.
#[derive(Debug, Clone, Default)]
pub struct DoryBatchAct {
    #[cfg(feature = "with-kernels")]
    inner: QuantizedActQ8,
    #[cfg(not(feature = "with-kernels"))]
    raw: Vec<f32>,
    m: usize,
    k: usize,
}

impl DoryBatchAct {
    /// Creates an empty reusable [`DoryBatchAct`] buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "with-kernels")]
            inner: QuantizedActQ8::new(),
            #[cfg(not(feature = "with-kernels"))]
            raw: Vec::new(),
            m: 0,
            k: 0,
        }
    }

    /// Prepares `[m, k]` row-major `f32` activations into `self`, reusing prior allocations.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `m == 0`, `k == 0`, or `act.len() != m * k`.
    pub fn prepare_into(&mut self, act: &[f32], m: usize, k: usize) -> Result<(), K3Error> {
        let expected = m.checked_mul(k).ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "DoryBatchAct".to_owned(),
            offset: 0,
            byte_len: size_of_val(act),
            buffer_len: usize::MAX,
            reason: "m * k overflow preparing DoryBatchAct".to_owned(),
        })?;
        if m == 0 || k == 0 || act.len() != expected {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "DoryBatchAct".to_owned(),
                offset: 0,
                byte_len: size_of_val(act),
                buffer_len: expected.saturating_mul(size_of::<f32>()),
                reason: format!(
                    "batched activation length {} != m ({m}) * k ({k})",
                    act.len()
                ),
            });
        }
        #[cfg(feature = "with-kernels")]
        {
            self.inner.quantize_into(act, m, k)?;
        }
        #[cfg(not(feature = "with-kernels"))]
        {
            self.raw.clear();
            self.raw.extend_from_slice(act);
        }
        self.m = m;
        self.k = k;
        Ok(())
    }

    /// Returns the number of token rows `M` currently stored.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.m
    }

    /// Returns the logical feature width `K` currently stored.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.k
    }
}

/// Physical sub-layer cache entry holding `128-byte`-aligned repacked matrices and pre-decoded
/// `nGPT` 1D scale vectors shared across all `n_loops` recurrent virtual steps.
#[cfg_attr(
    feature = "with-kernels",
    expect(
        clippy::large_enum_variant,
        reason = "Attention sub-layer stores 5 repacked projections and 3 aligned scale vectors vs 3 + 3 for MLP"
    )
)]
#[derive(Debug, Clone)]
pub enum DoryPhysicalSubLayer {
    /// Disaggregated attention sub-layer (`'S'` sliding-window or `'*'` / `'g'` full-context).
    Attention {
        /// Physical sub-layer index (`0..physical_layer_count`).
        physical_idx: u32,
        /// Attention sub-layer kind (`SlidingAttention` or `FullAttention`).
        kind: DorySubLayerKind,
        /// Repacked query projection (`blk.{p}.attn_q.weight`, `[q_dim, hidden_dim]`).
        wq: DoryCachedWeight,
        /// Repacked key projection (`blk.{p}.attn_k.weight`, `[kv_dim, hidden_dim]`).
        wk: DoryCachedWeight,
        /// Repacked value projection (`blk.{p}.attn_v.weight`, `[kv_dim, hidden_dim]`).
        wv: DoryCachedWeight,
        /// Repacked sigmoid gate linear projection (`blk.{p}.attn_gate.weight`, `[q_dim, hidden_dim]`).
        wgate: DoryCachedWeight,
        /// Repacked attention output projection (`blk.{p}.attn_output.weight`, `[hidden_dim, q_dim]`).
        wo: DoryCachedWeight,
        /// Pre-decoded `128-byte`-aligned QK scale vector (`blk.{p}.attn_qk_scale.weight`).
        sqk: AlignedF32Vec,
        /// Pre-decoded `128-byte`-aligned eigen-rate residual vector (`blk.{p}.residual_alpha.weight`).
        attn_alpha: AlignedF32Vec,
        /// Pre-decoded `128-byte`-aligned sigmoid gate scale vector (`blk.{p}.attn_gate_scale.weight`).
        attn_gate: AlignedF32Vec,
    },
    /// Disaggregated gated `SwiGLU` MLP sub-layer (`'-'`).
    Mlp {
        /// Physical sub-layer index (`0..physical_layer_count`).
        physical_idx: u32,
        /// Repacked up projection (`blk.{p}.ffn_up.weight`, `[ffn_dim, hidden_dim]`).
        wup: DoryCachedWeight,
        /// Repacked gate projection (`blk.{p}.ffn_gate.weight`, `[ffn_dim, hidden_dim]`).
        wgate: DoryCachedWeight,
        /// Repacked down projection (`blk.{p}.ffn_down.weight`, `[hidden_dim, ffn_dim]`).
        wdown: DoryCachedWeight,
        /// Pre-decoded `128-byte`-aligned up scale vector (`blk.{p}.ffn_up_scale.weight`).
        suv: AlignedF32Vec,
        /// Pre-decoded `128-byte`-aligned eigen-rate residual vector (`blk.{p}.residual_alpha.weight`).
        mlp_alpha: AlignedF32Vec,
        /// Pre-decoded `128-byte`-aligned gate scale vector (`blk.{p}.ffn_gate_scale.weight`).
        mlp_gate: AlignedF32Vec,
    },
}

impl DoryPhysicalSubLayer {
    /// Extracts, repacks, and pre-decodes one physical sub-layer (`0..physical_layer_count`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if sub-layer metadata, dimensions, or tensor bytes are missing or invalid.
    pub fn extract(
        recipe: &DoryRecipe,
        physical_idx: u32,
        weights: &dyn ModelWeights,
    ) -> Result<Self, K3Error> {
        let kind = recipe.sublayer_kind(physical_idx)?;
        match kind {
            DorySubLayerKind::SlidingAttention | DorySubLayerKind::FullAttention => {
                Self::extract_attention_with_kind(recipe, physical_idx, kind, weights)
            }
            DorySubLayerKind::Mlp => Self::extract_mlp_with_kind(recipe, physical_idx, weights),
        }
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "architectural dims fit cleanly in f32"
    )]
    pub(crate) fn extract_attention_with_kind(
        recipe: &DoryRecipe,
        physical_idx: u32,
        kind: DorySubLayerKind,
        weights: &dyn ModelWeights,
    ) -> Result<Self, K3Error> {
        let hidden_dim = (recipe.hidden_dim as usize).max(2);
        let attn_alpha = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.residual_alpha.weight"),
            hidden_dim,
            recipe.alpha_init,
        )?;
        let num_heads = (recipe.num_heads as usize).max(1);
        let num_kv_heads = (recipe.num_kv_heads as usize).max(1).min(num_heads);
        let head_dim = (recipe.head_dim as usize).max(2);
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let wq = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.attn_q.weight"),
            q_dim,
            hidden_dim,
        )?;
        let wk = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.attn_k.weight"),
            kv_dim,
            hidden_dim,
        )?;
        let wv = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.attn_v.weight"),
            kv_dim,
            hidden_dim,
        )?;
        let wgate = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.attn_gate.weight"),
            q_dim,
            hidden_dim,
        )?;
        let wo = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.attn_output.weight"),
            hidden_dim,
            q_dim,
        )?;

        let sqk = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.attn_qk_scale.weight"),
            q_dim,
            recipe.init_std / (head_dim as f32).sqrt(),
        )?;
        let attn_gate = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.attn_gate_scale.weight"),
            q_dim,
            recipe.init_std / recipe.gate_init.max(1e-6).sqrt(),
        )?;

        Ok(Self::Attention {
            physical_idx,
            kind,
            wq,
            wk,
            wv,
            wgate,
            wo,
            sqk,
            attn_alpha,
            attn_gate,
        })
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "architectural dims fit cleanly in f32"
    )]
    pub(crate) fn extract_mlp_with_kind(
        recipe: &DoryRecipe,
        physical_idx: u32,
        weights: &dyn ModelWeights,
    ) -> Result<Self, K3Error> {
        let hidden_dim = (recipe.hidden_dim as usize).max(2);
        let mlp_alpha = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.residual_alpha.weight"),
            hidden_dim,
            recipe.alpha_init,
        )?;
        let ffn_dim = (recipe.ffn_dim as usize).max(1);

        let wup = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.ffn_up.weight"),
            ffn_dim,
            hidden_dim,
        )?;
        let wgate = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.ffn_gate.weight"),
            ffn_dim,
            hidden_dim,
        )?;
        let wdown = DoryCachedWeight::load_from_weights(
            weights,
            &format!("blk.{physical_idx}.ffn_down.weight"),
            hidden_dim,
            ffn_dim,
        )?;

        let sqrt_embd = (hidden_dim as f32).sqrt();
        let suv = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.ffn_up_scale.weight"),
            ffn_dim,
            1.0_f32 / sqrt_embd,
        )?;
        let mlp_gate = AlignedF32Vec::decode_tensor_or_default(
            weights,
            &format!("blk.{physical_idx}.ffn_gate_scale.weight"),
            ffn_dim,
            1.0_f32 / sqrt_embd,
        )?;

        Ok(Self::Mlp {
            physical_idx,
            wup,
            wgate,
            wdown,
            suv,
            mlp_alpha,
            mlp_gate,
        })
    }

    /// Physical sub-layer index (`0..physical_layer_count`).
    #[must_use]
    pub const fn physical_idx(&self) -> u32 {
        match self {
            Self::Attention { physical_idx, .. } | Self::Mlp { physical_idx, .. } => *physical_idx,
        }
    }

    /// Borrows the pre-decoded `128-byte`-aligned eigen-rate residual interpolation vector (`residual_alpha`).
    #[must_use]
    pub const fn residual_alpha(&self) -> &AlignedF32Vec {
        match self {
            Self::Attention { attn_alpha, .. } => attn_alpha,
            Self::Mlp { mlp_alpha, .. } => mlp_alpha,
        }
    }

    /// Number of repacked 2D projection matrices stored in this physical sub-layer (`5` for attention, `3` for MLP).
    #[must_use]
    pub const fn repacked_matrix_count(&self) -> usize {
        match self {
            Self::Attention { .. } => 5,
            Self::Mlp { .. } => 3,
        }
    }

    /// Total `128-byte`-aligned byte footprint across projection matrices and cached scale vectors.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        match self {
            Self::Attention {
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
                wq.aligned_bytes()
                    + wk.aligned_bytes()
                    + wv.aligned_bytes()
                    + wgate.aligned_bytes()
                    + wo.aligned_bytes()
                    + sqk.aligned_bytes()
                    + attn_alpha.aligned_bytes()
                    + attn_gate.aligned_bytes()
            }
            Self::Mlp {
                wup,
                wgate,
                wdown,
                suv,
                mlp_alpha,
                mlp_gate,
                ..
            } => {
                wup.aligned_bytes()
                    + wgate.aligned_bytes()
                    + wdown.aligned_bytes()
                    + suv.aligned_bytes()
                    + mlp_alpha.aligned_bytes()
                    + mlp_gate.aligned_bytes()
            }
        }
    }
}

/// Dory1.0 `ChatML` (`bos = "<s>"`, `eos = "<|im_end|>"`) and `<think>` / `</think>`
/// reasoning boundary token metadata aligned with `jero-tokenizer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoryStopMetadata {
    /// Beginning-of-sequence token ID (`tokenizer.ggml.bos_token_id` or `"<s>"`).
    pub bos_token_id: Option<u32>,
    /// Primary end-of-sequence token ID (`tokenizer.ggml.eos_token_id`).
    pub eos_token_id: Option<u32>,
    /// Optional end-of-turn token ID (`tokenizer.ggml.eot_token_id`).
    pub eot_token_id: Option<u32>,
    /// `ChatML` turn-start token ID (`"<|im_start|>"`) resolved from vocabulary if present.
    pub im_start_token_id: Option<u32>,
    /// `ChatML` end-of-turn stop token ID (`"<|im_end|>"`) resolved from vocabulary if present.
    pub im_end_token_id: Option<u32>,
    /// Reasoning span opener token ID (`"<think>"`) resolved from vocabulary if present.
    pub think_open_token_id: Option<u32>,
    /// Reasoning span closer token ID (`"</think>"`) resolved from vocabulary if present.
    pub think_close_token_id: Option<u32>,
    /// Deduplicated generation-halting token IDs (contains EOS, EOT, and `<|im_end|>`, never `<think>`/`</think>`).
    pub stop_token_ids: Vec<u32>,
    /// Canonical text stop sequences (`"<|im_end|>"`, `"</s>"`).
    pub stop_sequences: Cow<'static, [&'static str]>,
    /// Whether `tokenizer.chat_template` / special tokens align with `ChatML` `<|im_end|>` semantics.
    pub chatml_template_aligned: bool,
    /// Whether `<think>` / `</think>` reasoning markers are present in metadata or vocabulary.
    pub has_think_tokens: bool,
}

impl Default for DoryStopMetadata {
    fn default() -> Self {
        Self::new()
    }
}

impl DoryStopMetadata {
    /// Constructs default Dory1.0 `ChatML` stop metadata (`<|im_end|>` and `</s>` text sequences).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bos_token_id: None,
            eos_token_id: None,
            eot_token_id: None,
            im_start_token_id: None,
            im_end_token_id: None,
            think_open_token_id: None,
            think_close_token_id: None,
            stop_token_ids: Vec::new(),
            stop_sequences: Cow::Borrowed(&[DORY_EOS_TOKEN, "</s>"]),
            chatml_template_aligned: false,
            has_think_tokens: false,
        }
    }

    /// Hydrates and validates Dory1.0 `ChatML` + `<think>` boundary token metadata from a [`GgufFile`].
    ///
    /// Note that `<think>` and `</think>` are tracked separately as reasoning boundaries and are
    /// intentionally **excluded** from [`Self::stop_token_ids`] so generation continues past
    /// `</think>` until `<|im_end|>` / EOS.
    ///
    /// # Errors
    ///
    /// Currently infallible on valid [`GgufFile`] headers, returning [`K3Error`] only if malformed
    /// tokenizer stop metadata is encountered in future strict modes.
    pub fn from_gguf(gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut meta = Self::new();
        meta.bos_token_id = gguf
            .metadata_value("tokenizer.ggml.bos_token_id")
            .and_then(jero_core::GgufValue::as_u32);
        meta.eos_token_id = gguf
            .metadata_value("tokenizer.ggml.eos_token_id")
            .and_then(jero_core::GgufValue::as_u32);
        meta.eot_token_id = gguf
            .metadata_value("tokenizer.ggml.eot_token_id")
            .and_then(jero_core::GgufValue::as_u32);

        let mut legacy_eos_id = None;
        if let Some(tokens_val) = gguf.metadata_value("tokenizer.ggml.tokens")
            && let Some(GgufArray::String(tokens)) = tokens_val.as_array()
        {
            for (idx, &tok) in tokens.iter().enumerate() {
                let Ok(id) = u32::try_from(idx) else {
                    break;
                };
                match tok {
                    DORY_BOS_TOKEN if meta.bos_token_id.is_none() => meta.bos_token_id = Some(id),
                    DORY_IM_START_TOKEN => meta.im_start_token_id = Some(id),
                    DORY_EOS_TOKEN => meta.im_end_token_id = Some(id),
                    DORY_THINK_OPEN_TOKEN => meta.think_open_token_id = Some(id),
                    DORY_THINK_CLOSE_TOKEN => meta.think_close_token_id = Some(id),
                    "</s>" => legacy_eos_id = Some(id),
                    _ => {}
                }
            }
        }

        for candidate in [
            meta.eos_token_id,
            meta.eot_token_id,
            meta.im_end_token_id,
            if meta.eos_token_id.is_none() && meta.im_end_token_id.is_none() {
                legacy_eos_id
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
        {
            if !meta.stop_token_ids.contains(&candidate) {
                meta.stop_token_ids.push(candidate);
            }
        }

        let tmpl = gguf
            .metadata_value("tokenizer.chat_template")
            .and_then(jero_core::GgufValue::as_str)
            .unwrap_or_default();

        meta.chatml_template_aligned = meta.im_end_token_id.is_some()
            || tmpl.contains(DORY_EOS_TOKEN)
            || tmpl.contains(DORY_IM_START_TOKEN);
        meta.has_think_tokens = meta.think_open_token_id.is_some()
            || meta.think_close_token_id.is_some()
            || tmpl.contains(DORY_THINK_OPEN_TOKEN);

        Ok(meta)
    }

    /// Returns `true` when `token_id` halts autoregressive generation (`eos` / `eot` / `<|im_end|>`).
    #[must_use]
    pub fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    /// Returns `true` when `token_id` marks `<think>` or `</think>` reasoning boundaries.
    #[must_use]
    pub fn is_reasoning_token(&self, token_id: u32) -> bool {
        Some(token_id) == self.think_open_token_id || Some(token_id) == self.think_close_token_id
    }

    /// Constructs a [`jero_core::StopCriteria`] bound to the deduplicated generation stop token IDs.
    #[must_use]
    pub fn stop_criteria(&self, max_tokens: usize, n_ctx: usize) -> StopCriteria<'_> {
        StopCriteria::new(&self.stop_token_ids, max_tokens, n_ctx)
    }

    /// Formats a user prompt using Dory1.0's canonical `ChatML` + `<think>` template.
    #[must_use]
    pub fn format_chatml_prompt(
        &self,
        user_prompt: &str,
        reasoning_effort: Option<&str>,
        enable_thinking: bool,
    ) -> String {
        let is_medium = reasoning_effort
            .map(str::trim)
            .is_some_and(|e| e.eq_ignore_ascii_case("medium"));
        let is_low = reasoning_effort
            .map(str::trim)
            .is_some_and(|e| e.eq_ignore_ascii_case("low"));

        let effective_thinking = enable_thinking && !is_low;
        let user_turn = if is_medium {
            format!("{user_prompt}{DORY_MEDIUM_REASONING_SUFFIX}")
        } else {
            user_prompt.to_owned()
        };

        if effective_thinking {
            format!(
                "{DORY_BOS_TOKEN}{DORY_IM_START_TOKEN}user\n{user_turn}{DORY_EOS_TOKEN}\n{DORY_IM_START_TOKEN}assistant\n{DORY_THINK_OPEN_TOKEN}\n"
            )
        } else {
            format!(
                "{DORY_BOS_TOKEN}{DORY_IM_START_TOKEN}user\n{user_turn}{DORY_EOS_TOKEN}\n{DORY_IM_START_TOKEN}assistant\n{DORY_THINK_OPEN_TOKEN}{DORY_THINK_CLOSE_TOKEN}"
            )
        }
    }
}

/// Model-scoped physical sub-layer weight and scale-vector cache for [`DoryRecipe`].
///
/// Stores strictly `recipe.physical_layer_count()` entries (`0..40` or `0..72`, never `112`
/// duplicated virtual copies), preserving `RLIMIT_MEMLOCK` headroom while avoiding per-token
/// block repacking and per-step `extract_scale_vector_f32` calls.
#[derive(Debug, Clone)]
pub struct DoryPhysicalCache {
    sublayers: Vec<DoryPhysicalSubLayer>,
    output_scale: Option<AlignedF32Vec>,
    stop_metadata: DoryStopMetadata,
}

impl DoryPhysicalCache {
    /// Builds a physical cache from `recipe` and `weights`, repacking all `0..recipe.physical_layer_count()`
    /// projection matrices once and pre-decoding all 1D `nGPT` scale vectors into `128-byte`-aligned
    /// `f32` storage.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `weights` is empty (`tensor_count() == 0`), any required sub-layer
    /// tensor is missing, dimensions mismatch, or quantization formats are unsupported.
    pub fn build(recipe: &DoryRecipe, weights: &dyn ModelWeights) -> Result<Self, K3Error> {
        recipe.validate_hyperparams()?;
        if weights.tensor_count() == 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "blk.0.attn_q.weight".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: "DoryPhysicalCache::build requires loaded model weights (tensor_count > 0)"
                    .to_owned(),
            });
        }

        let n_physical = recipe.physical_layer_count();
        let mut sublayers = Vec::with_capacity(n_physical as usize);
        for p in 0..n_physical {
            sublayers.push(DoryPhysicalSubLayer::extract(recipe, p, weights)?);
        }

        let hidden_dim = (recipe.hidden_dim as usize).max(2);
        let output_scale = if weights.contains_tensor("output_scale.weight") {
            let vocab_rows = if let Ok(out_w) = weights.tensor("output.weight") {
                quant::resolve_2d_dims(&out_w, hidden_dim)?.0
            } else if let Ok(embd_w) = weights.tensor("token_embd.weight") {
                quant::resolve_2d_dims(&embd_w, hidden_dim)?.0
            } else {
                recipe.vocab_size
            };
            Some(AlignedF32Vec::decode_tensor(
                weights,
                "output_scale.weight",
                vocab_rows,
            )?)
        } else {
            None
        };

        Ok(Self {
            sublayers,
            output_scale,
            stop_metadata: recipe.stop_metadata.clone(),
        })
    }

    /// Builds a [`DoryPhysicalCache`] directly from a parsed [`GgufFile`], synchronizing both the
    /// repacked physical sub-layers and `ChatML` / `<think>` stop-token metadata.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] if `recipe` hyperparameters, tensors, or tokenizer metadata are invalid.
    pub fn from_gguf(recipe: &DoryRecipe, gguf: &GgufFile<'_>) -> Result<Self, K3Error> {
        let mut cache = Self::build(recipe, gguf)?;
        cache.stop_metadata = DoryStopMetadata::from_gguf(gguf)?;
        Ok(cache)
    }

    /// Overrides the aligned `ChatML` / `<think>` stop-token metadata stored on this cache.
    #[must_use]
    pub fn with_stop_metadata(mut self, stop_metadata: DoryStopMetadata) -> Self {
        self.stop_metadata = stop_metadata;
        self
    }

    /// Number of physical sub-layers cached (`recipe.physical_layer_count()`, never `virtual_layer_count()`).
    #[must_use]
    pub fn len(&self) -> usize {
        self.sublayers.len()
    }

    /// Returns `true` if no physical sub-layers are cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sublayers.is_empty()
    }

    /// Borrows all cached physical sub-layers (`0..physical_layer_count`).
    #[must_use]
    pub fn sublayers(&self) -> &[DoryPhysicalSubLayer] {
        &self.sublayers
    }

    /// Borrows the cached physical sub-layer at `physical_idx`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `physical_idx >= self.len()`.
    pub fn get(&self, physical_idx: u32) -> Result<&DoryPhysicalSubLayer, K3Error> {
        self.sublayers
            .get(physical_idx as usize)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: format!("blk.{physical_idx}"),
                offset: physical_idx as usize,
                byte_len: 0,
                buffer_len: self.sublayers.len(),
                reason: format!(
                    "physical_idx {physical_idx} out of bounds for DoryPhysicalCache len {}",
                    self.sublayers.len()
                ),
            })
    }

    /// Borrows the optional pre-decoded `output_scale.weight` vector (`128-byte`-aligned).
    #[must_use]
    pub const fn output_scale(&self) -> Option<&AlignedF32Vec> {
        self.output_scale.as_ref()
    }

    /// Borrows the aligned `ChatML` (`<|im_end|>`) and `<think>` stop-token metadata.
    #[must_use]
    pub const fn stop_metadata(&self) -> &DoryStopMetadata {
        &self.stop_metadata
    }

    /// Total number of repacked 2D projection matrices stored across all cached physical sub-layers.
    #[must_use]
    pub fn repacked_matrix_count(&self) -> usize {
        self.sublayers
            .iter()
            .map(DoryPhysicalSubLayer::repacked_matrix_count)
            .sum()
    }

    /// Total `128-byte`-aligned bytes held by this physical cache (projection tiles + scale vectors).
    #[must_use]
    pub fn total_aligned_bytes(&self) -> usize {
        let sub_bytes: usize = self
            .sublayers
            .iter()
            .map(DoryPhysicalSubLayer::aligned_bytes)
            .sum();
        let out_bytes = self
            .output_scale
            .as_ref()
            .map_or(0, AlignedF32Vec::aligned_bytes);
        sub_bytes + out_bytes
    }
}
