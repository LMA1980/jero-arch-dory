//! Quantized block dequantization, geometry validation, and projection kernels for Dory (`dory`).
//!
//! Dispatches to `jero-kernels-rvv` hardware/scalar kernels when `with-kernels` is enabled, and
//! executes bit-equivalent portable scalar math across `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`,
//! `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` when built with `--no-default-features`.

use jero_core::{K3Error, QuantType, TensorView};
#[cfg(not(miri))]
use rayon::prelude::*;

#[cfg(feature = "with-kernels")]
pub(crate) use jero_kernels_rvv::bf16_to_f32_finite;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, BlockQ8_0, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
    Q4_K_M_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, QuantizedActQ8View, RepackedWeight,
    VectorKernel, dequantize_token_row, f16_to_f32_finite, gemv_q4_0_q8_0, gemv_q5_0_q8_0,
    quantize_row_q8_0, vec_dot_bf16, vec_dot_f16, vec_dot_q4_0,
};

#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q4_0_BLOCK_BYTES: usize = 18;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_ELEMENTS: usize = 32;
#[cfg(not(feature = "with-kernels"))]
const Q8_0_BLOCK_BYTES: usize = 34;
#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q4_K_M_BLOCK_BYTES: usize = 144;
#[cfg(not(feature = "with-kernels"))]
const Q5_K_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q5_K_BLOCK_BYTES: usize = 176;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_ELEMENTS: usize = 256;
#[cfg(not(feature = "with-kernels"))]
const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of scalar elements per `Q5_0` block.
const Q5_0_BLOCK_ELEMENTS: usize = 32;
/// Storage size in bytes of a single `Q5_0` block (`2B f16` scale + `4B qh` + `16B qs`).
const Q5_0_BLOCK_BYTES: usize = 22;

#[cfg(not(feature = "with-kernels"))]
#[inline]
pub(crate) fn f16_to_f32_finite(bits: u16) -> f32 {
    let sign = u32::from((bits >> 15) & 0x0001);
    let exp = u32::from((bits >> 10) & 0x001F);
    let mant = u32::from(bits & 0x03FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 127_u32 - 14;
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
#[inline]
pub(crate) fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0x00FF;
    let sign = (bits >> 15) & 0x0001;
    if exp == 0x00FF {
        return if sign == 1 { -0.0625 } else { 0.0625 };
    }
    f32::from_bits(u32::from(bits) << 16)
}

#[inline]
fn sanitize_f32(val: f32) -> f32 {
    if val.is_finite() { val } else { 0.0 }
}

/// Returns `(block_elements, block_bytes)` for all Dory admitted quantization formats.
#[must_use]
pub(crate) const fn quant_block_geometry(quant: QuantType) -> Option<(usize, usize)> {
    match quant {
        QuantType::F32 => Some((1, 4)),
        QuantType::F16 | QuantType::BF16 => Some((1, 2)),
        QuantType::Q4_0 => Some((Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES)),
        QuantType::Q5_0 => Some((Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES)),
        QuantType::Q8_0 => Some((Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES)),
        QuantType::Q4_K_M => Some((Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES)),
        QuantType::Q5_K => Some((Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES)),
        QuantType::Q6_K => Some((Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES)),
        _ => None,
    }
}

/// Computes the byte stride of one contiguous row of `cols` elements for `quant`.
pub(crate) fn quant_row_bytes(tensor: &TensorView<'_>, cols: usize) -> Result<usize, K3Error> {
    let Some((blk_elems, blk_bytes)) = quant_block_geometry(tensor.quant()) else {
        return Err(K3Error::unsupported_tensor_quant(
            "dory",
            tensor.name(),
            tensor.quant(),
        ));
    };
    if cols == 0 || cols % blk_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "column width {cols} is not a positive multiple of {:?} block size {blk_elems}",
                tensor.quant()
            ),
        });
    }
    (cols / blk_elems)
        .checked_mul(blk_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: "quantized row byte stride overflow".to_owned(),
        })
}

/// Resolves `(rows, cols)` of a 2D tensor given `expected_cols`, supporting both GGUF
/// `[cols, rows]` (`ne[0] = cols`) and `[rows, cols]`.
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] or [`K3Error::CapabilityMismatch`] if `tensor`
/// is non-2D, does not match `expected_cols`, uses an unsupported quant, or lacks bytes.
pub(crate) fn resolve_2d_dims(
    tensor: &TensorView<'_>,
    expected_cols: usize,
) -> Result<(usize, usize), K3Error> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("expected 2D tensor, got shape {shape:?}"),
        });
    }
    let (d0, d1) = (shape[0], shape[1]);
    let (rows, cols) = if d0 == expected_cols {
        (d1, d0)
    } else if d1 == expected_cols {
        (d0, d1)
    } else {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "2D tensor shape {shape:?} does not match expected cols {expected_cols}"
            ),
        });
    };

    let row_bytes = quant_row_bytes(tensor, cols)?;
    let expected_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: "matrix byte length overflow".to_owned(),
        })?;
    if tensor.as_bytes().len() < expected_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: expected_bytes,
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "tensor byte length {} smaller than required {expected_bytes}",
                tensor.as_bytes().len()
            ),
        });
    }
    Ok((rows, cols))
}

/// Validates a 2D weight tensor has exact `[expected_rows, expected_cols]` dimensions (in either
/// `[cols, rows]` or `[rows, cols]` layout) and sufficient quantized storage.
pub(crate) fn validate_2d_dims(
    tensor: &TensorView<'_>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), K3Error> {
    let (rows, _cols) = resolve_2d_dims(tensor, expected_cols)?;
    if rows != expected_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!(
                "weight matrix output rows {rows} != expected rows {expected_rows} (shape {:?})",
                tensor.shape()
            ),
        });
    }
    Ok(())
}

/// Dequantizes a 22-byte `Q5_0` block into 32 `f32` elements matching GGML `dequantize_row_q5_0`.
pub(crate) fn dequantize_q5_0_block(
    block: &[u8; Q5_0_BLOCK_BYTES],
    out: &mut [f32; Q5_0_BLOCK_ELEMENTS],
) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for (j, &packed) in qs.iter().enumerate() {
        let xh0 = u8::try_from(((qh >> j) & 1) << 4).unwrap_or(0);
        let xh1 = u8::try_from(((qh >> (j + 16)) & 1) << 4).unwrap_or(0);
        #[expect(
            clippy::cast_possible_wrap,
            reason = "5-bit unsigned 0..31 - 16 fits i8"
        )]
        let q0 = ((packed & 0x0F) | xh0) as i8 - 16;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "5-bit unsigned 0..31 - 16 fits i8"
        )]
        let q1 = (((packed >> 4) & 0x0F) | xh1) as i8 - 16;
        out[j] = d * f32::from(q0);
        out[j + 16] = d * f32::from(q1);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q4_0_block(block: &[u8; Q4_0_BLOCK_BYTES], out: &mut [f32; Q4_0_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for (i, &byte) in block[2..18].iter().enumerate() {
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble 0..15 - 8 fits i8")]
        let q0 = (byte & 0x0F) as i8 - 8;
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble 0..15 - 8 fits i8")]
        let q1 = ((byte >> 4) & 0x0F) as i8 - 8;
        out[i] = d * f32::from(q0);
        out[i + 16] = d * f32::from(q1);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q8_0_block(block: &[u8; Q8_0_BLOCK_BYTES], out: &mut [f32; Q8_0_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    for (dst, &byte) in out.iter_mut().zip(&block[2..34]) {
        #[expect(clippy::cast_possible_wrap, reason = "Q8_0 byte represents signed i8")]
        let q = byte as i8;
        *dst = d * f32::from(q);
    }
}

#[cfg(not(feature = "with-kernels"))]
fn unpack_k_scales_6bit(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0_u8; 8];
    let mut m = [0_u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q4_k_m_block(
    block: &[u8; Q4_K_M_BLOCK_BYTES],
    out: &mut [f32; Q4_K_M_BLOCK_ELEMENTS],
) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }
    let (sc, m) = unpack_k_scales_6bit(&block[4..16]);
    let qs = &block[16..144];
    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);
        let out_idx = chunk * 64;
        let q_offset = chunk * 32;
        for l in 0..32 {
            out[out_idx + l] = d1 * f32::from(qs[q_offset + l] & 0x0F) - m1;
            out[out_idx + 32 + l] = d2 * f32::from((qs[q_offset + l] >> 4) & 0x0F) - m2;
        }
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q5_k_block(block: &[u8; Q5_K_BLOCK_BYTES], out: &mut [f32; Q5_K_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }
    let (sc, m) = unpack_k_scales_6bit(&block[4..16]);
    let qh = &block[16..48];
    let qs = &block[48..176];
    let mut u1 = 1_u8;
    let mut u2 = 2_u8;
    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);
        let out_idx = chunk * 64;
        let q_offset = chunk * 32;
        for l in 0..32 {
            let hb1 = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
            let hb2 = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
            out[out_idx + l] = d1 * f32::from((qs[q_offset + l] & 0x0F) + hb1) - m1;
            out[out_idx + 32 + l] = d2 * f32::from(((qs[q_offset + l] >> 4) & 0x0F) + hb2) - m2;
        }
        u1 <<= 2;
        u2 <<= 2;
    }
}

#[cfg(not(feature = "with-kernels"))]
fn dequantize_q6_k_block(block: &[u8; Q6_K_BLOCK_BYTES], out: &mut [f32; Q6_K_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[208], block[209]]));
    if d == 0.0 {
        out.fill(0.0);
        return;
    }
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    for (half_idx, &n) in [0_usize, 128_usize].iter().enumerate() {
        let low_base = half_idx * 64;
        let high_base = half_idx * 32;
        let sc_base = half_idx * 8;
        for l in 0..32 {
            let is = l / 16;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q1 = (((ql[low_base + l] & 0x0F) | ((qh[high_base + l] & 0x03) << 4)) as i8) - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q2 = (((ql[low_base + l + 32] & 0x0F) | (((qh[high_base + l] >> 2) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q3 = ((((ql[low_base + l] >> 4) & 0x0F) | (((qh[high_base + l] >> 4) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "6-bit signed shift -32")]
            let q4 = ((((ql[low_base + l + 32] >> 4) & 0x0F)
                | (((qh[high_base + l] >> 6) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc1 = f32::from(scales[sc_base + is] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc2 = f32::from(scales[sc_base + is + 2] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc3 = f32::from(scales[sc_base + is + 4] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
            let sc4 = f32::from(scales[sc_base + is + 6] as i8);
            out[n + l] = d * sc1 * f32::from(q1);
            out[n + l + 32] = d * sc2 * f32::from(q2);
            out[n + l + 64] = d * sc3 * f32::from(q3);
            out[n + l + 96] = d * sc4 * f32::from(q4);
        }
    }
}

#[inline]
pub(crate) fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "with-kernels")]
    {
        BaselineKernel::new().dot_product_f32(a, b)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        a.iter().zip(b).map(|(&x, &y)| x * y).sum()
    }
}

fn dequant_block_row_payload<const ELEMS: usize, const BYTES: usize>(
    tensor_name: &str,
    row_raw: &[u8],
    out: &mut [f32],
    decode: fn(&[u8; BYTES], &mut [f32; ELEMS]),
) -> Result<(), K3Error> {
    let mut block_out = [0.0_f32; ELEMS];
    for (blk_idx, out_chunk) in out.chunks_exact_mut(ELEMS).enumerate() {
        let start = blk_idx * BYTES;
        let block: &[u8; BYTES] =
            row_raw[start..start + BYTES]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: tensor_name.to_owned(),
                    offset: start,
                    byte_len: BYTES,
                    buffer_len: row_raw.len(),
                    reason: "quantized block slice length mismatch".to_owned(),
                })?;
        decode(block, &mut block_out);
        out_chunk.copy_from_slice(&block_out);
    }
    Ok(())
}

fn dot_block_rows<const ELEMS: usize, const BYTES: usize>(
    tensor: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
    output: &mut [f32],
    decode: fn(&[u8; BYTES], &mut [f32; ELEMS]),
) -> Result<(), K3Error> {
    let blocks_per_row = cols / ELEMS;
    let row_bytes = blocks_per_row * BYTES;
    let raw = tensor.as_bytes();
    let mut block_out = [0.0_f32; ELEMS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_raw = &raw[row * row_bytes..(row + 1) * row_bytes];
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let start = blk_idx * BYTES;
            let block: &[u8; BYTES] = row_raw[start..start + BYTES].try_into().map_err(|_| {
                K3Error::TensorBoundsError {
                    tensor_name: tensor.name().to_owned(),
                    offset: row * row_bytes + start,
                    byte_len: BYTES,
                    buffer_len: raw.len(),
                    reason: "quantized block slice length mismatch".to_owned(),
                }
            })?;
            decode(block, &mut block_out);
            let in_chunk = &input[blk_idx * ELEMS..(blk_idx + 1) * ELEMS];
            row_sum += dot_product_f32(&block_out, in_chunk);
        }
        *out_slot = row_sum;
    }
    Ok(())
}

fn decode_unquant_payload(
    tensor: &TensorView<'_>,
    raw: &[u8],
    out: &mut [f32],
) -> Result<(), K3Error> {
    let elem_bytes = match tensor.quant() {
        QuantType::F32 => 4,
        QuantType::F16 | QuantType::BF16 => 2,
        other => {
            return Err(K3Error::CapabilityMismatch {
                architecture: "dory".to_owned(),
                reason: format!(
                    "tensor '{}' uses unsupported unquantized format {other}",
                    tensor.name()
                ),
            });
        }
    };
    let required = out.len().saturating_mul(elem_bytes);
    if raw.len() < required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: required,
            buffer_len: raw.len(),
            reason: format!(
                "buffer len {} smaller than required {required} bytes",
                raw.len()
            ),
        });
    }
    match tensor.quant() {
        QuantType::F32 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(4)) {
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                *slot = sanitize_f32(val);
            }
        }
        QuantType::F16 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(2)) {
                *slot = f16_to_f32_finite(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        QuantType::BF16 => {
            for (slot, chunk) in out.iter_mut().zip(raw[..required].chunks_exact(2)) {
                *slot = bf16_to_f32_finite(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Decodes a single row (`cols` elements) from `tensor` (`token_embd.weight` or weight matrix)
/// across all admitted formats (`F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, `Q6_K`).
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on out-of-range row/shape mismatch or
/// [`K3Error::CapabilityMismatch`] on unsupported quantization types.
pub(crate) fn dequantize_row_to_f32(
    tensor: &TensorView<'_>,
    row: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if out.len() != cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: tensor.offset(),
            byte_len: tensor.byte_len(),
            buffer_len: tensor.as_bytes().len(),
            reason: format!("output row len {} != cols {cols}", out.len()),
        });
    }

    let (num_rows, _) = resolve_2d_dims(tensor, cols)?;
    if row >= num_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor.name().to_owned(),
            offset: row,
            byte_len: cols,
            buffer_len: num_rows,
            reason: format!("row index {row} out of bounds for {num_rows} rows"),
        });
    }

    #[cfg(feature = "with-kernels")]
    if !matches!(tensor.quant(), QuantType::Q5_0 | QuantType::F32) {
        return dequantize_token_row(tensor, row, out);
    }

    let row_bytes = quant_row_bytes(tensor, cols)?;
    let start = row * row_bytes;
    let row_raw = &tensor.as_bytes()[start..start + row_bytes];

    match tensor.quant() {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            decode_unquant_payload(tensor, row_raw, out)
        }
        QuantType::Q5_0 => dequant_block_row_payload::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q5_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q4_0 => dequant_block_row_payload::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q4_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q8_0 => dequant_block_row_payload::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q8_0_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q4_K_M => {
            dequant_block_row_payload::<Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES>(
                tensor.name(),
                row_raw,
                out,
                dequantize_q4_k_m_block,
            )
        }
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q5_K => dequant_block_row_payload::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q5_k_block,
        ),
        #[cfg(not(feature = "with-kernels"))]
        QuantType::Q6_K => dequant_block_row_payload::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            tensor.name(),
            row_raw,
            out,
            dequantize_q6_k_block,
        ),
        #[cfg(feature = "with-kernels")]
        other => Err(K3Error::unsupported_tensor_quant(
            "dory",
            tensor.name(),
            other,
        )),
        #[cfg(not(feature = "with-kernels"))]
        other => Err(K3Error::unsupported_tensor_quant(
            "dory",
            tensor.name(),
            other,
        )),
    }
}

/// Computes `output[0..expected_rows] = weights * input[0..expected_cols]` strictly matching
/// full matrix output row count across all admitted Dory quantization types.
pub(crate) fn project_matvec(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let expected_cols = input.len();
    let (rows, cols) = resolve_2d_dims(weights, expected_cols)?;
    if output.len() != rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "weight matrix output rows {rows} != expected output len {}",
                output.len()
            ),
        });
    }
    project_matvec_full(output, weights, rows, cols, input)
}

fn project_matvec_full(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        project_matvec_kernels(output, weights, rows, cols, input)
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        project_matvec_scalar(output, weights, rows, cols, input)
    }
}

#[cfg(feature = "with-kernels")]
fn project_matvec_kernels(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(output, weights, input),
        QuantType::Q8_0 => BaselineKernel::new().vec_dot_q8_0(output, weights, input),
        QuantType::Q6_K => BaselineKernel::new().vec_dot_q6_k(output, weights, input),
        QuantType::Q5_K => BaselineKernel::new().vec_dot_q5_k(output, weights, input),
        QuantType::Q4_0 => vec_dot_q4_0(output, weights, input),
        QuantType::F16 => vec_dot_f16(output, weights, input),
        QuantType::BF16 => vec_dot_bf16(output, weights, input),
        QuantType::Q5_0 => dot_block_rows::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_0_block,
        ),
        QuantType::F32 => project_f32_rows(output, weights, rows, cols, input),
        other => Err(K3Error::unsupported_tensor_quant(
            "dory",
            weights.name(),
            other,
        )),
    }
}

#[cfg(not(feature = "with-kernels"))]
fn project_matvec_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::F32 => project_f32_rows(output, weights, rows, cols, input),
        QuantType::F16 | QuantType::BF16 => project_half_raw_rows(
            output,
            weights.name(),
            weights.quant(),
            weights.as_bytes(),
            rows,
            cols,
            input,
        ),
        QuantType::Q4_0 => dot_block_rows::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q4_0_block,
        ),
        QuantType::Q5_0 => dot_block_rows::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_0_block,
        ),
        QuantType::Q8_0 => dot_block_rows::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q8_0_block,
        ),
        QuantType::Q4_K_M => dot_block_rows::<Q4_K_M_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q4_k_m_block,
        ),
        QuantType::Q5_K => dot_block_rows::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q5_k_block,
        ),
        QuantType::Q6_K => dot_block_rows::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            weights,
            rows,
            cols,
            input,
            output,
            dequantize_q6_k_block,
        ),
        other => Err(K3Error::unsupported_tensor_quant(
            "dory",
            weights.name(),
            other,
        )),
    }
}

fn project_f32_rows(
    output: &mut [f32],
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    project_f32_raw_rows(
        output,
        weights.name(),
        weights.as_bytes(),
        rows,
        cols,
        input,
    )
}

/// Output-row chunk multiple guaranteeing aligned 4-row/8-row micro-kernel equivalence across
/// `1`, `4`, and `8` worker partitions.
pub(crate) const PARTITION_ROW_ALIGN: usize = 8;

/// Descriptor for a validated row-major 2D weight payload (`total_rows x cols`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct RawWeightMatrix<'a> {
    pub name: &'a str,
    pub quant: QuantType,
    pub total_rows: usize,
    pub cols: usize,
    pub raw: &'a [u8],
}

impl<'a> RawWeightMatrix<'a> {
    pub(crate) fn from_tensor_view(
        view: &'a TensorView<'a>,
        expected_rows: usize,
        expected_cols: usize,
    ) -> Result<Self, K3Error> {
        validate_2d_dims(view, expected_rows, expected_cols)?;
        let row_bytes = quant_row_bytes(view, expected_cols)?;
        let req_bytes =
            expected_rows
                .checked_mul(row_bytes)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: view.name().to_owned(),
                    offset: view.offset(),
                    byte_len: view.byte_len(),
                    buffer_len: view.as_bytes().len(),
                    reason: "matrix byte length overflow".to_owned(),
                })?;
        let raw = view.as_bytes();
        if raw.len() < req_bytes {
            return Err(K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: req_bytes,
                buffer_len: raw.len(),
                reason: format!(
                    "matrix raw payload {} shorter than required {req_bytes}",
                    raw.len()
                ),
            });
        }
        Ok(Self {
            name: view.name(),
            quant: view.quant(),
            total_rows: expected_rows,
            cols: expected_cols,
            raw,
        })
    }
}

/// Quantized `M = 1` activation views (`Q8_0`) reused across sibling projections.
#[derive(Debug, Clone, Copy)]
pub(crate) struct M1Activation<'a> {
    pub input: &'a [f32],
    #[cfg(feature = "with-kernels")]
    pub q8_0: &'a [BlockQ8_0],
}

#[cfg(not(feature = "with-kernels"))]
impl<'a> M1Activation<'a> {
    pub(crate) const fn f32_only(input: &'a [f32]) -> Self {
        Self { input }
    }
}

/// Returns `true` when `quant` consumes `BlockQ8_0` (`32`-element) quantized activations on raw LM heads.
#[inline]
pub(crate) const fn uses_q8_0_activation(quant: QuantType) -> bool {
    matches!(quant, QuantType::Q4_0 | QuantType::Q5_0)
}

/// Required byte length for `BlockQ8_0` scratch accommodating `max_cols` elements.
#[inline]
pub(crate) const fn q8_0_scratch_bytes(max_cols: usize) -> usize {
    #[cfg(feature = "with-kernels")]
    {
        (max_cols / Q8_0_BLOCK_ELEMENTS).saturating_mul(size_of::<BlockQ8_0>())
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = max_cols;
        0
    }
}

/// Converts a `128-byte`-aligned zero-initialized byte slice into a mutable `&mut [BlockQ8_0]` view.
#[cfg(feature = "with-kernels")]
fn bytes_to_q8_0_blocks_mut(bytes: &mut [u8], blocks: usize) -> Result<&mut [BlockQ8_0], K3Error> {
    let needed =
        blocks
            .checked_mul(size_of::<BlockQ8_0>())
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "q8_0_scratch".to_owned(),
                offset: 0,
                byte_len: usize::MAX,
                buffer_len: bytes.len(),
                reason: "q8_0 scratch size overflow".to_owned(),
            })?;
    if bytes.len() < needed {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "q8_0_scratch".to_owned(),
            offset: 0,
            byte_len: needed,
            buffer_len: bytes.len(),
            reason: format!(
                "q8_0 scratch bytes {} shorter than required {needed}",
                bytes.len()
            ),
        });
    }
    if blocks == 0 {
        return Ok(&mut []);
    }
    let slice = &mut bytes[..needed];
    if slice.as_ptr().align_offset(align_of::<BlockQ8_0>()) != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "q8_0_scratch".to_owned(),
            offset: 0,
            byte_len: needed,
            buffer_len: slice.len(),
            reason: "q8_0 scratch pointer is not 4-byte aligned".to_owned(),
        });
    }
    slice.fill(0);
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "verified align_offset(align_of::<BlockQ8_0>()) == 0 above"
    )]
    let ptr = slice.as_mut_ptr().cast::<BlockQ8_0>();
    // SAFETY: `slice` is exclusively borrowed, `BlockQ8_0`-aligned (`align_offset == 0` verified above),
    // zero-initialized (`d: 0.0`, `qs: [0; 32]` is valid with zero padding in `#[repr(C)] BlockQ8_0`),
    // and spans `blocks * size_of::<BlockQ8_0>()` bytes; `slice` is not accessed again.
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, blocks) })
}

/// Quantizes `input` into caller-supplied `Q8_0` scratch so sibling projections share
/// quantized activations without heap allocation.
#[cfg_attr(
    not(feature = "with-kernels"),
    expect(
        clippy::unnecessary_wraps,
        reason = "with-kernels feature branch quantizes Q8_0 fallibly into caller scratch"
    )
)]
pub(crate) fn prepare_m1_activation<'a>(
    input: &'a [f32],
    need_q8_0: bool,
    q8_0_scratch: &'a mut [u8],
) -> Result<M1Activation<'a>, K3Error> {
    #[cfg(feature = "with-kernels")]
    {
        let q8_0 = if need_q8_0 && !input.is_empty() && input.len() % Q8_0_BLOCK_ELEMENTS == 0 {
            let blocks = input.len() / Q8_0_BLOCK_ELEMENTS;
            let dst = bytes_to_q8_0_blocks_mut(q8_0_scratch, blocks)?;
            quantize_row_q8_0(input, dst)?;
            &*dst
        } else {
            &[]
        };
        Ok(M1Activation { input, q8_0 })
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        let _ = (need_q8_0, q8_0_scratch);
        Ok(M1Activation { input })
    }
}

/// Resolves the active worker partition count, honoring explicit recipe overrides or Rayon's pool.
#[inline]
pub(crate) fn effective_worker_count(worker_partitions: Option<usize>) -> usize {
    #[cfg(not(miri))]
    {
        worker_partitions
            .unwrap_or_else(rayon::current_num_threads)
            .max(1)
    }
    #[cfg(miri)]
    {
        worker_partitions.unwrap_or(1).max(1)
    }
}

/// Computes an `8`-row-aligned chunk height so every worker chunk starts on a 4-row / 8-row boundary,
/// preserving bit-identical GEMV outputs across `1`, `4`, and `8` worker partitions.
#[inline]
pub(crate) fn aligned_partition_rows(rows: usize, workers: usize) -> usize {
    let eff = workers.max(1);
    if eff <= 1 || rows <= PARTITION_ROW_ALIGN {
        return rows.max(1);
    }
    rows.div_ceil(eff)
        .next_multiple_of(PARTITION_ROW_ALIGN)
        .max(PARTITION_ROW_ALIGN)
}

fn raw_row_byte_len(tensor_name: &str, quant: QuantType, cols: usize) -> Result<usize, K3Error> {
    let Some((blk_elems, blk_bytes)) = quant_block_geometry(quant) else {
        return Err(K3Error::unsupported_tensor_quant(
            "dory",
            tensor_name,
            quant,
        ));
    };
    if cols == 0 || cols % blk_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: cols,
            buffer_len: blk_elems,
            reason: format!(
                "column width {cols} is not a positive multiple of {quant:?} block size {blk_elems}"
            ),
        });
    }
    (cols / blk_elems)
        .checked_mul(blk_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: cols,
            reason: "quantized row byte stride overflow".to_owned(),
        })
}

/// Projects `output[..out_rows] = matrix[..out_rows, ..] * act` across `workers` partitions with
/// zero per-call heap allocations.
pub(crate) fn project_raw_rows_partitioned(
    output: &mut [f32],
    matrix: RawWeightMatrix<'_>,
    act: &M1Activation<'_>,
    workers: usize,
) -> Result<(), K3Error> {
    let out_rows = output.len();
    if out_rows == 0 || matrix.cols == 0 || out_rows > matrix.total_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: matrix.name.to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: matrix.raw.len(),
            reason: format!(
                "invalid row-partitioned GEMV shape out_rows={out_rows}, total_rows={}, cols={}",
                matrix.total_rows, matrix.cols
            ),
        });
    }
    if act.input.len() != matrix.cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: matrix.name.to_owned(),
            offset: 0,
            byte_len: size_of_val(act.input),
            buffer_len: matrix.cols,
            reason: format!(
                "input length {} does not match matrix cols {}",
                act.input.len(),
                matrix.cols
            ),
        });
    }

    let row_bytes = raw_row_byte_len(matrix.name, matrix.quant, matrix.cols)?;
    let used_bytes = out_rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: matrix.name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: matrix.raw.len(),
            reason: "partitioned output byte range overflow".to_owned(),
        })?;
    if matrix.raw.len() < used_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: matrix.name.to_owned(),
            offset: 0,
            byte_len: used_bytes,
            buffer_len: matrix.raw.len(),
            reason: format!(
                "matrix payload len {} shorter than required {used_bytes}",
                matrix.raw.len()
            ),
        });
    }

    let chunk_rows = aligned_partition_rows(out_rows, workers);
    if chunk_rows >= out_rows {
        return project_raw_row_slice(
            output,
            matrix.name,
            matrix.quant,
            matrix.cols,
            &matrix.raw[..used_bytes],
            act,
        );
    }

    #[cfg(not(miri))]
    if rayon::current_num_threads() > 1 {
        return output.par_chunks_mut(chunk_rows).enumerate().try_for_each(
            |(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * chunk_rows;
                let byte_start = start_row * row_bytes;
                let byte_end = byte_start + out_chunk.len() * row_bytes;
                project_raw_row_slice(
                    out_chunk,
                    matrix.name,
                    matrix.quant,
                    matrix.cols,
                    &matrix.raw[byte_start..byte_end],
                    act,
                )
            },
        );
    }

    for (chunk_idx, out_chunk) in output.chunks_mut(chunk_rows).enumerate() {
        let start_row = chunk_idx * chunk_rows;
        let byte_start = start_row * row_bytes;
        let byte_end = byte_start + out_chunk.len() * row_bytes;
        project_raw_row_slice(
            out_chunk,
            matrix.name,
            matrix.quant,
            matrix.cols,
            &matrix.raw[byte_start..byte_end],
            act,
        )?;
    }
    Ok(())
}

/// Partitioned single-token `M = 1` RVV/IME GEMV over pre-repacked weight tiles and an
/// arena-backed [`QuantizedActQ8View`] with zero heap allocations and zero `TILE_M = 8` waste.
#[cfg(feature = "with-kernels")]
pub(crate) fn project_repacked_rows_partitioned(
    output: &mut [f32],
    weight: &RepackedWeight,
    act: &QuantizedActQ8View<'_>,
    workers: usize,
) -> Result<(), K3Error> {
    let out_rows = output.len();
    if out_rows == 0 || out_rows > weight.rows() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "repacked_weight".to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: weight.rows(),
            reason: format!(
                "invalid row-partitioned GEMV shape out_rows={out_rows}, total_rows={}",
                weight.rows()
            ),
        });
    }
    let chunk_rows = aligned_partition_rows(out_rows, workers);
    if chunk_rows >= out_rows {
        return weight.gemv_view_rows(output, act, 0..out_rows);
    }

    #[cfg(not(miri))]
    if rayon::current_num_threads() > 1 {
        return output.par_chunks_mut(chunk_rows).enumerate().try_for_each(
            |(chunk_idx, out_chunk)| {
                let start_row = chunk_idx * chunk_rows;
                weight.gemv_view_rows(out_chunk, act, start_row..start_row + out_chunk.len())
            },
        );
    }

    for (chunk_idx, out_chunk) in output.chunks_mut(chunk_rows).enumerate() {
        let start_row = chunk_idx * chunk_rows;
        weight.gemv_view_rows(out_chunk, act, start_row..start_row + out_chunk.len())?;
    }
    Ok(())
}

/// Projects `output[..rows] = matrix * input` through partitioned scalar kernels
/// (`--no-default-features` path).
#[cfg(not(feature = "with-kernels"))]
pub(crate) fn project_matrix_rows_stack(
    output: &mut [f32],
    matrix: RawWeightMatrix<'_>,
    input: &[f32],
    workers: usize,
) -> Result<(), K3Error> {
    let act = M1Activation::f32_only(input);
    project_raw_rows_partitioned(output, matrix, &act, workers)
}

fn project_raw_row_slice(
    output: &mut [f32],
    tensor_name: &str,
    quant: QuantType,
    cols: usize,
    chunk_raw: &[u8],
    act: &M1Activation<'_>,
) -> Result<(), K3Error> {
    if quant == QuantType::F32 {
        return project_f32_raw_rows(
            output,
            tensor_name,
            chunk_raw,
            output.len(),
            cols,
            act.input,
        );
    }

    let dims = [cols, output.len()];
    let sub_view = TensorView::from_bytes(tensor_name, quant, &dims, 0, chunk_raw)?;

    #[cfg(feature = "with-kernels")]
    {
        match quant {
            QuantType::Q4_0 if !act.q8_0.is_empty() => gemv_q4_0_q8_0(output, &sub_view, act.q8_0),
            QuantType::Q5_0 if !act.q8_0.is_empty() => gemv_q5_0_q8_0(output, &sub_view, act.q8_0),
            QuantType::Q4_0 => vec_dot_q4_0(output, &sub_view, act.input),
            QuantType::Q8_0 => BaselineKernel::new().vec_dot_q8_0(output, &sub_view, act.input),
            QuantType::Q4_K_M => BaselineKernel::new().vec_dot_q4_k_m(output, &sub_view, act.input),
            QuantType::Q5_K => BaselineKernel::new().vec_dot_q5_k(output, &sub_view, act.input),
            QuantType::Q6_K => BaselineKernel::new().vec_dot_q6_k(output, &sub_view, act.input),
            QuantType::Q5_0 => dot_block_rows::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
                &sub_view,
                output.len(),
                cols,
                act.input,
                output,
                dequantize_q5_0_block,
            ),
            QuantType::F16 => vec_dot_f16(output, &sub_view, act.input),
            QuantType::BF16 => vec_dot_bf16(output, &sub_view, act.input),
            _ => Err(K3Error::unsupported_tensor_quant(
                "dory",
                tensor_name,
                quant,
            )),
        }
    }
    #[cfg(not(feature = "with-kernels"))]
    {
        project_matvec_scalar(output, &sub_view, output.len(), cols, act.input)
    }
}

fn project_f32_raw_rows(
    output: &mut [f32],
    tensor_name: &str,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    let stride = cols
        .checked_mul(4)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: raw.len(),
            reason: "f32 row stride overflow".to_owned(),
        })?;
    let required = rows
        .checked_mul(stride)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: raw.len(),
            reason: "f32 matrix byte length overflow".to_owned(),
        })?;
    if raw.len() < required || input.len() != cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: required,
            buffer_len: raw.len(),
            reason: "f32 matrix slice shorter than expected row payload".to_owned(),
        });
    }
    let mut chunk_buf = [0.0_f32; 256];
    for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_bytes = &raw[r * stride..(r + 1) * stride];
        let mut sum = 0.0_f32;
        let mut col = 0usize;
        for chunk in row_bytes.chunks(chunk_buf.len() * 4) {
            let n = chunk.len() / 4;
            for (dst, b) in chunk_buf[..n].iter_mut().zip(chunk.chunks_exact(4)) {
                *dst = sanitize_f32(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            for (&w, &x) in chunk_buf[..n].iter().zip(&input[col..col + n]) {
                sum += w * x;
            }
            col += n;
        }
        *out_slot = sum;
    }
    Ok(())
}

#[cfg(not(feature = "with-kernels"))]
fn project_half_raw_rows(
    output: &mut [f32],
    tensor_name: &str,
    quant: QuantType,
    raw: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<(), K3Error> {
    let stride = cols
        .checked_mul(2)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: raw.len(),
            reason: "half row stride overflow".to_owned(),
        })?;
    let required = rows
        .checked_mul(stride)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: usize::MAX,
            buffer_len: raw.len(),
            reason: "half matrix byte length overflow".to_owned(),
        })?;
    if raw.len() < required || input.len() != cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: required,
            buffer_len: raw.len(),
            reason: "half matrix slice shorter than expected row payload".to_owned(),
        });
    }
    let mut chunk_buf = [0.0_f32; 256];
    for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_bytes = &raw[r * stride..(r + 1) * stride];
        let mut sum = 0.0_f32;
        let mut col = 0usize;
        for chunk in row_bytes.chunks(chunk_buf.len() * 2) {
            let n = chunk.len() / 2;
            match quant {
                QuantType::F16 => {
                    for (dst, b) in chunk_buf[..n].iter_mut().zip(chunk.chunks_exact(2)) {
                        *dst = f16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                    }
                }
                QuantType::BF16 => {
                    for (dst, b) in chunk_buf[..n].iter_mut().zip(chunk.chunks_exact(2)) {
                        *dst = bf16_to_f32_finite(u16::from_le_bytes([b[0], b[1]]));
                    }
                }
                _ => {
                    return Err(K3Error::unsupported_tensor_quant(
                        "dory",
                        tensor_name,
                        quant,
                    ));
                }
            }
            for (&w, &x) in chunk_buf[..n].iter().zip(&input[col..col + n]) {
                sum += w * x;
            }
            col += n;
        }
        *out_slot = sum;
    }
    Ok(())
}
