//! Quantized block dequantization, geometry validation, and projection kernels for Dory (`dory`).
//!
//! Dispatches to `jero-kernels-rvv` hardware/scalar kernels when `with-kernels` is enabled, and
//! executes bit-equivalent portable scalar math across `F32`, `F16`, `BF16`, `Q4_0`, `Q5_0`,
//! `Q8_0`, `Q4_K_M`, `Q5_K`, and `Q6_K` when built with `--no-default-features`.

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(feature = "with-kernels")]
pub(crate) use jero_kernels_rvv::bf16_to_f32_finite;
#[cfg(feature = "with-kernels")]
use jero_kernels_rvv::{
    BaselineKernel, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
    Q4_K_M_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS, VectorKernel, dequantize_token_row,
    f16_to_f32_finite, vec_dot_bf16, vec_dot_f16, vec_dot_q4_0,
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

/// Computes `output[0..rows] = weights[0..rows, 0..cols] * input[0..cols]` for either exact
/// full-matrix projection (`output.len() == rows`) or a leading vocabulary prefix
/// (`0 < output.len() <= rows`).
///
/// # Errors
/// Returns [`K3Error::TensorBoundsError`] on shape/buffer mismatch or
/// [`K3Error::CapabilityMismatch`] on unsupported quantization formats.
pub(crate) fn project_matvec_prefix(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let expected_cols = input.len();
    let (total_rows, cols) = resolve_2d_dims(weights, expected_cols)?;
    let out_rows = output.len();
    if out_rows == 0 || out_rows > total_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "output len {out_rows} out of bounds for matrix rows {total_rows} (shape {:?})",
                weights.shape()
            ),
        });
    }

    if out_rows == total_rows {
        return project_matvec_full(output, weights, total_rows, cols, input);
    }

    // Prefix projection: dequantize each requested leading row without requiring full-matrix output.
    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in output.iter_mut().enumerate() {
        dequantize_row_to_f32(weights, r, cols, &mut row_buf)?;
        *out_slot = dot_product_f32(&row_buf, input);
    }
    Ok(())
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
        QuantType::F16 | QuantType::BF16 => {
            let mut row_buf = vec![0.0_f32; cols];
            let stride = cols * 2;
            let raw = weights.as_bytes();
            for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
                decode_unquant_payload(weights, &raw[r * stride..(r + 1) * stride], &mut row_buf)?;
                *out_slot = dot_product_f32(&row_buf, input);
            }
            Ok(())
        }
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
    let raw = weights.as_bytes();
    let stride = cols * 4;
    let mut row_buf = vec![0.0_f32; cols];
    for (r, out_slot) in output.iter_mut().enumerate().take(rows) {
        decode_unquant_payload(weights, &raw[r * stride..(r + 1) * stride], &mut row_buf)?;
        *out_slot = dot_product_f32(&row_buf, input);
    }
    Ok(())
}
