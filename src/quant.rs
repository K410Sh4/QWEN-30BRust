//! GGML block decoding. Byte layouts follow the GGUF/GGML specification.
use anyhow::{bail, ensure, Result};
use half::f16;

pub fn layout(kind: u32) -> Result<(usize, usize)> {
    Ok(match kind {
        0 => (1, 4), 1 => (1, 2), 2 => (32, 18), 8 => (32, 34),
        12 => (256, 144), 13 => (256, 176), 14 => (256, 210),
        _ => bail!("GGML tensor type {kind} is not supported (supported: F32/F16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)"),
    })
}
pub fn row_bytes(kind: u32, cols: usize) -> Result<usize> {
    let (block, bytes) = layout(kind)?;
    ensure!(
        cols > 0 && cols.is_multiple_of(block),
        "row width {cols} is not divisible by block size {block}"
    );
    (cols / block)
        .checked_mul(bytes)
        .ok_or_else(|| anyhow::anyhow!("row size overflow"))
}
fn h(b: &[u8], p: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[p], b[p + 1]])).to_f32()
}
fn scale(b: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (b[j] & 63, b[j + 4] & 63)
    } else {
        (
            (b[j + 4] & 15) | ((b[j - 4] >> 6) << 4),
            (b[j + 4] >> 4) | ((b[j] >> 6) << 4),
        )
    }
}
/// Caller supplies a validated row and an in-range column.
pub fn value(b: &[u8], kind: u32, col: usize) -> f32 {
    match kind {
        0 => f32::from_le_bytes(b[col * 4..col * 4 + 4].try_into().unwrap()),
        1 => h(b, col * 2),
        2 => {
            let p = col / 32 * 18;
            let i = col % 32;
            let q = (b[p + 2 + i % 16] >> (if i < 16 { 0 } else { 4 })) & 15;
            h(b, p) * (q as i32 - 8) as f32
        }
        8 => {
            let p = col / 32 * 34;
            h(b, p) * (b[p + 2 + col % 32] as i8) as f32
        }
        12 | 13 => {
            let size = if kind == 12 { 144 } else { 176 };
            let p = col / 256 * size;
            let i = col % 256;
            let group = i / 32;
            let (s, m) = scale(&b[p + 4..p + 16], group);
            let qs = p + if kind == 12 { 16 } else { 48 };
            let mut q = (b[qs + (i / 64) * 32 + i % 32] >> ((group % 2) * 4)) & 15;
            if kind == 13 && b[p + 16 + i % 32] & (1 << group) != 0 {
                q += 16;
            }
            h(b, p) * s as f32 * q as f32 - h(b, p + 2) * m as f32
        }
        14 => {
            let p = col / 256 * 210;
            let i = col % 256;
            let chunk = i / 128;
            let j = i % 128;
            let quarter = j / 32;
            let lane = j % 32;
            let low = b[p + chunk * 64 + (quarter % 2) * 32 + lane];
            let lo = (low >> ((quarter / 2) * 4)) & 15;
            let hi = (b[p + 128 + chunk * 32 + lane] >> (quarter * 2)) & 3;
            let q = (lo | (hi << 4)) as i32 - 32;
            h(b, p + 208)
                * (b[p + 192 + chunk * 8 + quarter * 2 + lane / 16] as i8) as f32
                * q as f32
        }
        _ => unreachable!("tensor types validated by reader"),
    }
}
pub fn decode(b: &[u8], kind: u32, cols: usize) -> Result<Vec<f32>> {
    ensure!(b.len() >= row_bytes(kind, cols)?, "truncated quantized row");
    Ok((0..cols).map(|i| value(b, kind, i)).collect())
}
