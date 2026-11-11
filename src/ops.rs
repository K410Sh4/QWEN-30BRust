use anyhow::{ensure, Result};

pub fn rms(x: &mut [f32], weight: &[f32], eps: f32) -> Result<()> {
    ensure!(
        !x.is_empty() && x.len() == weight.len() && eps > 0.,
        "RMSNorm shape/epsilon invalid"
    );
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    for (v, &w) in x.iter_mut().zip(weight) {
        *v *= scale * w;
    }
    Ok(())
}
/// Qwen GGUF uses NeoX (split-half) rotary embeddings, not adjacent pairs.
pub fn rope(x: &mut [f32], head: usize, pos: usize, theta: f32) {
    for h in x.chunks_exact_mut(head) {
        for i in 0..head / 2 {
            let angle = pos as f32 * theta.powf(-2. * i as f32 / head as f32);
            let (s, c) = angle.sin_cos();
            let a = h[i];
            let b = h[i + head / 2];
            h[i] = a * c - b * s;
            h[i + head / 2] = a * s + b * c;
        }
    }
}
pub fn softmax(x: &mut [f32]) -> Result<()> {
    ensure!(
        !x.is_empty() && x.iter().all(|v| v.is_finite()),
        "invalid softmax logits"
    );
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x {
        *v /= sum;
    }
    Ok(())
}
pub fn route(logits: &[f32], top: usize) -> Result<Vec<(usize, f32)>> {
    ensure!(
        top > 0 && top <= logits.len() && logits.iter().all(|v| v.is_finite()),
        "invalid MoE router"
    );
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(top);
    let mut weights: Vec<_> = ids.iter().map(|&i| logits[i]).collect();
    softmax(&mut weights)?;
    Ok(ids.into_iter().zip(weights).collect())
}
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    kvheads: usize,
    d: usize,
) -> Result<Vec<f32>> {
    ensure!(
        kvheads > 0
            && heads > 0
            && d > 0
            && q.len() == heads * d
            && heads.is_multiple_of(kvheads)
            && k.len() == v.len()
            && k.len().is_multiple_of(kvheads * d),
        "attention shape mismatch"
    );
    let time = k.len() / (kvheads * d);
    ensure!(time > 0, "empty KV cache");
    let mut out = vec![0.; q.len()];
    for h in 0..heads {
        let kh = h / (heads / kvheads);
        let qs = &q[h * d..(h + 1) * d];
        let mut scores: Vec<f32> = (0..time)
            .map(|t| {
                qs.iter()
                    .zip(&k[(t * kvheads + kh) * d..(t * kvheads + kh + 1) * d])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (d as f32).sqrt()
            })
            .collect();
        softmax(&mut scores)?;
        for (t, &score) in scores.iter().enumerate() {
            for j in 0..d {
                out[h * d + j] += score * v[(t * kvheads + kh) * d + j];
            }
        }
    }
    Ok(out)
}
