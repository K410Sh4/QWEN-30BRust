use anyhow::{ensure, Result};
use rand::{rngs::StdRng, Rng, SeedableRng};
pub struct Sampler {
    rng: StdRng,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}
impl Sampler {
    pub fn new(seed: u64, temperature: f32, top_k: usize, top_p: f32) -> Result<Self> {
        ensure!(
            temperature.is_finite()
                && temperature >= 0.
                && top_p.is_finite()
                && top_p > 0.
                && top_p <= 1.,
            "invalid sampling parameters"
        );
        Ok(Self {
            rng: StdRng::seed_from_u64(seed),
            temperature,
            top_k,
            top_p,
        })
    }
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure!(
            !logits.is_empty() && logits.iter().all(|v| v.is_finite()),
            "invalid logits"
        );
        let order = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
        if self.temperature == 0. {
            let id = (0..logits.len()).min_by(order).unwrap();
            return Ok(id as u32);
        }
        let mut ids: Vec<_> = (0..logits.len()).collect();
        if self.top_k > 0 && self.top_k < ids.len() {
            ids.select_nth_unstable_by(self.top_k, order);
            ids.truncate(self.top_k);
        }
        ids.sort_unstable_by(order);
        let max = logits[ids[0]];
        let mut probs: Vec<f32> = ids
            .iter()
            .map(|&i| ((logits[i] - max) / self.temperature).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        let mut cumulative = 0.;
        let mut n = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p / total;
            if cumulative >= self.top_p {
                n = i + 1;
                break;
            }
        }
        probs.truncate(n);
        ids.truncate(n);
        let mut draw = self.rng.gen::<f32>() * probs.iter().sum::<f32>();
        for (&id, &p) in ids.iter().zip(&probs) {
            if draw < p {
                return Ok(id as u32);
            }
            draw -= p;
        }
        Ok(*ids.last().unwrap() as u32)
    }
}
