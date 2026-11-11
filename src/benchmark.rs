use crate::{
    backend::{Backend, Stats},
    gguf::Gguf,
    model::Model,
};
use anyhow::{ensure, Result};
use serde::Serialize;
use std::{path::Path, time::Instant};

#[derive(Serialize)]
pub struct Report {
    pub version: &'static str,
    pub device_resident: bool,
    pub backend: serde_json::Value,
    pub context: usize,
    pub input_tokens: Vec<u32>,
    pub weight_cache_note: &'static str,
    pub cold_seconds: f64,
    pub cold_stats: Stats,
    pub warmup_runs: usize,
    pub measured_seconds: Vec<f64>,
    pub measured_stats: Vec<Stats>,
    pub median_seconds: f64,
    pub forward_tokens_per_second: f64,
}
pub fn delta(a: &Stats, b: &Stats) -> Stats {
    Stats {
        matvecs: a.matvecs - b.matvecs,
        uploaded_bytes: a.uploaded_bytes - b.uploaded_bytes,
        cache_hits: a.cache_hits - b.cache_hits,
        matvec_ms: a.matvec_ms - b.matvec_ms,
        gpu_dispatches: a.gpu_dispatches - b.gpu_dispatches,
        submissions: a.submissions - b.submissions,
        readbacks: a.readbacks - b.readbacks,
        readback_bytes: a.readback_bytes - b.readback_bytes,
        cache_misses: a.cache_misses - b.cache_misses,
        cache_evictions: a.cache_evictions - b.cache_evictions,
        workspace_allocations: a.workspace_allocations - b.workspace_allocations,
        device_forward_ms: a.device_forward_ms - b.device_forward_ms,
        weight_requested_bytes: a.weight_requested_bytes - b.weight_requested_bytes,
        weight_buffer_allocations: a.weight_buffer_allocations - b.weight_buffer_allocations,
        weight_buffer_reuses: a.weight_buffer_reuses - b.weight_buffer_reuses,
        weight_prepare_ms: a.weight_prepare_ms - b.weight_prepare_ms,
        gpu_readback_wait_ms: a.gpu_readback_wait_ms - b.gpu_readback_wait_ms,
    }
}
pub fn run<F>(
    path: &Path,
    tokens: &[u32],
    ctx: usize,
    warmup: usize,
    runs: usize,
    create: F,
) -> Result<(Report, Vec<Vec<f32>>)>
where
    F: FnOnce() -> Result<Box<dyn Backend>>,
{
    ensure!(
        !tokens.is_empty() && tokens.len() <= ctx && runs > 0 && runs <= 100 && warmup <= 100,
        "invalid benchmark parameters"
    );
    let mut m = Model::new(Gguf::open(path)?, ctx)?;
    let mut backend = create()?;
    let resident = backend.device_resident();
    let backend_metadata = backend.metadata();
    let mut times = Vec::new();
    let mut stats = Vec::new();
    let mut cold_seconds = 0.;
    let mut cold_stats = Stats::default();
    let mut outputs = Vec::new();
    for cycle in 0..1 + warmup + runs {
        m.reset();
        let before = backend.stats().clone();
        let start = Instant::now();
        for &token in tokens {
            let logits = m.forward(token, &mut *backend)?;
            if cycle == 0 {
                outputs.push(logits);
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let changes = delta(backend.stats(), &before);
        eprintln!(
            "Benchmark cycle {}/{}: {:.6}s | readbacks={} dispatches={} uploaded_bytes={}",
            cycle + 1,
            1 + warmup + runs,
            elapsed,
            changes.readbacks,
            changes.gpu_dispatches,
            changes.uploaded_bytes
        );
        if cycle == 0 {
            cold_seconds = elapsed;
            cold_stats = changes;
        } else if cycle > warmup {
            times.push(elapsed);
            stats.push(changes);
        }
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.
    } else {
        sorted[sorted.len() / 2]
    };
    let report = Report {
        version: env!("CARGO_PKG_VERSION"),
        device_resident: resident,
        backend: backend_metadata,
        context: ctx,
        input_tokens: tokens.to_vec(),
        weight_cache_note:
            "KV reset per cycle; weight cache retained per mode. Cold includes graph setup.",
        cold_seconds,
        cold_stats,
        warmup_runs: warmup,
        measured_seconds: times,
        measured_stats: stats,
        median_seconds: median,
        forward_tokens_per_second: tokens.len() as f64 / median,
    };
    Ok((report, outputs))
}
#[derive(Serialize)]
pub struct Parity {
    pub passed: bool,
    pub max_absolute_error: f32,
    pub mismatches: usize,
    pub greedy_agreement: bool,
    pub atol: f32,
    pub rtol: f32,
}
pub fn parity(reference: &[Vec<f32>], actual: &[Vec<f32>], atol: f32, rtol: f32) -> Result<Parity> {
    ensure!(
        reference.len() == actual.len() && !reference.is_empty(),
        "parity prefix count mismatch"
    );
    let mut max = 0f32;
    let mut mismatches = 0;
    let mut greedy = true;
    for (a, b) in reference.iter().zip(actual) {
        ensure!(
            a.len() == b.len() && !a.is_empty(),
            "parity vocabulary mismatch"
        );
        let argmax = |x: &[f32]| {
            (0..x.len())
                .min_by(|&i, &j| x[j].total_cmp(&x[i]).then(i.cmp(&j)))
                .unwrap()
        };
        greedy &= argmax(a) == argmax(b);
        for (&x, &y) in a.iter().zip(b) {
            let diff = (x - y).abs();
            max = max.max(diff);
            if !x.is_finite() || !y.is_finite() || diff > atol + rtol * x.abs() {
                mismatches += 1;
            }
        }
    }
    Ok(Parity {
        passed: mismatches == 0,
        max_absolute_error: max,
        mismatches,
        greedy_agreement: greedy,
        atol,
        rtol,
    })
}
