use crate::{backend::Backend, gguf::Gguf, ops};
use anyhow::{ensure, Context, Result};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kvheads: usize,
    pub head_dim: usize,
    pub experts: usize,
    pub top: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_context: usize,
    pub eps: f32,
    pub theta: f32,
}
impl Config {
    pub fn from_gguf(m: &Gguf) -> Result<Self> {
        ensure!(
            m.metadata
                .get("general.architecture")
                .and_then(|v| v.as_str())
                == Some("qwen3moe"),
            "only qwen3moe architecture is implemented"
        );
        let get = |key: &str| m.uint(&format!("qwen3moe.{key}"));
        let hidden = get("embedding_length")?;
        let heads = get("attention.head_count")?;
        let top = get("expert_used_count")?;
        let head_dim = get("attention.key_length").unwrap_or(hidden / heads.max(1));
        ensure!(
            get("attention.value_length").unwrap_or(head_dim) == head_dim,
            "different key/value head sizes unsupported"
        );
        let c = Self {
            hidden,
            layers: get("block_count")?,
            heads,
            kvheads: get("attention.head_count_kv")?,
            head_dim,
            experts: get("expert_count")?,
            top,
            intermediate: get("expert_feed_forward_length")
                .or_else(|_| get("feed_forward_length").map(|v| v / top.max(1)))?,
            vocab: m
                .tensor("token_embd.weight")?
                .dims
                .get(1)
                .copied()
                .context("invalid embedding rank")?,
            max_context: get("context_length")?,
            eps: m.float("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6),
            theta: m.float("qwen3moe.rope.freq_base", 1_000_000.),
        };
        ensure!(
            c.hidden > 0
                && c.layers > 0
                && c.layers <= 256
                && c.kvheads > 0
                && c.heads > 0
                && c.heads.is_multiple_of(c.kvheads)
                && c.head_dim > 0
                && c.head_dim.is_multiple_of(2)
                && c.top > 0
                && c.top <= c.experts
                && c.experts <= 1024
                && c.intermediate > 0
                && c.eps.is_finite()
                && c.eps > 0.
                && c.theta.is_finite()
                && c.theta > 0.,
            "invalid Qwen3 MoE config"
        );
        ensure!(
            get("rope.dimension_count").unwrap_or(c.head_dim) == c.head_dim,
            "partial rotary dimensions unsupported"
        );
        ensure!(
            m.metadata
                .get("qwen3moe.rope.scaling.type")
                .and_then(|v| v.as_str())
                .map(|v| v == "none")
                .unwrap_or(true),
            "RoPE scaling is not implemented; use the original native-context model"
        );
        Ok(c)
    }
    pub fn validate_tensors(&self, m: &Gguf) -> Result<()> {
        let check = |name: &str, shape: &[usize]| -> Result<()> {
            let t = m.tensor(name)?;
            ensure!(
                t.dims == shape,
                "{name}: expected {shape:?}, found {:?}",
                t.dims
            );
            Ok(())
        };
        check("token_embd.weight", &[self.hidden, self.vocab])?;
        check("output_norm.weight", &[self.hidden])?;
        if m.tensors.contains_key("output.weight") {
            check("output.weight", &[self.hidden, self.vocab])?;
        }
        for i in 0..self.layers {
            for name in ["attn_norm", "ffn_norm"] {
                check(&format!("blk.{i}.{name}.weight"), &[self.hidden])?;
            }
            for name in ["attn_q_norm", "attn_k_norm"] {
                check(&format!("blk.{i}.{name}.weight"), &[self.head_dim])?;
            }
            check(
                &format!("blk.{i}.attn_q.weight"),
                &[self.hidden, self.heads * self.head_dim],
            )?;
            for name in ["attn_k", "attn_v"] {
                check(
                    &format!("blk.{i}.{name}.weight"),
                    &[self.hidden, self.kvheads * self.head_dim],
                )?;
            }
            check(
                &format!("blk.{i}.attn_output.weight"),
                &[self.heads * self.head_dim, self.hidden],
            )?;
            check(
                &format!("blk.{i}.ffn_gate_inp.weight"),
                &[self.hidden, self.experts],
            )?;
            for name in ["ffn_gate_exps", "ffn_up_exps"] {
                check(
                    &format!("blk.{i}.{name}.weight"),
                    &[self.hidden, self.intermediate, self.experts],
                )?;
            }
            check(
                &format!("blk.{i}.ffn_down_exps.weight"),
                &[self.intermediate, self.hidden, self.experts],
            )?;
            ensure!(
                !m.tensors
                    .keys()
                    .any(|n| n.starts_with(&format!("blk.{i}.")) && n.ends_with(".bias")),
                "unexpected bias tensors"
            );
        }
        Ok(())
    }
    pub fn kv_bytes(&self, ctx: usize) -> Result<usize> {
        ctx.checked_mul(self.layers)
            .and_then(|n| n.checked_mul(self.kvheads))
            .and_then(|n| n.checked_mul(self.head_dim))
            .and_then(|n| n.checked_mul(8))
            .context("KV size overflow")
    }
}
#[derive(Default)]
struct Kv {
    k: Vec<f32>,
    v: Vec<f32>,
}
pub struct Model {
    pub gguf: Gguf,
    pub config: Config,
    cache: Vec<Kv>,
    norms: Vec<Vec<Vec<f32>>>,
    output_norm: Vec<f32>,
    pub position: usize,
    context: usize,
}
impl Model {
    pub fn new(gguf: Gguf, context: usize) -> Result<Self> {
        let config = Config::from_gguf(&gguf)?;
        config.validate_tensors(&gguf)?;
        ensure!(
            context > 0 && context <= config.max_context,
            "context must be in 1..={}",
            config.max_context
        );
        // Avoid accidental multi-gigabyte KV reservations; allocated incrementally.
        let mut norms = Vec::new();
        for i in 0..config.layers {
            let mut n = Vec::new();
            for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"] {
                n.push(gguf.row(&format!("blk.{i}.{name}.weight"), 0)?);
            }
            norms.push(n);
        }
        let output_norm = gguf.row("output_norm.weight", 0)?;
        let cache = (0..config.layers).map(|_| Kv::default()).collect();
        Ok(Self {
            gguf,
            config,
            cache,
            norms,
            output_norm,
            position: 0,
            context,
        })
    }
    pub fn reset(&mut self) {
        for c in &mut self.cache {
            c.k.clear();
            c.v.clear();
        }
        self.position = 0;
    }
    /// Single-token causal forward. On error the cache is rolled back to a clean state.
    pub fn forward(&mut self, token: u32, backend: &mut dyn Backend) -> Result<Vec<f32>> {
        self.forward_mode(token, backend, true)
    }
    /// Advance the causal state without computing an unused vocabulary projection.
    pub fn forward_without_logits(&mut self, token: u32, backend: &mut dyn Backend) -> Result<()> {
        self.forward_mode(token, backend, false).map(|_| ())
    }
    fn forward_mode(
        &mut self,
        token: u32,
        backend: &mut dyn Backend,
        logits: bool,
    ) -> Result<Vec<f32>> {
        let result = self.forward_inner(token, backend, logits);
        if result.is_err() {
            self.reset();
        }
        result
    }
    fn forward_inner(
        &mut self,
        token: u32,
        b: &mut dyn Backend,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        ensure!(
            self.position < self.context,
            "context limit {} reached",
            self.context
        );
        ensure!((token as usize) < c.vocab, "token outside vocabulary");
        if let Some(logits) = b.forward_device(
            &self.gguf,
            c,
            token,
            self.position,
            self.context,
            want_logits,
        )? {
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "nonfinite GPU model output"
            );
            self.position += 1;
            return Ok(logits);
        }
        let mut x = self.gguf.row("token_embd.weight", token as usize)?;
        for i in 0..c.layers {
            let base = format!("blk.{i}");
            let mut xn = x.clone();
            ops::rms(&mut xn, &self.norms[i][0], c.eps)?;
            let mut q = b.matvec(&self.gguf, &format!("{base}.attn_q.weight"), None, &xn)?;
            let mut k = b.matvec(&self.gguf, &format!("{base}.attn_k.weight"), None, &xn)?;
            let v = b.matvec(&self.gguf, &format!("{base}.attn_v.weight"), None, &xn)?;
            for h in q.chunks_exact_mut(c.head_dim) {
                ops::rms(h, &self.norms[i][1], c.eps)?;
            }
            for h in k.chunks_exact_mut(c.head_dim) {
                ops::rms(h, &self.norms[i][2], c.eps)?;
            }
            ops::rope(&mut q, c.head_dim, self.position, c.theta);
            ops::rope(&mut k, c.head_dim, self.position, c.theta);
            let kv = &mut self.cache[i];
            kv.k.extend_from_slice(&k);
            kv.v.extend_from_slice(&v);
            let a = ops::attention(&q, &kv.k, &kv.v, c.heads, c.kvheads, c.head_dim)?;
            let a = b.matvec(&self.gguf, &format!("{base}.attn_output.weight"), None, &a)?;
            for (v, a) in x.iter_mut().zip(a) {
                *v += a;
            }
            let mut xn = x.clone();
            ops::rms(&mut xn, &self.norms[i][3], c.eps)?;
            let logits = b.matvec(
                &self.gguf,
                &format!("{base}.ffn_gate_inp.weight"),
                None,
                &xn,
            )?;
            let selected = ops::route(&logits, c.top)?;
            let mut moe = vec![0.; c.hidden];
            for (e, weight) in selected {
                let mut gate = b.matvec(
                    &self.gguf,
                    &format!("{base}.ffn_gate_exps.weight"),
                    Some(e),
                    &xn,
                )?;
                let up = b.matvec(
                    &self.gguf,
                    &format!("{base}.ffn_up_exps.weight"),
                    Some(e),
                    &xn,
                )?;
                for (g, u) in gate.iter_mut().zip(up) {
                    *g = (*g / (1. + (-*g).exp())) * u;
                }
                let down = b.matvec(
                    &self.gguf,
                    &format!("{base}.ffn_down_exps.weight"),
                    Some(e),
                    &gate,
                )?;
                for (m, d) in moe.iter_mut().zip(down) {
                    *m += weight * d;
                }
            }
            for (v, m) in x.iter_mut().zip(moe) {
                *v += m;
            }
        }
        if !want_logits {
            self.position += 1;
            return Ok(Vec::new());
        }
        ops::rms(&mut x, &self.output_norm, c.eps)?;
        let out = if self.gguf.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        let logits = b.matvec(&self.gguf, out, None, &x)?;
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "nonfinite model output"
        );
        self.position += 1;
        Ok(logits)
    }
}
