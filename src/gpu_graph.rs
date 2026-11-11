//! Device-resident token graph. The CPU reads only router logits and final logits.
//! Selected expert matrices are cached separately, copied into reusable GPU banks,
//! and evaluated in three batched dispatches per layer (gate/up/down).
use crate::{
    backend::{matrix, Vulkan},
    gguf::Gguf,
    model::Config,
    ops, quant,
};
use anyhow::{ensure, Context, Result};
use std::{collections::HashMap, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

struct Layer {
    norms: Vec<wgpu::Buffer>,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
}
pub struct Graph {
    pub context: usize,
    position: usize,
    frame: wgpu::Buffer,
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
    params: HashMap<[u32; 8], wgpu::Buffer>,
    layers: Vec<Layer>,
    output_norm: wgpu::Buffer,
    x: wgpu::Buffer,
    n: wgpu::Buffer,
    qraw: wgpu::Buffer,
    kraw: wgpu::Buffer,
    v: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    attention: wgpu::Buffer,
    projection: wgpu::Buffer,
    router: wgpu::Buffer,
    scores: wgpu::Buffer,
    gate: wgpu::Buffer,
    up: wgpu::Buffer,
    activated: wgpu::Buffer,
    down: wgpu::Buffer,
    rates: wgpu::Buffer,
    banks: Vec<wgpu::Buffer>,
    logits: wgpu::Buffer,
    router_readback: wgpu::Buffer,
    logits_readback: wgpu::Buffer,
}
fn buffer(
    v: &mut Vulkan,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> Result<wgpu::Buffer> {
    ensure!(
        size > 0 && size <= v.device.limits().max_buffer_size,
        "{label} exceeds Vulkan buffer limit"
    );
    if usage.contains(wgpu::BufferUsages::STORAGE) {
        ensure!(
            size <= v.device.limits().max_storage_buffer_binding_size as u64,
            "{label} exceeds Vulkan storage binding limit; reduce --ctx"
        );
    }
    v.stats.workspace_allocations += 1;
    Ok(v.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    }))
}
fn floats(v: &mut Vulkan, label: &str, count: usize) -> Result<wgpu::Buffer> {
    buffer(
        v,
        label,
        (count as u64)
            .checked_mul(4)
            .context("buffer size overflow")?,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
}
fn constant(v: &mut Vulkan, values: &[f32]) -> wgpu::Buffer {
    v.stats.workspace_allocations += 1;
    v.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("norm weights"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE,
        })
}
fn padded(n: usize) -> usize {
    (n + 3) & !3
}
impl Graph {
    pub fn new(v: &mut Vulkan, m: &Gguf, c: &Config, context: usize) -> Result<Self> {
        let d = c
            .kvheads
            .checked_mul(c.head_dim)
            .context("KV dimension overflow")?;
        let kv = d.checked_mul(context).context("KV length overflow")?;
        // Explicit budget protects against accidentally allocating the advertised 256K
        // context. This is a workspace safety limit, not a claim about free VRAM.
        ensure!((c.kv_bytes(context)? as u64)<=v.kv_budget,"GPU KV cache exceeds --gpu-kv-mib budget; reduce --ctx or explicitly increase that budget");
        v.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = v.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident operations"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ops.wgsl").into()),
        });
        let mut pipelines = HashMap::new();
        for entry in [
            "norm",
            "qrope",
            "krope",
            "store_kv",
            "scores",
            "normalize_scores",
            "attention_values",
            "residual",
            "swiglu",
            "mix",
        ] {
            pipelines.insert(
                entry,
                v.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: None,
                        module: &shader,
                        entry_point: entry,
                    }),
            );
        }
        if let Some(e) = pollster::block_on(v.device.pop_error_scope()) {
            anyhow::bail!("resident shader validation: {e}");
        }
        let mut layers = Vec::new();
        for i in 0..c.layers {
            let mut norms = Vec::new();
            for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"] {
                norms.push(constant(v, &m.row(&format!("blk.{i}.{name}.weight"), 0)?));
            }
            layers.push(Layer {
                norms,
                k: floats(v, "K cache", kv)?,
                v: floats(v, "V cache", kv)?,
            });
        }
        let output_norm = constant(v, &m.row("output_norm.weight", 0)?);
        let mut banks = Vec::new();
        for name in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let mut max = 0;
            for i in 0..c.layers {
                let (data, _, _, _) = matrix(m, &format!("blk.{i}.{name}.weight"), Some(0))?;
                max = max.max(padded(data.len()));
            }
            banks.push(buffer(
                v,
                "selected packed experts",
                (max as u64)
                    .checked_mul(c.top as u64)
                    .context("expert bank overflow")?,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )?);
        }
        let frame = buffer(
            v,
            "frame",
            48,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        )?;
        Ok(Self {
            context,
            position: 0,
            frame,
            pipelines,
            params: HashMap::new(),
            layers,
            output_norm,
            x: floats(v, "hidden", c.hidden)?,
            n: floats(v, "normalized hidden", c.hidden)?,
            qraw: floats(v, "Q raw", c.heads * c.head_dim)?,
            kraw: floats(v, "K raw", d)?,
            v: floats(v, "V current", d)?,
            q: floats(v, "Q rotated", c.heads * c.head_dim)?,
            k: floats(v, "K rotated", d)?,
            attention: floats(v, "attention", c.heads * c.head_dim)?,
            projection: floats(v, "attention projection", c.hidden)?,
            router: floats(v, "router", c.experts)?,
            scores: floats(
                v,
                "attention scores",
                c.heads
                    .checked_mul(context)
                    .context("score length overflow")?,
            )?,
            gate: floats(v, "expert gates", c.top * c.intermediate)?,
            up: floats(v, "expert up", c.top * c.intermediate)?,
            activated: floats(v, "expert activations", c.top * c.intermediate)?,
            down: floats(v, "expert down", c.top * c.hidden)?,
            rates: floats(v, "expert rates", c.top)?,
            banks,
            logits: floats(v, "logits", c.vocab)?,
            router_readback: buffer(
                v,
                "router staging",
                (c.experts * 4) as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )?,
            logits_readback: buffer(
                v,
                "logits staging",
                (c.vocab * 4) as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )?,
        })
    }
    fn op(
        &self,
        v: &mut Vulkan,
        e: &mut wgpu::CommandEncoder,
        name: &str,
        bindings: &[(u32, &wgpu::Buffer)],
        groups: u32,
    ) {
        let pipeline = &self.pipelines[name];
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: self.frame.as_entire_binding(),
        }];
        entries.extend(bindings.iter().map(|(binding, b)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: b.as_entire_binding(),
        }));
        let bind = v.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(name),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
        v.stats.gpu_dispatches += 1;
    }
    #[allow(clippy::too_many_arguments)]
    fn mm(
        v: &mut Vulkan,
        e: &mut wgpu::CommandEncoder,
        params: &mut HashMap<[u32; 8], wgpu::Buffer>,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
        shape: [u32; 8],
        batches: u32,
    ) {
        let uniform = params.entry(shape).or_insert_with(|| {
            v.stats.workspace_allocations += 1;
            v.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("matrix shape"),
                    contents: bytemuck::cast_slice(&shape),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
        });
        let bind = v.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident matvec"),
            layout: &v.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resident matvec"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&v.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(shape[1].min(65535), shape[1].div_ceil(65535), batches);
        v.stats.gpu_dispatches += 1;
        v.stats.matvecs += batches as u64;
    }
    #[allow(clippy::too_many_arguments)]
    fn matrix(
        v: &mut Vulkan,
        e: &mut wgpu::CommandEncoder,
        params: &mut HashMap<[u32; 8], wgpu::Buffer>,
        m: &Gguf,
        name: &str,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> Result<()> {
        let (data, kind, rows, cols) = matrix(m, name, None)?;
        let rb = quant::row_bytes(kind, cols)?;
        let limit = (v.device.limits().max_storage_buffer_binding_size as usize)
            .min(v.device.limits().max_buffer_size as usize)
            .min(u32::MAX as usize - 3);
        let tile_rows = limit / rb;
        ensure!(tile_rows > 0, "matrix row exceeds device limits");
        for first in (0..rows).step_by(tile_rows) {
            let n = tile_rows.min(rows - first);
            let data = &data[first * rb..(first + n) * rb];
            let w = v.upload_weight((data.as_ptr() as usize, data.len()), data)?;
            Self::mm(
                v,
                e,
                params,
                &w,
                x,
                out,
                [
                    cols as u32,
                    n as u32,
                    kind,
                    rb as u32,
                    first as u32,
                    0,
                    0,
                    0,
                ],
                1,
            );
        }
        Ok(())
    }
    fn submit(v: &mut Vulkan, e: wgpu::CommandEncoder) {
        v.queue.submit(Some(e.finish()));
        v.stats.submissions += 1;
    }
    fn read(v: &mut Vulkan, b: &wgpu::Buffer) -> Result<Vec<f32>> {
        let slice = b.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        v.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        b.unmap();
        v.stats.readbacks += 1;
        v.stats.readback_bytes += b.size();
        Ok(values)
    }
    pub fn forward(
        &mut self,
        v: &mut Vulkan,
        m: &Gguf,
        c: &Config,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let started = Instant::now();
        ensure!(
            pos == 0 || pos == self.position,
            "GPU cache position mismatch; reset model before switching backends"
        );
        let frame = [
            c.hidden as u32,
            c.heads as u32,
            c.kvheads as u32,
            c.head_dim as u32,
            c.intermediate as u32,
            c.top as u32,
            self.context as u32,
            pos as u32,
            c.eps.to_bits(),
            c.theta.to_bits(),
            c.experts as u32,
            c.vocab as u32,
        ];
        v.queue
            .write_buffer(&self.frame, 0, bytemuck::cast_slice(&frame));
        v.queue.write_buffer(
            &self.x,
            0,
            bytemuck::cast_slice(&m.row("token_embd.weight", token as usize)?),
        );
        // Carry the expert batch into the next attention/router submission.
        // Only router readbacks and final logits require a queue boundary.
        let mut e = v
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resident token graph"),
            });
        for i in 0..c.layers {
            let base = format!("blk.{i}");
            self.op(
                v,
                &mut e,
                "norm",
                &[(1, &self.x), (2, &self.layers[i].norms[0]), (3, &self.n)],
                1,
            );
            for (name, out) in [
                ("attn_q", &self.qraw),
                ("attn_k", &self.kraw),
                ("attn_v", &self.v),
            ] {
                Self::matrix(
                    v,
                    &mut e,
                    &mut self.params,
                    m,
                    &format!("{base}.{name}.weight"),
                    &self.n,
                    out,
                )?;
            }
            self.op(
                v,
                &mut e,
                "qrope",
                &[(1, &self.qraw), (2, &self.layers[i].norms[1]), (3, &self.q)],
                c.heads as u32,
            );
            self.op(
                v,
                &mut e,
                "krope",
                &[(1, &self.kraw), (2, &self.layers[i].norms[2]), (3, &self.k)],
                c.kvheads as u32,
            );
            self.op(
                v,
                &mut e,
                "store_kv",
                &[
                    (1, &self.k),
                    (2, &self.v),
                    (3, &self.layers[i].k),
                    (4, &self.layers[i].v),
                ],
                (c.kvheads * c.head_dim).div_ceil(64) as u32,
            );
            self.op(
                v,
                &mut e,
                "scores",
                &[(1, &self.q), (2, &self.layers[i].k), (3, &self.scores)],
                (c.heads * (pos + 1)).div_ceil(64) as u32,
            );
            self.op(
                v,
                &mut e,
                "normalize_scores",
                &[(3, &self.scores)],
                c.heads as u32,
            );
            self.op(
                v,
                &mut e,
                "attention_values",
                &[
                    (1, &self.scores),
                    (2, &self.layers[i].v),
                    (3, &self.attention),
                ],
                (c.heads * c.head_dim).div_ceil(64) as u32,
            );
            Self::matrix(
                v,
                &mut e,
                &mut self.params,
                m,
                &format!("{base}.attn_output.weight"),
                &self.attention,
                &self.projection,
            )?;
            self.op(
                v,
                &mut e,
                "residual",
                &[(1, &self.projection), (3, &self.x)],
                c.hidden.div_ceil(64) as u32,
            );
            self.op(
                v,
                &mut e,
                "norm",
                &[(1, &self.x), (2, &self.layers[i].norms[3]), (3, &self.n)],
                1,
            );
            Self::matrix(
                v,
                &mut e,
                &mut self.params,
                m,
                &format!("{base}.ffn_gate_inp.weight"),
                &self.n,
                &self.router,
            )?;
            e.copy_buffer_to_buffer(
                &self.router,
                0,
                &self.router_readback,
                0,
                (c.experts * 4) as u64,
            );
            Self::submit(v, e);
            let selected = ops::route(&Self::read(v, &self.router_readback)?, c.top)?;
            let rates: Vec<f32> = selected.iter().map(|v| v.1).collect();
            v.queue
                .write_buffer(&self.rates, 0, bytemuck::cast_slice(&rates));
            e = v
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("experts and next attention"),
                });
            let mut shapes = Vec::new();
            for (bank, name) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .enumerate()
            {
                let mut shape = [0; 8];
                for (slot, &(expert, _)) in selected.iter().enumerate() {
                    let (data, kind, rows, cols) =
                        matrix(m, &format!("{base}.{name}.weight"), Some(expert))?;
                    let stride = padded(data.len());
                    let w = v.upload_weight((data.as_ptr() as usize, data.len()), data)?;
                    e.copy_buffer_to_buffer(
                        &w,
                        0,
                        &self.banks[bank],
                        (slot * stride) as u64,
                        stride as u64,
                    );
                    shape = [
                        cols as u32,
                        rows as u32,
                        kind,
                        quant::row_bytes(kind, cols)? as u32,
                        0,
                        stride as u32,
                        if bank == 2 { c.intermediate as u32 } else { 0 },
                        0,
                    ];
                }
                shapes.push(shape);
            }
            Self::mm(
                v,
                &mut e,
                &mut self.params,
                &self.banks[0],
                &self.n,
                &self.gate,
                shapes[0],
                c.top as u32,
            );
            Self::mm(
                v,
                &mut e,
                &mut self.params,
                &self.banks[1],
                &self.n,
                &self.up,
                shapes[1],
                c.top as u32,
            );
            self.op(
                v,
                &mut e,
                "swiglu",
                &[(1, &self.gate), (2, &self.up), (3, &self.activated)],
                (c.top * c.intermediate).div_ceil(64) as u32,
            );
            Self::mm(
                v,
                &mut e,
                &mut self.params,
                &self.banks[2],
                &self.activated,
                &self.down,
                shapes[2],
                c.top as u32,
            );
            self.op(
                v,
                &mut e,
                "mix",
                &[(1, &self.down), (2, &self.rates), (3, &self.x)],
                c.hidden.div_ceil(64) as u32,
            );
        }
        if !want_logits {
            Self::submit(v, e);
            self.position = pos + 1;
            v.stats.device_forward_ms += started.elapsed().as_secs_f64() * 1000.;
            return Ok(Vec::new());
        }
        self.op(
            v,
            &mut e,
            "norm",
            &[(1, &self.x), (2, &self.output_norm), (3, &self.n)],
            1,
        );
        let name = if m.tensors.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        Self::matrix(v, &mut e, &mut self.params, m, name, &self.n, &self.logits)?;
        e.copy_buffer_to_buffer(
            &self.logits,
            0,
            &self.logits_readback,
            0,
            (c.vocab * 4) as u64,
        );
        Self::submit(v, e);
        let result = Self::read(v, &self.logits_readback)?;
        self.position = pos + 1;
        v.stats.device_forward_ms += started.elapsed().as_secs_f64() * 1000.;
        Ok(result)
    }
}
