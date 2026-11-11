use crate::{gguf::Gguf, quant};
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{collections::HashMap, sync::mpsc, time::Instant};
use wgpu::util::DeviceExt;

#[derive(Default, Serialize, Debug)]
pub struct Stats {
    pub matvecs: u64,
    pub uploaded_bytes: u64,
    pub cache_hits: u64,
    pub matvec_ms: f64,
}
pub trait Backend {
    fn matvec(
        &mut self,
        model: &Gguf,
        name: &str,
        expert: Option<usize>,
        x: &[f32],
    ) -> Result<Vec<f32>>;
    fn stats(&self) -> &Stats;
}
#[derive(Default)]
pub struct Cpu {
    stats: Stats,
}
impl Backend for Cpu {
    fn matvec(
        &mut self,
        m: &Gguf,
        name: &str,
        expert: Option<usize>,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let start = Instant::now();
        let (data, kind, rows, cols) = matrix(m, name, expert)?;
        ensure!(cols == x.len(), "matvec shape mismatch {name}");
        let rb = quant::row_bytes(kind, cols)?;
        let mut out = vec![0.; rows];
        for (row, y) in out.iter_mut().enumerate() {
            let b = &data[row * rb..(row + 1) * rb];
            *y = x
                .iter()
                .enumerate()
                .map(|(i, &v)| v * quant::value(b, kind, i))
                .sum();
        }
        self.stats.matvecs += 1;
        self.stats.matvec_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(out)
    }
    fn stats(&self) -> &Stats {
        &self.stats
    }
}
pub fn matrix<'a>(
    m: &'a Gguf,
    name: &str,
    expert: Option<usize>,
) -> Result<(&'a [u8], u32, usize, usize)> {
    let t = m.tensor(name)?;
    ensure!(
        t.dims.len() == if expert.is_some() { 3 } else { 2 },
        "matrix rank mismatch: {name}"
    );
    let cols = t.dims[0];
    let rows = t.dims[1];
    let len = quant::row_bytes(t.kind, cols)?
        .checked_mul(rows)
        .context("matrix overflow")?;
    let e = expert.unwrap_or(0);
    ensure!(expert.is_none() || e < t.dims[2], "expert out of bounds");
    Ok((&m.data(t)[e * len..(e + 1) * len], t.kind, rows, cols))
}
struct Cached {
    buffer: wgpu::Buffer,
    bytes: u64,
    age: u64,
}
pub struct Vulkan {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    cache: HashMap<(usize, usize), Cached>,
    budget: u64,
    used: u64,
    clock: u64,
    stats: Stats,
    pub adapter_name: String,
    pub software: bool,
    active_model: usize,
}
impl Vulkan {
    pub fn new(budget_mib: u64, filter: Option<&str>, allow_software: bool) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapters = instance.enumerate_adapters(wgpu::Backends::VULKAN);
        let mut adapters: Vec<_> = adapters
            .into_iter()
            .filter(|a| {
                let info = a.get_info();
                (allow_software || info.device_type != wgpu::DeviceType::Cpu)
                    && filter
                        .map(|f| info.name.to_lowercase().contains(&f.to_lowercase()))
                        .unwrap_or(true)
            })
            .collect();
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            wgpu::DeviceType::Cpu => 3,
            _ => 2,
        });
        let adapter=adapters.into_iter().next().context("no matching hardware Vulkan adapter; install/update the GPU driver. Software devices require --allow-software")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        ensure!(
            limits.max_compute_workgroups_per_dimension >= 65535,
            "Vulkan grid limit too small"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("qwen-rust"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ))?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quantized matvec"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/matvec.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: "main",
        });
        if let Some(e) = pollster::block_on(device.pop_error_scope()) {
            anyhow::bail!("GPU shader validation failed: {e}");
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            cache: HashMap::new(),
            budget: budget_mib
                .checked_mul(1024 * 1024)
                .context("cache budget overflow")?,
            used: 0,
            clock: 0,
            stats: Stats::default(),
            adapter_name: info.name,
            software: info.device_type == wgpu::DeviceType::Cpu,
            active_model: 0,
        })
    }
    pub fn devices() -> Vec<String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .map(|a| format!("{} ({:?})", a.get_info().name, a.get_info().device_type))
            .collect()
    }
    /// Public for direct numerical validation with known matrices.
    pub fn packed_matvec(
        &mut self,
        key: (usize, usize),
        data: &[u8],
        kind: u32,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let start = Instant::now();
        let rb = quant::row_bytes(kind, cols)?;
        ensure!(
            rows > 0 && cols == x.len() && rows.checked_mul(rb) == Some(data.len()),
            "GPU matvec shape mismatch"
        );
        ensure!(rows <= 65535usize * 65535, "matrix too tall");
        let bytes = (data.len() as u64 + 3) & !3;
        let limit = (self.device.limits().max_storage_buffer_binding_size as u64)
            .min(self.device.limits().max_buffer_size)
            .min(u32::MAX as u64);
        if bytes > limit {
            let tile_rows = (limit as usize / rb).min(65535);
            ensure!(
                tile_rows > 0,
                "one matrix row exceeds the Vulkan buffer limit"
            );
            let mut output = Vec::with_capacity(rows);
            for first in (0..rows).step_by(tile_rows) {
                let n = tile_rows.min(rows - first);
                let tile = &data[first * rb..(first + n) * rb];
                output.extend(self.packed_matvec(
                    (key.0 + first * rb, tile.len()),
                    tile,
                    kind,
                    n,
                    cols,
                    x,
                )?);
            }
            return Ok(output);
        }
        self.clock += 1;
        let ephemeral;
        if self.cache.contains_key(&key) {
            self.stats.cache_hits += 1;
            self.cache.get_mut(&key).unwrap().age = self.clock;
        } else {
            while self.used + bytes > self.budget && !self.cache.is_empty() {
                let old = *self.cache.iter().min_by_key(|(_, v)| v.age).unwrap().0;
                self.used -= self.cache.remove(&old).unwrap().bytes;
            }
            let mut padded = Vec::new();
            let packed = if !data.len().is_multiple_of(4) {
                padded.extend_from_slice(data);
                padded.resize(bytes as usize, 0);
                &padded[..]
            } else {
                data
            };
            let buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("packed weights"),
                    contents: packed,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            self.stats.uploaded_bytes += bytes;
            if bytes <= self.budget {
                self.used += bytes;
                self.cache.insert(
                    key,
                    Cached {
                        buffer,
                        bytes,
                        age: self.clock,
                    },
                );
            } else {
                ephemeral = buffer;
                return self.dispatch(&ephemeral, kind, rows, cols, rb, x, start);
            }
        }
        // Temporarily remove to borrow self mutably while retaining the buffer.
        let cached = self.cache.remove(&key).unwrap();
        let result = self.dispatch(&cached.buffer, kind, rows, cols, rb, x, start);
        self.cache.insert(key, cached);
        result
    }
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &mut self,
        w: &wgpu::Buffer,
        kind: u32,
        rows: usize,
        cols: usize,
        rb: usize,
        x: &[f32],
        start: Instant,
    ) -> Result<Vec<f32>> {
        let input = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (rows * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let params = [
            u32::try_from(cols)?,
            u32::try_from(rows)?,
            kind,
            u32::try_from(rb)?,
        ];
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(rows.min(65535) as u32, rows.div_ceil(65535) as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (rows * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        readback.unmap();
        self.stats.matvecs += 1;
        self.stats.matvec_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(values)
    }
}
impl Backend for Vulkan {
    fn matvec(&mut self, m: &Gguf, name: &str, e: Option<usize>, x: &[f32]) -> Result<Vec<f32>> {
        if self.active_model != m.id {
            self.cache.clear();
            self.used = 0;
            self.active_model = m.id;
        }
        let (data, kind, rows, cols) = matrix(m, name, e)?;
        // Actual memory address + byte length makes cache identity unique per mapped model.
        self.packed_matvec(
            (data.as_ptr() as usize, data.len()),
            data,
            kind,
            rows,
            cols,
            x,
        )
    }
    fn stats(&self) -> &Stats {
        &self.stats
    }
}
