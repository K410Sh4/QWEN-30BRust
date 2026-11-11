use anyhow::{ensure, Result};
use serde::Serialize;
use std::sync::mpsc;

#[derive(Clone, Default, Debug, Serialize)]
pub struct ProfileStats {
    pub gpu_timestamp_steps: u64,
    pub gpu_attention_ms: f64,
    pub gpu_router_ms: f64,
    pub gpu_expert_copies_ms: f64,
    pub gpu_experts_ms: f64,
    pub gpu_logits_ms: f64,
    pub host_graph_init_ms: f64,
    pub host_route_ms: f64,
    pub host_submit_ms: f64,
    pub host_router_wait_ms: f64,
    pub host_logits_wait_ms: f64,
    pub host_profile_read_ms: f64,
    pub host_weight_alloc_ms: f64,
    pub host_weight_write_ms: f64,
    pub host_weight_padding_ms: f64,
    pub expert_staged_bytes: u64,
    pub expert_uploaded_bytes: u64,
    pub expert_bank_copy_bytes: u64,
}

impl ProfileStats {
    pub(crate) fn delta(&self, before: &Self) -> Self {
        Self {
            gpu_timestamp_steps: self.gpu_timestamp_steps - before.gpu_timestamp_steps,
            gpu_attention_ms: self.gpu_attention_ms - before.gpu_attention_ms,
            gpu_router_ms: self.gpu_router_ms - before.gpu_router_ms,
            gpu_expert_copies_ms: self.gpu_expert_copies_ms - before.gpu_expert_copies_ms,
            gpu_experts_ms: self.gpu_experts_ms - before.gpu_experts_ms,
            gpu_logits_ms: self.gpu_logits_ms - before.gpu_logits_ms,
            host_graph_init_ms: self.host_graph_init_ms - before.host_graph_init_ms,
            host_route_ms: self.host_route_ms - before.host_route_ms,
            host_submit_ms: self.host_submit_ms - before.host_submit_ms,
            host_router_wait_ms: self.host_router_wait_ms - before.host_router_wait_ms,
            host_logits_wait_ms: self.host_logits_wait_ms - before.host_logits_wait_ms,
            host_profile_read_ms: self.host_profile_read_ms - before.host_profile_read_ms,
            host_weight_alloc_ms: self.host_weight_alloc_ms - before.host_weight_alloc_ms,
            host_weight_write_ms: self.host_weight_write_ms - before.host_weight_write_ms,
            host_weight_padding_ms: self.host_weight_padding_ms - before.host_weight_padding_ms,
            expert_staged_bytes: self.expert_staged_bytes - before.expert_staged_bytes,
            expert_uploaded_bytes: self.expert_uploaded_bytes - before.expert_uploaded_bytes,
            expert_bank_copy_bytes: self.expert_bank_copy_bytes - before.expert_bank_copy_bytes,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Attention,
    Router,
    ExpertCopies,
    Experts,
    Logits,
}

pub(crate) struct GpuProfile {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    staging: wgpu::Buffer,
    stages: Vec<Stage>,
    ns_per_tick: f64,
}

impl GpuProfile {
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue, layers: usize) -> Result<Self> {
        // Two timestamps per stage, four stages per layer and one for final logits.
        ensure!(
            layers <= (4096 - 2) / 8,
            "too many layers for Vulkan timestamp profiling"
        );
        let count = layers * 8 + 2;
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Qwen stage timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: count as u32,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (count * 8) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            queries,
            resolved: buffer(
                "Qwen resolved timestamps",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: buffer(
                "Qwen timestamp readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            stages: Vec::with_capacity(count / 2),
            ns_per_tick: f64::from(queue.get_timestamp_period()),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.stages.clear();
    }

    pub(crate) fn begin(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.write_timestamp(&self.queries, (self.stages.len() * 2) as u32);
    }

    pub(crate) fn end(&mut self, encoder: &mut wgpu::CommandEncoder, stage: Stage) {
        encoder.write_timestamp(&self.queries, (self.stages.len() * 2 + 1) as u32);
        self.stages.push(stage);
    }

    pub(crate) fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = (self.stages.len() * 2) as u32;
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.staging, 0, u64::from(count) * 8);
    }

    pub(crate) fn collect(&self, device: &wgpu::Device, stats: &mut ProfileStats) -> Result<()> {
        let slice = self.staging.slice(..(self.stages.len() * 16) as u64);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        {
            let data = slice.get_mapped_range();
            let ticks = bytemuck::cast_slice::<u8, [u64; 2]>(&data);
            for (&stage, pair) in self.stages.iter().zip(ticks) {
                let ms = pair[1].saturating_sub(pair[0]) as f64 * self.ns_per_tick / 1_000_000.;
                match stage {
                    Stage::Attention => stats.gpu_attention_ms += ms,
                    Stage::Router => stats.gpu_router_ms += ms,
                    Stage::ExpertCopies => stats.gpu_expert_copies_ms += ms,
                    Stage::Experts => stats.gpu_experts_ms += ms,
                    Stage::Logits => stats.gpu_logits_ms += ms,
                }
            }
        }
        self.staging.unmap();
        stats.gpu_timestamp_steps += 1;
        Ok(())
    }
}
