use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use qwen_rust::{
    backend::{Backend, Cpu, Vulkan},
    gguf::Gguf,
    model::{Config, Model},
    sampling::Sampler,
};
use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(
    version,
    about = "Experimental native Rust Qwen3 MoE inference; Vulkan matrix kernels, CPU vector/attention operations"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Clone, Copy, ValueEnum)]
enum Device {
    Cpu,
    Vulkan,
}
#[derive(clap::Args)]
struct Compute {
    #[arg(long, value_enum, default_value = "vulkan")]
    device: Device,
    /// GPU packed-weight cache only; does not include working buffers or driver memory.
    #[arg(long, default_value_t = 4096)]
    gpu_cache_mib: u64,
    /// Substring of Vulkan adapter name. No silent fallback to CPU.
    #[arg(long)]
    adapter: Option<String>,
    #[arg(long)]
    allow_software: bool,
}
impl Compute {
    fn backend(&self) -> Result<Box<dyn Backend>> {
        Ok(match self.device {
            Device::Cpu => Box::new(Cpu::default()),
            Device::Vulkan => {
                let v = Vulkan::new(
                    self.gpu_cache_mib,
                    self.adapter.as_deref(),
                    self.allow_software,
                )?;
                eprintln!(
                    "Vulkan adapter: {} | software={}",
                    v.adapter_name, v.software
                );
                Box::new(v)
            }
        })
    }
}
#[derive(Subcommand)]
enum Command {
    /// Show available Vulkan devices.
    Devices,
    /// Parse and validate a single-file Qwen3 MoE GGUF, without allocating GPU buffers.
    Inspect {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
    },
    /// Compute logits for explicit token IDs; useful for parity checks.
    Logits {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', required = true)]
        tokens: Vec<u32>,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        #[command(flatten)]
        compute: Compute,
    },
    /// One-turn chat with the original tokenizer.json. The model weights are not bundled.
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "You are a helpful coding assistant.")]
        system: String,
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.2)]
        temperature: f32,
        #[arg(long, default_value_t = 20)]
        top_k: usize,
        #[arg(long, default_value_t = 0.8)]
        top_p: f32,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Encode the prompt directly, without ChatML wrappers.
        #[arg(long)]
        raw: bool,
        #[command(flatten)]
        compute: Compute,
    },
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Devices => {
            let devices = Vulkan::devices();
            ensure!(!devices.is_empty(), "no Vulkan adapter found");
            for d in devices {
                println!("{d}");
            }
        }
        Command::Inspect { model, ctx } => {
            let g = Gguf::open(&model)?;
            let c = Config::from_gguf(&g)?;
            c.validate_tensors(&g)?;
            ensure!(ctx > 0 && ctx <= c.max_context, "invalid context");
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"config":c,"tensor_count":g.tensors.len(),"packed_weight_bytes":g.tensors.values().map(|t|t.len as u64).sum::<u64>(),"kv_f32_bytes_at_context":c.kv_bytes(ctx)?})
                )?
            );
        }
        Command::Logits {
            model,
            tokens,
            ctx,
            compute,
        } => {
            ensure!(tokens.len() <= ctx, "tokens exceed context");
            let mut m = Model::new(Gguf::open(&model)?, ctx)?;
            let mut b = compute.backend()?;
            let start = Instant::now();
            let mut logits = Vec::new();
            for id in tokens {
                logits = m.forward(id, &mut *b)?;
            }
            println!(
                "{}",
                serde_json::to_string(
                    &serde_json::json!({"logits":logits,"seconds":start.elapsed().as_secs_f64(),"stats":b.stats()})
                )?
            );
        }
        Command::Chat {
            model,
            tokenizer,
            prompt,
            system,
            ctx,
            max_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            raw,
            compute,
        } => {
            ensure!(max_tokens > 0, "max-tokens must be positive");
            let tok = tokenizers::Tokenizer::from_file(&tokenizer)
                .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
            let gguf = Gguf::open(&model)?;
            // Validate token-id identity using the GGUF token table, not vocabulary size alone.
            let table = gguf
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .context("GGUF has no token table")?;
            ensure!(
                tok.get_vocab_size(true) <= table.len(),
                "tokenizer vocabulary exceeds GGUF"
            );
            for (text, id) in tok.get_vocab(true) {
                ensure!(
                    table.get(id as usize).and_then(|v| v.as_str()) == Some(text.as_str()),
                    "tokenizer differs from GGUF at token {id}"
                );
            }
            // Qwen's embedding size may be padded beyond the actual tokenizer vocabulary.
            let valid_tokens: Vec<bool> = (0..table.len())
                .map(|id| tok.id_to_token(id as u32).is_some())
                .collect();
            let ids = if raw {
                tok.encode(prompt.as_str(), false)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec()
            } else {
                let im_start = tok
                    .token_to_id("<|im_start|>")
                    .context("missing ChatML im_start token")?;
                let im_end = tok
                    .token_to_id("<|im_end|>")
                    .context("missing ChatML im_end token")?;
                // Encode untrusted content independently; reject model control tokens in content.
                let vocab = tok.get_added_tokens_decoder();
                for content in [system.as_str(), prompt.as_str()] {
                    let body = tok
                        .encode(content, false)
                        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                    ensure!(!body.get_ids().iter().any(|id|vocab.get(id).map(|t|t.special).unwrap_or(false)),"prompt/system contains reserved special tokens; use --raw for exact tokenized inputs");
                }
                let _ = (im_start, im_end);
                let chat=format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
                tok.encode(chat, false)
                    .map_err(|e| anyhow::anyhow!("encode: {e}"))?
                    .get_ids()
                    .to_vec()
            };
            ensure!(
                !ids.is_empty()
                    && ids
                        .len()
                        .checked_add(max_tokens)
                        .map(|n| n <= ctx)
                        .unwrap_or(false),
                "prompt + max-tokens exceed --ctx; reduce the prompt or generation length"
            );
            let eos = gguf
                .metadata
                .get("tokenizer.ggml.eos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);
            let mut stops = Vec::new();
            if let Some(eos) = eos {
                stops.push(eos);
            }
            for t in ["<|im_end|>", "<|endoftext|>"] {
                if let Some(id) = tok.token_to_id(t) {
                    stops.push(id);
                }
            }
            ensure!(!stops.is_empty(), "no EOS token found");
            let mut sampler = Sampler::new(seed, temperature, top_k, top_p)?;
            let mut m = Model::new(gguf, ctx)?;
            eprintln!(
                "KV cache maximum: {:.1} MiB (FP32, CPU RAM)",
                m.config.kv_bytes(ctx)? as f64 / 1048576.
            );
            let mut b = compute.backend()?;
            let start = Instant::now();
            let mut logits = Vec::new();
            for (id, &token) in ids.iter().enumerate() {
                logits = m.forward(token, &mut *b)?;
                eprint!("\rPrefill {}/{}", id + 1, ids.len());
            }
            let prefill_seconds = start.elapsed().as_secs_f64();
            eprintln!();
            let decode = Instant::now();
            let mut count = 0;
            let mut stream = tok.decode_stream(true);
            for i in 0..max_tokens {
                for (logit, &valid) in logits.iter_mut().zip(&valid_tokens) {
                    if !valid {
                        *logit = f32::MIN;
                    }
                }
                let id = sampler.sample(&logits)?;
                if stops.contains(&id) {
                    break;
                }
                if let Some(text) = stream
                    .step(id)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?
                {
                    print!("{text}");
                    io::stdout().flush()?;
                }
                count += 1;
                if i + 1 < max_tokens {
                    logits = m.forward(id, &mut *b)?;
                }
            }
            println!();
            eprintln!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"prompt_tokens":ids.len(),"generated_tokens":count,"prefill_seconds":prefill_seconds,"generation_seconds":decode.elapsed().as_secs_f64(),"stats":b.stats(),"note":"timings include orchestration, transfers and CPU operations; not a pure GPU benchmark"})
                )?
            );
        }
    }
    Ok(())
}
