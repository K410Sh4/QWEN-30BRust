use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct Fixture(pub PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn string(dst: &mut Vec<u8>, s: &str) {
    dst.extend((s.len() as u64).to_le_bytes());
    dst.extend(s.as_bytes());
}
pub fn long_context() -> Fixture {
    let mut bytes = fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny-qwen3moe.gguf"),
    )
    .unwrap();
    let key = b"qwen3moe.context_length";
    let at = bytes.windows(key.len()).position(|w| w == key).unwrap() + key.len();
    assert_eq!(&bytes[at..at + 4], &4u32.to_le_bytes());
    bytes[at + 4..at + 8].copy_from_slice(&256u32.to_le_bytes());
    let path = std::env::temp_dir().join(format!(
        "qwen-context-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, bytes).unwrap();
    Fixture(path)
}
/// Reproducible, untrained GGUF. No Python, network, real weights or large git blobs.
pub fn quantized() -> Fixture {
    let path = std::env::temp_dir().join(format!(
        "qwen-mixed-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut metadata = Vec::new();
    metadata.push(("general.architecture".to_string(), 8u32, {
        let mut b = Vec::new();
        string(&mut b, "qwen3moe");
        b
    }));
    for (key, value) in [
        ("embedding_length", 256u32),
        ("block_count", 2),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 64),
        ("attention.value_length", 64),
        ("expert_count", 4),
        ("expert_used_count", 2),
        ("expert_feed_forward_length", 256),
        ("context_length", 32),
    ] {
        metadata.push((format!("qwen3moe.{key}"), 4, value.to_le_bytes().to_vec()));
    }
    metadata.push((
        "qwen3moe.rope.freq_base".to_string(),
        6,
        10000f32.to_le_bytes().to_vec(),
    ));
    metadata.push((
        "qwen3moe.attention.layer_norm_rms_epsilon".to_string(),
        6,
        0.000001f32.to_le_bytes().to_vec(),
    ));
    let mut shapes = vec![
        ("token_embd.weight".to_string(), vec![256, 16]),
        ("output.weight".to_string(), vec![256, 16]),
        ("output_norm.weight".to_string(), vec![256]),
    ];
    for i in 0..2 {
        for (name, dims) in [
            ("attn_norm", vec![256]),
            ("ffn_norm", vec![256]),
            ("attn_q_norm", vec![64]),
            ("attn_k_norm", vec![64]),
            ("attn_q", vec![256, 256]),
            ("attn_k", vec![256, 128]),
            ("attn_v", vec![256, 128]),
            ("attn_output", vec![256, 256]),
            ("ffn_gate_inp", vec![256, 4]),
            ("ffn_gate_exps", vec![256, 256, 4]),
            ("ffn_up_exps", vec![256, 256, 4]),
            ("ffn_down_exps", vec![256, 256, 4]),
        ] {
            shapes.push((format!("blk.{i}.{name}.weight"), dims));
        }
    }
    let mut header = b"GGUF".to_vec();
    header.extend(3u32.to_le_bytes());
    header.extend((shapes.len() as u64).to_le_bytes());
    header.extend((metadata.len() as u64).to_le_bytes());
    for (key, kind, bytes) in metadata {
        string(&mut header, &key);
        header.extend(kind.to_le_bytes());
        header.extend(bytes);
    }
    let mut payload = Vec::new();
    let mut state = 20261110u64;
    for (index, (name, dims)) in shapes.into_iter().enumerate() {
        let count = dims.iter().product::<usize>();
        let norm = name.contains("norm");
        let kind = if norm {
            1
        } else {
            [12u32, 13, 14, 8, 2][index % 5]
        };
        let (block, bytes) = match kind {
            1 => (1, 2),
            12 => (256, 144),
            13 => (256, 176),
            14 => (256, 210),
            8 => (32, 34),
            2 => (32, 18),
            _ => unreachable!(),
        };
        let mut packed = vec![0u8; count / block * bytes];
        if norm {
            for h in packed.as_chunks_mut::<2>().0 {
                h.copy_from_slice(&half::f16::from_f32(1.).to_bits().to_le_bytes());
            }
        } else {
            for b in packed.chunks_exact_mut(bytes) {
                for x in b.iter_mut() {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    *x = (state >> 32) as u8;
                }
                let h = |x: f32| half::f16::from_f32(x).to_bits().to_le_bytes();
                match kind {
                    12 | 13 => {
                        b[..2].copy_from_slice(&h(0.0001));
                        b[2..4].copy_from_slice(&h(0.0005));
                    }
                    14 => {
                        for x in &mut b[192..208] {
                            *x = ((*x % 9) as i8 - 4) as u8;
                        }
                        b[208..210].copy_from_slice(&h(0.0002));
                    }
                    8 => b[..2].copy_from_slice(&h(0.0003)),
                    2 => b[..2].copy_from_slice(&h(0.003)),
                    _ => unreachable!(),
                }
            }
        }
        payload.resize(payload.len().div_ceil(32) * 32, 0);
        string(&mut header, &name);
        header.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            header.extend((d as u64).to_le_bytes());
        }
        header.extend(kind.to_le_bytes());
        header.extend((payload.len() as u64).to_le_bytes());
        payload.extend(packed);
    }
    header.resize(header.len().div_ceil(32) * 32, 0);
    header.extend(payload);
    fs::write(&path, header).unwrap();
    Fixture(path)
}
