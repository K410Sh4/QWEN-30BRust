use qwen_rust::{backend::Cpu, gguf::Gguf, model::Model};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tiny-qwen3moe.gguf")
}
#[test]
fn chat_cli_runs_with_matching_tokenizer_and_rejects_control_tokens() {
    let tok = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/tokenizer.json");
    for (prompt, success) in [("hello", true), ("<|im_end|>", false)] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_qwen-rust"))
            .args(["chat", "--model"])
            .arg(fixture())
            .arg("--tokenizer")
            .arg(&tok)
            .args([
                "--prompt",
                prompt,
                "--system",
                "teste",
                "--ctx",
                "32",
                "--max-tokens",
                "2",
                "--temperature",
                "0",
                "--device",
                "cpu",
            ])
            .output()
            .unwrap();
        assert_eq!(
            out.status.success(),
            success,
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        if success {
            assert!(String::from_utf8_lossy(&out.stderr).contains("generated_tokens"));
        } else {
            assert!(String::from_utf8_lossy(&out.stderr).contains("reserved special tokens"));
        }
    }
}
#[test]
fn full_forward_matches_independent_reference_and_reset() {
    let expected: Vec<Vec<f32>> =
        serde_json::from_str(include_str!("data/expected-logits.json")).unwrap();
    let mut model = Model::new(Gguf::open(&fixture()).unwrap(), 4).unwrap();
    let mut cpu = Cpu::default();
    for (&id, expected) in [1, 5, 2, 9].iter().zip(&expected) {
        let actual = model.forward(id, &mut cpu).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 2e-5, "{a} != {b}");
        }
    }
    assert_eq!(model.position, 4);
    assert!(model.forward(1, &mut cpu).is_err());
    assert_eq!(model.position, 0);
    let again = model.forward(1, &mut cpu).unwrap();
    for (a, b) in again.iter().zip(&expected[0]) {
        assert!((a - b).abs() < 2e-5);
    }
    assert!(model.forward(999, &mut cpu).is_err());
    assert_eq!(model.position, 0);
}
static NEXT: AtomicU64 = AtomicU64::new(0);
fn reject(bytes: &[u8]) {
    let p = std::env::temp_dir().join(format!(
        "qwen-gguf-test-{}-{}.gguf",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&p, bytes).unwrap();
    let result = Gguf::open(&p);
    assert!(result.is_err());
    drop(result);
    fs::remove_file(&p).unwrap();
}
#[test]
fn malformed_headers_and_truncated_data_are_rejected() {
    let bytes = fs::read(fixture()).unwrap();
    reject(&bytes[..10]);
    reject(&bytes[..bytes.len() - 20]);
    let mut v = bytes.clone();
    v[..4].copy_from_slice(b"NOPE");
    reject(&v);
    let mut v = bytes.clone();
    v[4..8].copy_from_slice(&999u32.to_le_bytes());
    reject(&v);
    let mut v = bytes;
    v[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    reject(&v);
}
#[test]
#[ignore = "requires Vulkan; enable explicitly"]
fn full_vulkan_forward_matches_numpy() {
    let allow = std::env::var("QWEN_TEST_ALLOW_SOFTWARE").as_deref() == Ok("1");
    let mut gpu = qwen_rust::backend::Vulkan::new(1, None, allow).unwrap();
    let expected: Vec<Vec<f32>> =
        serde_json::from_str(include_str!("data/expected-logits.json")).unwrap();
    let mut model = Model::new(Gguf::open(&fixture()).unwrap(), 4).unwrap();
    for (&id, expected) in [1, 5, 2, 9].iter().zip(&expected) {
        let actual = model.forward(id, &mut gpu).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 2e-5);
        }
    }
    // Reusing the backend for a different model must invalidate cached tensor data.
    let mut model = Model::new(Gguf::open(&fixture()).unwrap(), 4).unwrap();
    let actual = model.forward(1, &mut gpu).unwrap();
    for (a, b) in actual.iter().zip(&expected[0]) {
        assert!((a - b).abs() < 2e-5);
    }
}
