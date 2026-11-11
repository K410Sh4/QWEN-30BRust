mod common;
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

#[test]
fn skipping_prefill_logits_preserves_causal_state() {
    let mut full = Model::new(Gguf::open(&fixture()).unwrap(), 8).unwrap();
    let mut fast = Model::new(Gguf::open(&fixture()).unwrap(), 8).unwrap();
    let mut a = Cpu::default();
    let mut b = Cpu::default();
    let mut expected = Vec::new();
    for token in [1, 5, 2] {
        expected = full.forward(token, &mut a).unwrap();
    }
    fast.forward_without_logits(1, &mut b).unwrap();
    fast.forward_without_logits(5, &mut b).unwrap();
    assert_eq!(fast.forward(2, &mut b).unwrap(), expected);
    use qwen_rust::backend::Backend;
    assert_eq!(a.stats().matvecs - b.stats().matvecs, 2);
}

#[test]
#[ignore = "requires Vulkan; checks persistent allocations, counters, reset and skipped prefill"]
fn resident_state_and_counters() {
    use qwen_rust::backend::{Backend, Vulkan};
    let allow = std::env::var("QWEN_TEST_ALLOW_SOFTWARE").as_deref() == Ok("1");
    for budget in [0, 1] {
        let mut gpu = Vulkan::new(budget, None, allow).unwrap();
        let mut m = Model::new(Gguf::open(&fixture()).unwrap(), 8).unwrap();
        m.forward_without_logits(1, &mut gpu).unwrap();
        let allocations = gpu.stats().workspace_allocations;
        m.forward_without_logits(5, &mut gpu).unwrap();
        let actual = m.forward(2, &mut gpu).unwrap();
        let expected: Vec<Vec<f32>> =
            serde_json::from_str(include_str!("data/expected-logits.json")).unwrap();
        for (a, b) in actual.iter().zip(&expected[2]) {
            assert!((a - b).abs() < 2e-5);
        }
        assert_eq!(gpu.stats().readbacks, 7); // 3*2 routers + 1 final logits.
                                              // The only extra persistent object is the LM-head matrix-shape uniform.
        assert!(gpu.stats().workspace_allocations <= allocations + 1);
        m.reset();
        let actual = m.forward(1, &mut gpu).unwrap();
        for (a, b) in actual.iter().zip(&expected[0]) {
            assert!((a - b).abs() < 2e-5);
        }
    }
    let mut gpu = Vulkan::new(0, None, allow)
        .unwrap()
        .kv_budget_mib(0)
        .unwrap();
    let mut m = Model::new(Gguf::open(&fixture()).unwrap(), 8).unwrap();
    assert!(m
        .forward(1, &mut gpu)
        .unwrap_err()
        .to_string()
        .contains("GPU KV cache"));
    assert_eq!(m.position, 0);
}

#[test]
#[ignore = "requires Vulkan; compares mixed packed expert banks to CPU and reference Vulkan"]
fn mixed_quantized_resident_matches_reference() {
    use qwen_rust::backend::Vulkan;
    let allow = std::env::var("QWEN_TEST_ALLOW_SOFTWARE").as_deref() == Ok("1");
    let file = common::quantized();
    let path = &file.0;
    let mut cpu = Cpu::default();
    let mut model = Model::new(Gguf::open(path).unwrap(), 8).unwrap();
    let mut expected = Vec::new();
    for id in [1, 5, 2, 9] {
        expected.push(model.forward(id, &mut cpu).unwrap());
    }
    for reference in [true, false] {
        let mut gpu = Vulkan::new(1, None, allow).unwrap().reference(reference); // force eviction on this >1MiB model.
        let mut m = Model::new(Gguf::open(path).unwrap(), 8).unwrap();
        let mut actual = Vec::new();
        for id in [1, 5, 2, 9] {
            actual.push(m.forward(id, &mut gpu).unwrap());
        }
        let comparison = qwen_rust::benchmark::parity(&expected, &actual, 0.00005, 0.0002).unwrap();
        assert!(
            comparison.passed,
            "reference={reference}: max_error={} mismatches={}",
            comparison.max_absolute_error, comparison.mismatches
        );
    }
}

#[test]
fn benchmark_cli_reports_measured_steps() {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_qwen-rust"))
        .args(["bench", "--model"])
        .arg(fixture())
        .args([
            "--tokens", "1,5", "--ctx", "8", "--runs", "2", "--warmup", "0", "--device", "cpu",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["measured_seconds"].as_array().unwrap().len(), 2);
    assert!(report["forward_tokens_per_second"].as_f64().unwrap() > 0.);
    assert_eq!(report["measured_stats"][0]["matvecs"], 46);
}

#[test]
#[ignore = "requires Vulkan; covers attention softmax across the 64-thread boundary"]
fn resident_attention_longer_than_one_workgroup() {
    let file = common::long_context();
    let mut cpu = Cpu::default();
    let allow = std::env::var("QWEN_TEST_ALLOW_SOFTWARE").as_deref() == Ok("1");
    let mut gpu = qwen_rust::backend::Vulkan::new(1, None, allow).unwrap();
    let mut a = Model::new(Gguf::open(&file.0).unwrap(), 80).unwrap();
    let mut b = Model::new(Gguf::open(&file.0).unwrap(), 80).unwrap();
    for i in 0..80 {
        let token = ((i * 7) % 16) as u32;
        if [0, 31, 63, 64, 79].contains(&i) {
            let expected = a.forward(token, &mut cpu).unwrap();
            let actual = b.forward(token, &mut gpu).unwrap();
            assert!(
                qwen_rust::benchmark::parity(&[expected], &[actual], 0.00002, 0.0002)
                    .unwrap()
                    .passed
            );
        } else {
            a.forward_without_logits(token, &mut cpu).unwrap();
            b.forward_without_logits(token, &mut gpu).unwrap();
        }
    }
}
