// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Differential gate for the native unified evidence renderer.

#[allow(dead_code)]
#[path = "../src/unified_view.rs"]
mod unified_view;

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn python() -> OsString {
    env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn python_path() -> OsString {
    let mut paths = vec![repository()];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).unwrap()
}

fn generate_bundle(root: &std::path::Path) -> PathBuf {
    let bundle = root.join("evidence_bundle.json");
    let fixture = Command::new(python())
        .args([
            "-c",
            concat!(
                "import sys; from pathlib import Path; ",
                "from offline_replayer.tests.test_unified_view import _bundle; ",
                "from offline_replayer.measurement.evidence import write_bundle; ",
                "root=Path(sys.argv[1]); write_bundle(_bundle(root), Path(sys.argv[2]))",
            ),
        ])
        .arg(root)
        .arg(&bundle)
        .current_dir(repository())
        .env("PYTHONPATH", python_path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap();
    assert!(
        fixture.status.success(),
        "{}",
        String::from_utf8_lossy(&fixture.stderr)
    );
    bundle
}

#[test]
fn all_unified_exports_match_python_byte_for_byte() {
    let root = TempDir::new().unwrap();
    let bundle = generate_bundle(root.path());
    let mut bundle_value: serde_json::Value =
        serde_json::from_slice(&fs::read(&bundle).unwrap()).unwrap();
    bundle_value["provenance"]["float_serialization_edges"] =
        serde_json::json!([1e-7_f64, 1e-6_f64, 1e-5_f64, 1e15_f64, 1e16_f64, 1e20_f64, -0.0_f64,]);
    bundle_value["provenance"]["html_embedding_edge"] =
        serde_json::json!("</script>& \"quoted\" 'snowman: ☃'");
    let scalar = bundle_value["streams"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|stream| stream["stream_id"] == "memory-peak")
        .unwrap();
    scalar["records"][0]["value"] = serde_json::json!(999_999_999.9_f64);
    let labeled_track = bundle_value["streams"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|stream| stream["stream_id"] == "phase-events")
        .unwrap();
    labeled_track["display"]["label"] = serde_json::json!("phase \"events\" <bounded>");
    labeled_track["display"]["group"] = serde_json::json!("Fixture's & profiler");
    fs::write(&bundle, serde_json::to_vec(&bundle_value).unwrap()).unwrap();

    let python_output = root.path().join("python");
    let oracle = Command::new(python())
        .args(["-m", "offline_replayer.reporting.unified_view"])
        .arg(&bundle)
        .arg("--output-dir")
        .arg(&python_output)
        .current_dir(repository())
        .env("PYTHONPATH", python_path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap();
    assert!(
        oracle.status.success(),
        "{}",
        String::from_utf8_lossy(&oracle.stderr)
    );

    let rust_output = root.path().join("rust");
    let native = Command::new(binary())
        .args(["report", "unified-view"])
        .arg(&bundle)
        .arg("--output-dir")
        .arg(&rust_output)
        .current_dir(repository())
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            root.path().join("fallback-must-not-run"),
        )
        .output()
        .unwrap();
    assert!(
        native.status.success(),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );
    for name in [
        "view_model.json",
        "unified_profile.html",
        "unified_profile.svg",
        "unified_profile.animated.svg",
    ] {
        let actual = fs::read(rust_output.join(name)).unwrap();
        let expected = fs::read(python_output.join(name)).unwrap();
        if actual != expected {
            let offset = actual
                .iter()
                .zip(&expected)
                .position(|(left, right)| left != right)
                .unwrap_or(actual.len().min(expected.len()));
            let start = offset.saturating_sub(80);
            let actual_end = (offset + 160).min(actual.len());
            let expected_end = (offset + 160).min(expected.len());
            panic!(
                "{name} differs at {offset} (lengths {} vs {})\nactual: {:?}\nexpected: {:?}",
                actual.len(),
                expected.len(),
                String::from_utf8_lossy(&actual[start..actual_end]),
                String::from_utf8_lossy(&expected[start..expected_end]),
            );
        }
    }
    let native_stdout = String::from_utf8(native.stdout)
        .unwrap()
        .replace(rust_output.to_string_lossy().as_ref(), "<OUTPUT>");
    let python_stdout = String::from_utf8(oracle.stdout)
        .unwrap()
        .replace(python_output.to_string_lossy().as_ref(), "<OUTPUT>");
    assert_eq!(native_stdout, python_stdout);
}

#[test]
fn malformed_or_drifted_evidence_fails_before_artifact_publication() {
    let root = TempDir::new().unwrap();
    let bundle_path = generate_bundle(root.path());
    let pristine: serde_json::Value =
        serde_json::from_slice(&fs::read(bundle_path).unwrap()).unwrap();

    let mut unknown_field = pristine.clone();
    unknown_field["streams"][0]["fabricated_claim"] = serde_json::json!(true);
    let output = root.path().join("unknown-field");
    let error = unified_view::write_unified_view_artifacts(&unknown_field, &output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("stream has unknown fields"), "{error}");
    assert_eq!(fs::read_dir(output).unwrap().count(), 0);

    let mut outside_window = pristine.clone();
    let stream = outside_window["streams"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|stream| stream["stream_id"] == "memory-current")
        .unwrap();
    stream["records"][0]["canonical_timestamp_ns"] = serde_json::json!(12_000_000_000_i64);
    let error = unified_view::assemble_view_model(&outside_window)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("contradicts its clock mapping")
            || error.contains("falls outside the run window"),
        "{error}"
    );

    let mut incomplete_mapping = pristine;
    incomplete_mapping["clock_mappings"][0]["status"] = serde_json::json!("valid");
    incomplete_mapping["clock_mappings"][0]["reasons"] = serde_json::json!([]);
    let error = unified_view::assemble_view_model(&incomplete_mapping)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("requires a complete affine transform"),
        "{error}"
    );
}

#[test]
fn refuses_overwrite_before_mutating_existing_artifacts() {
    let root = TempDir::new().unwrap();
    let output = root.path().join("output");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("unified_profile.svg"), "sentinel").unwrap();
    let error = unified_view::write_unified_view_artifacts(&serde_json::json!({}), &output)
        .unwrap_err()
        .to_string();
    assert!(error.contains("refusing to overwrite unified view artifacts"));
    assert_eq!(
        fs::read_to_string(output.join("unified_profile.svg")).unwrap(),
        "sentinel"
    );
}
