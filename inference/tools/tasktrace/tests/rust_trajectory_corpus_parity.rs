// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in read/validation and action-sequence parity gate for a trajectory corpus.
//!
//! Set `RIPTIDE_TRAJECTORY_CORPUS` to a JSONL export whose rows contain
//! `source_trajectory`. The ordinary suite returns immediately when the corpus
//! is not available, while campaign preflight can exercise the exact source set.

#[path = "../src/atif_replay.rs"]
mod atif_replay;
#[path = "../src/pacing.rs"]
mod pacing;
#[path = "../src/timebase.rs"]
mod timebase;

use atif_replay::{read_trajectory, validate_supported_actions, AtifTrajectory};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python() -> OsString {
    env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn corpus_paths(manifest: &Path) -> Vec<PathBuf> {
    BufReader::new(File::open(manifest).expect("open trajectory corpus manifest"))
        .lines()
        .enumerate()
        .map(|(offset, line)| {
            let row: Value = serde_json::from_str(&line.expect("read trajectory corpus row"))
                .unwrap_or_else(|error| {
                    panic!("manifest row {} is invalid JSON: {error}", offset + 1)
                });
            PathBuf::from(
                row.get("source_trajectory")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| {
                        panic!("manifest row {} has no source_trajectory", offset + 1)
                    }),
            )
        })
        .collect()
}

fn action_sequence(trajectory: &AtifTrajectory) -> Value {
    json!({
        "schema_version": trajectory.schema_version,
        "agent": trajectory.agent,
        "source_agent_original_format": trajectory.source_agent_original_format,
        "session_id": trajectory.session_id,
        "actions": trajectory.actions.iter().map(|action| json!({
            "index": action.index,
            "step_id": action.step_id,
            "timestamp_sec": action.timestamp_sec,
            "tool": action.tool,
            "source_tool": action.source_tool,
            "source_agent_original_format": action.source_agent_original_format,
            "arguments": action.arguments,
            "observation": action.observation,
            "observation_text": action.observation_text,
            "code_mode": action.code_mode,
            "parallel_group_id": action.parallel_group_id,
        })).collect::<Vec<_>>(),
    })
}

fn python_oracle(manifest: &Path) -> Child {
    let script = r#"
import dataclasses
import json
import pathlib
import sys

from offline_replayer.adapters.atif_replay import read_trajectory, validate_supported_actions

manifest = pathlib.Path(sys.argv[1])
for offset, line in enumerate(manifest.open(encoding="utf-8"), 1):
    source = pathlib.Path(json.loads(line)["source_trajectory"])
    try:
        trajectory = read_trajectory(source)
        validate_supported_actions(trajectory)
        payload = {
            "schema_version": trajectory.schema_version,
            "agent": trajectory.agent,
            "source_agent_original_format": trajectory.source_agent_original_format,
            "session_id": trajectory.session_id,
            "actions": [dataclasses.asdict(action) for action in trajectory.actions],
        }
        print(json.dumps({"offset": offset, "payload": payload}, allow_nan=False, ensure_ascii=False, separators=(",", ":")), flush=True)
    except Exception as error:
        print(json.dumps({"offset": offset, "error": f"{type(error).__name__}: {error}"}, ensure_ascii=False, separators=(",", ":")), flush=True)
"#;
    Command::new(python())
        .arg("-c")
        .arg(script)
        .arg(manifest)
        .current_dir(repository())
        .env("PYTHONPATH", repository())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start Python trajectory oracle")
}

#[test]
fn configured_trajectory_corpus_reads_validates_and_matches_python() {
    let Some(manifest) = env::var_os("RIPTIDE_TRAJECTORY_CORPUS").map(PathBuf::from) else {
        return;
    };
    let paths = corpus_paths(&manifest);
    assert!(!paths.is_empty(), "trajectory corpus is empty");

    let mut oracle = python_oracle(&manifest);
    let stdout = oracle.stdout.take().expect("capture Python oracle stdout");
    let mut oracle_lines = BufReader::new(stdout).lines();
    let mut tool_counts = BTreeMap::<String, usize>::new();
    let mut action_count = 0usize;

    for (offset, source) in paths.iter().enumerate() {
        let trajectory = read_trajectory(source).unwrap_or_else(|error| {
            panic!(
                "Rust failed to read row {} {}: {error}",
                offset + 1,
                source.display()
            )
        });
        validate_supported_actions(&trajectory).unwrap_or_else(|error| {
            panic!(
                "Rust failed to validate row {} {}: {error}",
                offset + 1,
                source.display()
            )
        });
        action_count += trajectory.actions.len();
        for action in &trajectory.actions {
            *tool_counts.entry(action.tool.clone()).or_default() += 1;
        }

        let line = oracle_lines
            .next()
            .unwrap_or_else(|| panic!("Python oracle ended before row {}", offset + 1))
            .expect("read Python oracle row");
        let oracle_row: Value = serde_json::from_str(&line).unwrap_or_else(|error| {
            panic!("Python oracle row {} is invalid JSON: {error}", offset + 1)
        });
        assert_eq!(
            oracle_row["offset"],
            offset + 1,
            "Python oracle row order changed"
        );
        assert!(
            oracle_row.get("error").is_none(),
            "Python failed to read/validate row {} {}: {}",
            offset + 1,
            source.display(),
            oracle_row["error"]
        );
        assert_eq!(
            action_sequence(&trajectory),
            oracle_row["payload"],
            "Python/Rust action-sequence mismatch for row {} {}",
            offset + 1,
            source.display()
        );
    }
    assert!(
        oracle_lines.next().is_none(),
        "Python oracle produced extra rows"
    );

    let output = oracle.wait_with_output().expect("wait for Python oracle");
    assert!(
        output.status.success(),
        "Python oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let unique_paths = paths.iter().collect::<BTreeSet<_>>().len();
    println!(
        "trajectory corpus gate: rows={} unique_paths={} actions={} tool_counts={}",
        paths.len(),
        unique_paths,
        action_count,
        serde_json::to_string(&tool_counts).unwrap()
    );
}
