// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![recursion_limit = "256"]

//! Public `tasktrace` command dispatcher.
//!
//! Every public command is implemented in this crate.  Python modules retained
//! in the source tree are compatibility or differential-test oracles; the
//! installed command never imports or starts them.

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::Path;

mod agentscry_comparison;
#[allow(dead_code)] // Includes testable profile/preflight helpers beyond CLI dispatch.
mod atif_profile;
mod atif_replay;
mod ceiling_hitters;
#[allow(dead_code)] // Includes current-container inspection and validation helpers.
mod current_profile;
#[allow(dead_code)] // Captures content-bound Enroot runtime/config/state identity for ordered runs.
mod enroot_provenance;
#[allow(dead_code)] // Includes Enroot lifecycle and evidence helpers.
mod enroot_runtime;
#[allow(dead_code)] // Includes caller-owned-container inspection helpers.
mod existing_container;
#[allow(dead_code)] // Includes native PMU collector parsing and provenance helpers.
mod hardware_perf;
#[allow(dead_code)] // Includes native Docker network-policy lifecycle helpers.
mod network_enforcement;
#[allow(dead_code)] // Shared native network-policy model.
mod network_policy;
#[allow(dead_code)] // Executed only inside the native allowlist sidecar.
mod network_proxy;
#[allow(dead_code)] // Includes sampled host-pressure comparison helpers.
mod pacing;
mod proc_stat;
mod process_bpf_reports;
#[path = "profile_container/process_exec.rs"]
mod process_exec;
#[allow(dead_code)] // Includes the full native profile surface and evidence builders.
mod profile_container;
mod profile_pipeline;
mod report_sets;
mod resource_timeline;
#[allow(dead_code)] // Includes native TaskTrace-trace profile/preflight helpers.
mod trace_profile;
mod trace_replay;
mod run_execute;
mod run_host;
#[allow(dead_code)] // Retains stable schema aliases and inspection helpers.
mod run_spec;
mod scheduler_batch;
#[allow(dead_code)]
mod scheduler_bcc;
mod scheduler_perf;
mod task_catalog;
mod tb2_profiles;
mod timebase;
mod transcript_profile;
mod transcript_replay;
mod unified_view;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy)]
struct CommandSpec {
    group: &'static str,
    name: &'static str,
    summary: &'static str,
}

const TOP_LEVEL: CommandSpec = CommandSpec {
    group: "",
    name: "run",
    summary: "Run ATIF tasks from one strict YAML file.",
};

const COMMANDS: &[CommandSpec] = &[
    CommandSpec {
        group: "profile",
        name: "container",
        summary: "Measure one container command.",
    },
    CommandSpec {
        group: "profile",
        name: "atif",
        summary: "Measure one ATIF trajectory replay.",
    },
    CommandSpec {
        group: "profile",
        name: "trace",
        summary: "Measure one TaskTrace trace replay.",
    },
    CommandSpec {
        group: "profile",
        name: "transcript",
        summary: "Measure one transcript trial replay.",
    },
    CommandSpec {
        group: "replay",
        name: "atif",
        summary: "Replay an ATIF trajectory.",
    },
    CommandSpec {
        group: "replay",
        name: "trace",
        summary: "Replay a TaskTrace trace.",
    },
    CommandSpec {
        group: "replay",
        name: "transcript",
        summary: "Replay a transcript trial.",
    },
    CommandSpec {
        group: "report",
        name: "render-batch",
        summary: "Render report summaries.",
    },
    CommandSpec {
        group: "report",
        name: "render-set",
        summary: "Render declared report sets.",
    },
    CommandSpec {
        group: "report",
        name: "analyze-tb2",
        summary: "Analyze Terminal-Bench reports.",
    },
    CommandSpec {
        group: "report",
        name: "resource-timeline",
        summary: "Render aligned cgroup CPU and memory over time.",
    },
    CommandSpec {
        group: "report",
        name: "unified-view",
        summary: "Render every profiler stream on one synchronized view.",
    },
    CommandSpec {
        group: "report",
        name: "ceiling-hitters",
        summary: "Compare ceiling hitters.",
    },
    CommandSpec {
        group: "report",
        name: "task-catalog",
        summary: "Export task catalog JSON.",
    },
    CommandSpec {
        group: "report",
        name: "process-inventory-bpf",
        summary: "Build a loss-audited BPF process inventory proof.",
    },
    CommandSpec {
        group: "report",
        name: "process-inventory-bpf-repeatability",
        summary: "Compare content-bound BPF process inventory proofs.",
    },
    CommandSpec {
        group: "report",
        name: "compare-agentscry",
        summary: "Compare container CPU accounting with AgentScry.",
    },
];

fn program_name() -> String {
    env::args_os()
        .next()
        .as_deref()
        .and_then(|argument| Path::new(argument).file_name())
        .and_then(OsStr::to_str)
        .filter(|name| matches!(*name, "tasktrace" | "tasktrace.exe"))
        .unwrap_or("tasktrace")
        .to_owned()
}

fn print_help(program: &str, group: Option<&str>) {
    if let Some(group) = group {
        println!("usage: {program} {group} <command> [arguments]");
        println!();
        println!("commands:");
        let width = COMMANDS
            .iter()
            .filter(|command| command.group == group)
            .map(|command| command.name.len())
            .max()
            .unwrap_or(0)
            .max(19);
        for command in COMMANDS.iter().filter(|command| command.group == group) {
            println!(
                "  {:width$} {}",
                command.name,
                command.summary,
                width = width
            );
        }
        return;
    }

    println!("Replay recorded tool-side work and measure CPU and memory demand.");
    println!();
    println!("usage: {program} run FILE.yaml");
    println!("       {program} <group> <command> [arguments]");
    println!();
    println!("commands:");
    println!("  {:19} {}", TOP_LEVEL.name, TOP_LEVEL.summary);
    println!();
    println!("groups:");
    println!("  profile");
    println!("  replay");
    println!("  report");
    println!();
    println!("examples:");
    println!("  {program} replay transcript <transcript.jsonl> --workspace <dir>");
    println!("  {program} profile atif <trajectory.json> --image <image> --output-dir <dir>");
    println!();
    println!("Run `{program} <group> --help` for commands.");
}

fn dispatch_native(spec: CommandSpec, arguments: &[OsString], public_program: &str) -> i32 {
    match (spec.group, spec.name) {
        ("profile", "container") => profile_container::main(arguments, public_program),
        ("profile", "atif") => atif_profile::main(arguments, public_program),
        ("profile", "trace") => trace_profile::main(arguments, public_program),
        ("profile", "transcript") => transcript_profile::main(arguments, public_program),
        ("replay", "atif") => atif_replay::main(arguments, public_program),
        ("replay", "trace") => trace_replay::main(arguments, public_program),
        ("replay", "transcript") => transcript_replay::main(arguments, public_program),
        ("report", "ceiling-hitters") => ceiling_hitters::main(arguments, public_program),
        ("report", "task-catalog") => task_catalog::main(arguments, public_program),
        ("report", "analyze-tb2") => tb2_profiles::main(arguments, public_program),
        ("report", "render-batch") => scheduler_batch::main(arguments, public_program),
        ("report", "render-set") => report_sets::main(arguments, public_program),
        ("report", "resource-timeline") => resource_timeline::main(arguments, public_program),
        ("report", "unified-view") => unified_view::main(arguments, public_program),
        ("report", "process-inventory-bpf") => {
            process_bpf_reports::inventory_main(arguments, public_program)
        }
        ("report", "process-inventory-bpf-repeatability") => {
            process_bpf_reports::repeatability_main(arguments, public_program)
        }
        ("report", "compare-agentscry") => agentscry_comparison::main(arguments, public_program),
        _ => unreachable!("native command table and dispatcher diverged"),
    }
}

fn run(arguments: Vec<OsString>) -> i32 {
    let program = program_name();
    let mut arguments = arguments.into_iter();
    let Some(first) = arguments.next() else {
        print_help(&program, None);
        return 0;
    };
    if matches!(first.to_str(), Some("-h" | "--help")) {
        print_help(&program, None);
        return 0;
    }
    if matches!(first.to_str(), Some("-V" | "--version")) {
        println!("{VERSION}");
        return 0;
    }

    let first = first.to_string_lossy();
    let remaining: Vec<OsString> = arguments.collect();
    if first == "__network-proxy" {
        return network_proxy::internal_main(&remaining);
    }
    if first == enroot_runtime::enroot_network::INTERNAL_COMMAND {
        return enroot_runtime::enroot_network::internal_main(&remaining);
    }
    if first == TOP_LEVEL.name {
        let public_program = format!("{program} {}", TOP_LEVEL.name);
        return run_execute::main(&remaining, &public_program);
    }

    if !matches!(first.as_ref(), "profile" | "replay" | "report") {
        eprintln!("unknown command group: {first}");
        print_help(&program, None);
        return 2;
    }
    if remaining.is_empty() || matches!(remaining[0].to_str(), Some("-h" | "--help")) {
        print_help(&program, Some(&first));
        return 0;
    }

    let name = remaining[0].to_string_lossy();
    let leaf_arguments = &remaining[1..];
    let Some(spec) = COMMANDS
        .iter()
        .copied()
        .find(|command| command.group == first && command.name == name)
    else {
        eprintln!("unknown {first} command: {name}");
        print_help(&program, Some(&first));
        return 2;
    };
    let public_program = format!("{program} {} {}", spec.group, spec.name);
    dispatch_native(spec, leaf_arguments, &public_program)
}

fn main() {
    std::process::exit(run(env::args_os().skip(1).collect()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_unique_public_leaves() {
        for (index, left) in COMMANDS.iter().enumerate() {
            assert!(!COMMANDS[index + 1..]
                .iter()
                .any(|right| left.group == right.group && left.name == right.name));
        }
    }

    #[test]
    fn all_public_leaves_are_dispatched_natively() {
        let native: Vec<_> = COMMANDS
            .iter()
            .map(|command| (command.group, command.name))
            .collect();
        assert_eq!(
            native,
            [
                ("profile", "container"),
                ("profile", "atif"),
                ("profile", "trace"),
                ("profile", "transcript"),
                ("replay", "atif"),
                ("replay", "trace"),
                ("replay", "transcript"),
                ("report", "render-batch"),
                ("report", "render-set"),
                ("report", "analyze-tb2"),
                ("report", "resource-timeline"),
                ("report", "unified-view"),
                ("report", "ceiling-hitters"),
                ("report", "task-catalog"),
                ("report", "process-inventory-bpf"),
                ("report", "process-inventory-bpf-repeatability"),
                ("report", "compare-agentscry"),
            ]
        );
    }
}
