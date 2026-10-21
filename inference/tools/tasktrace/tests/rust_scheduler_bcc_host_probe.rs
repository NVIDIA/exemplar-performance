// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only host capability probe for the optional native libbcc backend.

#[path = "../src/scheduler_bcc.rs"]
mod scheduler_bcc;
#[path = "../src/scheduler_perf.rs"]
mod scheduler_perf;

use scheduler_bcc::BccSchedulerTracer;

#[test]
fn host_bcc_preflight_reports_capability_or_an_exact_blocker() {
    match BccSchedulerTracer::inspect_library(None) {
        Ok(report) => {
            assert_eq!(report["ready"], true);
            eprintln!("libbcc runtime probe: {report}");
        }
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains("unable to load libbcc")
                    || message.contains("missing required symbol"),
                "unexpected libbcc probe failure: {message}"
            );
            eprintln!("libbcc runtime unavailable: {message}");
            return;
        }
    }

    match BccSchedulerTracer::preflight(None, None) {
        Ok(report) => {
            assert_eq!(report["ready"], true);
            assert_eq!(report["kernel_program_load"], "passed");
            eprintln!("BCC scheduler host preflight: {report}");
        }
        Err(error) => {
            let message = error.to_string();
            let classified = [
                "requires root privileges",
                "requires a cgroup v2 hierarchy",
                "unable to read online logical CPUs",
                "unable to read kernel symbols",
                "unable to read traceable kernel functions",
                "finish_task_switch kprobe symbol is unavailable",
                "unable to load libbcc",
                "missing required symbol",
                "unable to compile scheduler BPF program",
                "unable to load scheduler BPF program",
            ]
            .iter()
            .any(|needle| message.contains(needle));
            assert!(classified, "unclassified BCC preflight failure: {message}");
            eprintln!("BCC scheduler host preflight blocked: {message}");
        }
    }
}
