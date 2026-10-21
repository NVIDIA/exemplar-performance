// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../src/enroot_runtime.rs"]
mod enroot_runtime;
#[path = "../src/network_policy.rs"]
#[allow(dead_code)]
mod network_policy;
#[path = "../src/timebase.rs"]
#[allow(dead_code)]
mod timebase;

use enroot_runtime::{
    build_start_command, cleanup_process_group_after_leader_exit_with, format_enroot_mount,
    normalized_mount_provenance, preflight, run, run_with_cancel, EnrootImageKind,
    EnrootRuntimeConfig, ENROOT_WALL_TIME_METHOD,
};
use network_policy::NetworkMode;
use sha2::{Digest, Sha256};
use std::fs;
use std::os::unix::fs::symlink;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

struct FakeEnroot {
    executable: PathBuf,
    control: PathBuf,
}

impl FakeEnroot {
    fn create(root: &Path) -> Self {
        let control = root.join("fake-enroot-control");
        fs::create_dir(&control).unwrap();
        let executable = root.join("fake-enroot");
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake_enroot_runtime.sh");
        let mut permissions = fs::metadata(&fixture).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fixture, permissions).unwrap();
        symlink(fixture, &executable).unwrap();
        Self {
            executable,
            control,
        }
    }

    fn calls(&self) -> Vec<Vec<String>> {
        fs::read_to_string(self.control.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(|line| line.split('\t').skip(1).map(ToOwned::to_owned).collect())
            .collect()
    }
}

fn base_config(root: &Path, fake: &FakeEnroot, image: impl Into<String>) -> EnrootRuntimeConfig {
    let output = root.join("output");
    fs::create_dir_all(&output).unwrap();
    let entrypoint = root.join("entrypoint.sh");
    fs::write(&entrypoint, "#!/bin/sh\nexit 0\n").unwrap();
    let mut config = EnrootRuntimeConfig::new(image, "printf replay", output, entrypoint);
    config.enroot = fake.executable.to_string_lossy().into_owned();
    config.allocated_cpus = Some(1.0);
    config
}

#[test]
fn translates_bind_mounts_and_builds_requested_affinity_without_claiming_isolation() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    let output = temporary.path().join("profile output");
    fs::create_dir(&output).unwrap();
    let entrypoint = temporary.path().join("entry point.sh");
    fs::write(&entrypoint, "#!/bin/sh\n").unwrap();
    let mut config =
        EnrootRuntimeConfig::new(image.to_string_lossy(), "make -j8", &output, &entrypoint);
    config.enroot = fake.executable.to_string_lossy().into_owned();
    config.cpuset_cpus = Some("0-3,6".to_owned());
    config.user = Some("root".to_owned());
    config.workdir = Some("/work dir/it's".to_owned());
    config.environment = vec!["MODE=test".to_owned()];
    config.mounts = vec!["type=bind,src=/host state,dst=/work dir,readonly".to_owned()];

    assert_eq!(
        format_enroot_mount(&config.mounts[0]).unwrap(),
        "/host\\040state:/work\\040dir:none:x-create=auto,bind,ro"
    );
    assert!(format_enroot_mount("type=volume,src=/a,dst=/b")
        .unwrap_err()
        .to_string()
        .contains("only supports bind mounts"));
    let provenance = normalized_mount_provenance(&config.mounts[0]).unwrap();
    assert_eq!(provenance.destination.as_deref(), Some("/work dir"));
    assert!(provenance.readonly);
    assert!(!serde_json::to_string(&provenance)
        .unwrap()
        .contains("/host state"));

    let command = build_start_command(&config, "extracted-name").unwrap();
    assert_eq!(
        &command[..8],
        [
            config.unshare.as_str(),
            "--user",
            "--map-current-user",
            "--pid",
            "--fork",
            "--kill-child=SIGKILL",
            "--mount-proc",
            "--",
        ]
    );
    let taskset = command.iter().position(|value| value == "taskset").unwrap();
    assert_eq!(&command[taskset..taskset + 3], ["taskset", "-c", "0-3,6"]);
    let enroot = command
        .iter()
        .position(|value| value == &config.enroot)
        .unwrap();
    assert_eq!(
        &command[enroot..enroot + 3],
        [&config.enroot, "start", "--rw"]
    );
    assert!(command.contains(&"--root".to_owned()));
    assert!(
        command.contains(&"/host\\040state:/work\\040dir:none:x-create=auto,bind,ro".to_owned())
    );
    assert!(command.iter().any(|value| value
        .contains("entry\\040point.sh:/tasktrace-profile-entrypoint.sh:none:x-create=auto,bind,ro")));
    assert!(command.iter().any(|value| value
        .contains("profile\\040output:/tasktrace-profile-output:none:x-create=auto,bind,rw")));
    assert!(command.contains(&"TASKTRACE_PROFILE_SKIP_CPU_STATS=1".to_owned()));
    let inner = command.last().unwrap();
    assert!(inner.starts_with("cd '/work dir/it'\"'\"'s' && { set -- "));
    assert!(inner.contains(" TASKTRACE_PROFILE_SKIP_CPU_STATS=1 "));
    assert!(inner.contains(" MODE=test;"));
    assert!(inner.contains("done < /etc/environment"));
    assert!(inner.ends_with(
        "exec /usr/bin/env -i HOME=\"$HOME\" PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \"$@\" /bin/sh /tasktrace-profile-entrypoint.sh; }"
    ));
}

#[test]
fn sqsh_lifecycle_hashes_runs_and_removes_even_after_workload_failure() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    let image_bytes = b"immutable enroot image";
    fs::write(&image, image_bytes).unwrap();
    fs::write(fake.control.join("start-exit"), "17\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let result = run(&config).unwrap();
    assert_eq!(result.exit_code, Some(17));
    assert_eq!(result.image_identity.kind, EnrootImageKind::ImageFile);
    assert!(result.image_identity.immutable);
    assert_eq!(
        result.image_identity.sha256.as_deref(),
        Some(format!("{:x}", Sha256::digest(image_bytes)).as_str())
    );
    assert!(result.cleanup_required);
    assert!(result.cleanup_succeeded);
    assert!(result
        .owned_container_name
        .as_deref()
        .unwrap()
        .starts_with("tasktrace-profile-"));
    assert!(result.create_timing.is_some());
    assert!(result.cleanup_timing.is_some());
    assert!(result.start_timing.end_monotonic_ns > result.start_timing.start_monotonic_ns);
    assert_eq!(result.start_timing.clock_id, "host-boottime");
    assert_eq!(result.start_timing.clock_source, "CLOCK_BOOTTIME");
    assert!(result.start_timing.clock_resolution_ns >= 1);
    assert!(!fake.control.join("created-name").exists());

    let cpu = result.wall_time_cpu_report(&config).unwrap();
    assert_eq!(cpu["method"], ENROOT_WALL_TIME_METHOD);
    assert!(cpu["usage_sec"].is_null());
    assert!(cpu["average_logical_cores_used"].is_null());
    assert_eq!(cpu["cpu_containment_enforced"], false);
    assert_eq!(cpu["allocated_logical_cpus"], 1.0);
    assert_eq!(
        cpu["timing_boundary"],
        "host-enroot-start-lifecycle-inclusive"
    );

    let calls = fake.calls();
    assert_eq!(
        calls
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "create", "start", "remove", "list"]
    );
    let name = &calls[1][2];
    assert_eq!(calls[1][1], "--name");
    assert_eq!(
        calls[2][calls[2].iter().position(|value| value == "--").unwrap() + 1],
        *name
    );
    assert!(!calls[2].contains(&image.to_string_lossy().into_owned()));
    assert_eq!(calls[3], ["remove", "-f", name]);
}

#[test]
fn named_created_container_runs_in_place_without_becoming_owned() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let config = base_config(temporary.path(), &fake, "named-container");

    let result = run(&config).unwrap();
    assert_eq!(
        result.image_identity.kind,
        EnrootImageKind::CreatedContainer
    );
    assert!(!result.image_identity.immutable);
    assert_eq!(
        result.image_identity.container_name.as_deref(),
        Some("named-container")
    );
    assert!(!result.cleanup_required);
    assert!(result.create_timing.is_none());
    assert!(result.cleanup_timing.is_none());
    assert_eq!(result.exit_code, Some(0));
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "list", "start"]
    );
}

#[test]
fn cleanup_failure_fails_closed_instead_of_returning_a_valid_run() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("fail-remove"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let error = run(&config).unwrap_err().to_string();
    assert!(error.contains("cleanup failed"), "{error}");
    assert!(fake.control.join("created-name").exists());
    assert_eq!(
        fake.calls()
            .iter()
            .filter(|call| call.first().map(String::as_str) == Some("remove"))
            .count(),
        1
    );
}

#[test]
fn partial_create_failure_still_attempts_owned_name_cleanup() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("fail-create"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let error = run(&config).unwrap_err().to_string();
    assert!(error.contains("enroot create failed"), "{error}");
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "create", "remove", "list"]
    );
}

fn process_is_gone(pid: u32) -> bool {
    for _ in 0..100 {
        if !PathBuf::from(format!("/proc/{pid}")).exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    false
}

#[test]
fn normal_completion_terminates_background_group_members_before_returning() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("start-background-child"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let result = run(&config).unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.cleanup_succeeded);
    assert_eq!(
        result.process_containment.backend,
        "linux-user-pid-namespace-v1"
    );
    assert!(
        result
            .process_containment
            .namespace_init_exit_kills_remaining_processes
    );
    assert!(result.process_containment.outer_exit_kills_namespace_init);
    assert!(result.start_process_group_cleanup.verified_gone);
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "create", "start", "remove", "list"]
    );
}

#[test]
fn normal_completion_escalates_for_background_group_members_ignoring_sigterm() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("start-stubborn-background-child"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let result = run(&config).unwrap();
    assert_eq!(result.exit_code, Some(0));
    assert!(result.cleanup_succeeded);
    assert_eq!(
        result.process_containment.backend,
        "linux-user-pid-namespace-v1"
    );
    assert!(result.start_process_group_cleanup.verified_gone);
}

#[test]
fn completed_process_group_cleanup_fails_closed_when_absence_cannot_be_proved() {
    let mut signals = Vec::new();
    let error = cleanup_process_group_after_leader_exit_with(
        4242,
        "synthetic Enroot start",
        Duration::ZERO,
        |_| Ok(true),
        |_, signal| {
            signals.push(signal);
            Ok(true)
        },
    )
    .unwrap_err()
    .to_string();

    assert_eq!(signals, [libc::SIGTERM, libc::SIGKILL]);
    assert!(error.contains("cleanup failed"), "{error}");
    assert!(error.contains("process scope is uncertain"), "{error}");
    assert!(error.contains("remained present"), "{error}");
}

#[test]
fn completed_process_group_cleanup_fails_closed_on_final_inspection_error() {
    let mut inspections = 0;
    let mut signals = Vec::new();
    let error = cleanup_process_group_after_leader_exit_with(
        4343,
        "synthetic Enroot create",
        Duration::ZERO,
        |_| {
            inspections += 1;
            if inspections >= 3 {
                Err(std::io::Error::other("synthetic inspection failure").into())
            } else {
                Ok(true)
            }
        },
        |_, signal| {
            signals.push(signal);
            Ok(true)
        },
    )
    .unwrap_err()
    .to_string();

    assert_eq!(signals, [libc::SIGTERM, libc::SIGKILL]);
    assert!(error.contains("cleanup failed"), "{error}");
    assert!(error.contains("post-SIGKILL inspection failed"), "{error}");
    assert!(error.contains("synthetic inspection failure"), "{error}");
}

#[test]
fn cancellation_kills_only_the_start_group_then_verifies_owned_cleanup() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("start-sleep"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );
    let cancel = || fake.control.join("start-entered").exists();

    let error = run_with_cancel(&config, Some(&cancel))
        .unwrap_err()
        .to_string();
    assert!(error.contains("start cancelled"), "{error}");
    let evidence: serde_json::Value = serde_json::from_slice(
        &fs::read(config.output_dir.join("enroot_runtime_failure.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(evidence["status"], "cancelled");
    assert_eq!(evidence["phase"], "start");
    assert_eq!(evidence["profiling_valid"], false);
    assert_eq!(evidence["cleanup_required"], true);
    assert_eq!(evidence["cleanup_succeeded"], true);
    assert_eq!(evidence["cleanup"]["verified_absent"], true);
    assert_eq!(evidence["host_process_group"]["owned_by_controller"], true);
    assert_eq!(
        evidence["process_containment"]["backend"],
        "linux-user-pid-namespace-v1"
    );
    assert_eq!(evidence["host_process_group"]["verified_gone"], true);
    assert!(!fake.control.join("created-name").exists());
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "create", "start", "remove", "list"]
    );
}

#[test]
fn cancellation_waits_for_and_kills_group_members_after_the_leader_exits() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("start-stubborn-child"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );
    let cancel = || fake.control.join("start-entered").exists();

    let error = run_with_cancel(&config, Some(&cancel))
        .unwrap_err()
        .to_string();
    assert!(error.contains("start cancelled"), "{error}");
    let evidence: serde_json::Value = serde_json::from_slice(
        &fs::read(config.output_dir.join("enroot_runtime_failure.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(evidence["host_process_group"]["sigterm_sent"], true);
    assert_eq!(evidence["host_process_group"]["verified_gone"], true);
    assert_eq!(
        evidence["process_containment"]["backend"],
        "linux-user-pid-namespace-v1"
    );
    assert_eq!(evidence["cleanup"]["verified_absent"], true);
}

#[test]
fn cancellation_interrupts_create_and_verifies_the_unique_name_absent() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    fs::write(fake.control.join("create-sleep"), "1\n").unwrap();
    let config = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );
    let cancel = || fake.control.join("create-entered").exists();

    let error = run_with_cancel(&config, Some(&cancel))
        .unwrap_err()
        .to_string();
    assert!(error.contains("create cancelled"), "{error}");
    let evidence: serde_json::Value = serde_json::from_slice(
        &fs::read(config.output_dir.join("enroot_runtime_failure.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(evidence["phase"], "create");
    assert_eq!(evidence["cleanup_required"], true);
    assert_eq!(evidence["cleanup"]["verified_absent"], true);
    let pid = fs::read_to_string(fake.control.join("create-child-pid"))
        .unwrap()
        .trim()
        .parse::<u32>()
        .unwrap();
    assert!(
        process_is_gone(pid),
        "cancelled create child {pid} survived"
    );
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "create", "remove", "list"]
    );
}

#[test]
fn cancellation_of_named_container_never_removes_caller_owned_state() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    fs::write(fake.control.join("start-sleep"), "1\n").unwrap();
    let config = base_config(temporary.path(), &fake, "named-container");
    let cancel = || fake.control.join("start-entered").exists();

    let error = run_with_cancel(&config, Some(&cancel))
        .unwrap_err()
        .to_string();
    assert!(error.contains("start cancelled"), "{error}");
    let evidence: serde_json::Value = serde_json::from_slice(
        &fs::read(config.output_dir.join("enroot_runtime_failure.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(evidence["cleanup_required"], false);
    assert_eq!(evidence["caller_owned_named_container_removed"], false);
    assert_eq!(
        fake.calls()
            .iter()
            .map(|call| call[0].as_str())
            .collect::<Vec<_>>(),
        ["version", "list", "start"]
    );
}

#[test]
fn preflight_records_file_digest_without_creating_output() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"identity").unwrap();
    let entrypoint = temporary.path().join("entrypoint.sh");
    fs::write(&entrypoint, "#!/bin/sh\n").unwrap();
    let output = temporary.path().join("not-created");
    let mut config =
        EnrootRuntimeConfig::new(image.to_string_lossy(), "true", &output, &entrypoint);
    config.enroot = fake.executable.to_string_lossy().into_owned();
    config.allocated_cpus = Some(2.0);

    let report = preflight(&config).unwrap();
    assert!(report.ready);
    assert!(report.image_identity.immutable);
    assert_eq!(
        report.image_identity.sha256.as_deref(),
        Some(format!("{:x}", Sha256::digest(b"identity")).as_str())
    );
    assert!(report
        .limitations
        .iter()
        .any(|limitation| limitation.contains("wall-time-only")));
    assert!(!output.exists());
}

#[test]
fn validation_rejects_every_docker_or_profiler_only_contract() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    let base = base_config(
        temporary.path(),
        &fake,
        image.to_string_lossy().into_owned(),
    );

    let mut cases: Vec<(EnrootRuntimeConfig, &str)> = Vec::new();
    let mut config = base.clone();
    config.user = Some("1000".to_owned());
    cases.push((config, "only supports --user root"));
    let mut config = base.clone();
    config.cpu_limit = Some(1.0);
    cases.push((config, "--cpu-limit is Docker-only"));
    let mut config = base.clone();
    config.scheduler_concurrency = true;
    cases.push((config, "--scheduler-concurrency is currently Docker-only"));
    let mut config = base.clone();
    config.proc_stat_comparison = true;
    cases.push((config, "--proc-stat-comparison is currently Docker-only"));
    let mut config = base.clone();
    config.memory_profile = true;
    cases.push((config, "--memory-profile is currently Docker-only"));
    let mut config = base.clone();
    config.memory_limit_bytes = Some(1024);
    cases.push((config, "--memory-limit-bytes is currently Docker-only"));
    let mut config = base.clone();
    config.memory_swap_total_limit_bytes = Some(1024);
    cases.push((
        config,
        "--memory-swap-total-limit-bytes is currently Docker-only",
    ));
    let mut config = base.clone();
    config.profile_sets = vec!["cpu-average".to_owned()];
    cases.push((
        config,
        "--profile-set currently requires the Docker runtime",
    ));
    let mut config = base.clone();
    config.coordinated_baseline = true;
    cases.push((
        config,
        "--coordinated-baseline currently requires the Docker runtime",
    ));
    let mut config = base.clone();
    config.container = Some("docker-container".to_owned());
    cases.push((config, "enroot mode does not support --container"));
    let mut config = base.clone();
    config.network_mode = NetworkMode::NoNetwork;
    cases.push((
        config,
        "requires a controller-private network evidence path",
    ));
    let mut config = base.clone();
    config.network_evidence_path = Some(temporary.path().join("public-network.json"));
    cases.push((config, "only valid for no-network mode"));
    let mut config = base.clone();
    config.network_mode = NetworkMode::Allowlist;
    cases.push((config, "does not support allowlist network policy"));
    let mut config = base;
    config.strace_mode = "execve".to_owned();
    cases.push((config, "strace tracing requires Docker"));

    for (config, expected) in cases {
        let error = config.validate().unwrap_err().to_string();
        assert!(
            error.contains(expected),
            "expected {expected:?} in {error:?}"
        );
    }
}
