// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Linux monotonic-clock identity retained with replay evidence.

use serde_json::{json, Value};
use std::fs;
use std::path::Path;

fn clock_ns(clock_id: libc::clockid_t, clock_name: &str) -> u64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(clock_id, &mut value) };
    assert_eq!(result, 0, "clock_gettime({clock_name}) failed");
    assert!(
        value.tv_sec >= 0 && (0..1_000_000_000).contains(&value.tv_nsec),
        "{clock_name} reported an invalid timespec"
    );
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
}

pub fn monotonic_ns() -> u64 {
    // CLOCK_MONOTONIC is guaranteed on the supported Linux runtime. A failure
    // means the measurement cannot retain a valid clock boundary.
    clock_ns(libc::CLOCK_MONOTONIC, "CLOCK_MONOTONIC")
}

/// Read Linux CLOCK_REALTIME as Unix-epoch nanoseconds.
pub fn realtime_ns() -> u64 {
    clock_ns(libc::CLOCK_REALTIME, "CLOCK_REALTIME")
}

/// Correlate a monotonic read bracketed by two realtime reads.
///
/// A negative uncertainty is deliberately retained when CLOCK_REALTIME moves
/// backward: callers must fail closed instead of treating a reordered bracket
/// as a valid accuracy bound. Deltas outside JSON's signed-integer range use
/// `i64::MIN`, which is likewise an explicitly invalid bound.
pub fn realtime_midpoint_and_uncertainty(before_ns: u64, after_ns: u64) -> (u64, i64) {
    if after_ns >= before_ns {
        let width = after_ns - before_ns;
        (
            before_ns + width / 2,
            i64::try_from(width).unwrap_or(i64::MIN),
        )
    } else {
        let width = before_ns - after_ns;
        let midpoint = before_ns - (width / 2 + width % 2);
        let uncertainty = i64::try_from(width)
            .ok()
            .and_then(i64::checked_neg)
            .unwrap_or(i64::MIN);
        (midpoint, uncertainty)
    }
}

pub fn monotonic_resolution_ns() -> u64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_getres(libc::CLOCK_MONOTONIC, &mut value) };
    assert_eq!(result, 0, "clock_getres(CLOCK_MONOTONIC) failed");
    assert!(
        value.tv_sec >= 0 && value.tv_nsec >= 0,
        "CLOCK_MONOTONIC reported a negative resolution"
    );
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
        .max(1)
}

fn time_namespace_offsets(path: &Path) -> Option<(i64, Option<i64>)> {
    let text = fs::read_to_string(path).ok()?;
    let mut monotonic = None;
    let mut boottime = None;
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() != 3 || !matches!(fields[0], "monotonic" | "boottime") {
            return None;
        }
        let seconds: i64 = fields[1].parse().ok()?;
        let nanoseconds: i64 = fields[2].parse().ok()?;
        if !(0..1_000_000_000).contains(&nanoseconds) {
            return None;
        }
        let offset = seconds
            .checked_mul(1_000_000_000)?
            .checked_add(nanoseconds)?;
        match fields[0] {
            "monotonic" => monotonic = Some(offset),
            "boottime" => boottime = Some(offset),
            _ => unreachable!(),
        }
    }
    monotonic.map(|value| (value, boottime))
}

pub fn clock_reference() -> Value {
    let time_namespace = fs::read_link("/proc/self/ns/time")
        .ok()
        .map(|path| path.to_string_lossy().into_owned());
    let boot_id = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .ok()
        .map(|value| value.trim().to_owned());
    let monotonic_offset = time_namespace_offsets(Path::new("/proc/self/timens_offsets"))
        .map(|(monotonic, _)| monotonic);
    json!({
        "clock_id": "CLOCK_MONOTONIC",
        "timestamp_unit": "nanoseconds",
        "time_namespace": time_namespace,
        "time_namespace_monotonic_offset_ns": monotonic_offset,
        "boot_id": boot_id,
        "storage_precision_ns": 1,
        "accuracy_claim": Value::Null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_signed_namespace_offsets() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "monotonic -2 500000000\nboottime 1 4").unwrap();
        assert_eq!(
            time_namespace_offsets(file.path()),
            Some((-1_500_000_000, Some(1_000_000_004)))
        );
    }

    #[test]
    fn reference_names_the_monotonic_clock() {
        let reference = clock_reference();
        assert_eq!(reference["clock_id"], "CLOCK_MONOTONIC");
        assert_eq!(reference["timestamp_unit"], "nanoseconds");
        assert_eq!(reference["storage_precision_ns"], 1);
        assert!(monotonic_resolution_ns() >= 1);
    }

    #[test]
    fn realtime_clock_is_a_positive_epoch_timestamp() {
        assert!(realtime_ns() > 0);
    }

    #[test]
    fn realtime_correlation_retains_backward_clock_step() {
        assert_eq!(realtime_midpoint_and_uncertainty(10, 16), (13, 6));
        assert_eq!(realtime_midpoint_and_uncertainty(16, 10), (13, -6));
        assert_eq!(realtime_midpoint_and_uncertainty(16, 11), (13, -5));
    }

    #[test]
    fn realtime_correlation_midpoint_is_safe_at_u64_limits() {
        assert_eq!(
            realtime_midpoint_and_uncertainty(u64::MAX - 1, u64::MAX),
            (u64::MAX - 1, 1)
        );
        assert_eq!(
            realtime_midpoint_and_uncertainty(0, u64::MAX),
            (u64::MAX / 2, i64::MIN)
        );
        assert_eq!(
            realtime_midpoint_and_uncertainty(u64::MAX, 0),
            (u64::MAX / 2, i64::MIN)
        );
    }
}
