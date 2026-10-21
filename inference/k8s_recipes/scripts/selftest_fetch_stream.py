#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""selftest_fetch_stream.py — the per-entry stream in `fetch_results.sh`, compressed and uncompressed.

The bash under test is sliced out of fetch_results.sh, not copied. A stub `llmb::kc` runs the remote command
locally, so the tar flags and probe script the script builds are the ones exercised. The checks cover the
compression decision (C), equivalence of the two paths (E), integrity on a failed or truncated stream (P),
the retry budget (R), the mounter pod's lifecycle (L) and the run_meta.json cluster stamp (S).

Not covered: properties of the real mounter image, such as whether it ships gzip or how its tar reports a
compressor failure. C1/C2 cover the script's response to them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH_SH = ROOT / "scripts" / "fetch_results.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── the region under test, sliced out of the shipping script ────────────────────────────────────────────
def _extract_streamer() -> str:
    """resolve_stream_compression, stream_entry, the per-entry retry loop and the `exit 1` it feeds, verbatim.
    The exit is included so the script, not the harness, decides the exit code."""
    src = FETCH_SH.read_text()
    start = src.index("resolve_stream_compression() {")
    end = src.index("# ── cluster identity")
    return src[start:end]


def _extract_mounter() -> str:
    """MOUNTER_DEADLINE_S, mounter_manifest, spawn_mounter and ensure_mounter, verbatim."""
    src = FETCH_SH.read_text()
    start = src.index("# ── Mounter pod")
    end = src.index('log "Spawning transient mounter pod')
    return src[start:end]


def _extract_cluster_stamp() -> str:
    """The python that stamps the cluster into run_meta.json."""
    src = FETCH_SH.read_text()
    block = src[src.index("# ── cluster identity") :]
    start = block.index("<< 'PY'\n") + len("<< 'PY'\n")
    return block[start : block.index("\nPY\n", start)]


# Stub cluster: `llmb::kc exec POD -c mounter -- sh -c <cmd>` runs <cmd> locally. Faults are injected by env
# var. Probe calls and stream attempts are counted separately. The pod's state lives in $STUB_POD as the
# `{.status.phase}/{.metadata.deletionTimestamp}` that ensure_mounter reads; empty means no pod. Pod verbs are
# appended to $STUB_LOG, and the manifest of the last `create` or `apply` is kept in $STUB_MANIFEST.
STUB = r"""
set -eu
set -o pipefail
log() { printf '%s\n' "$*" >&2; }
err() { printf '%s\n' "$*" >&2; }
llmb::heal_auth() { :; }
sleep() { :; }            # skip the retry backoff
write_receipt() { :; }    # covered by selftest_fetch_receipt.py

stub_pod() {
  local verb="$1" state; state="$(cat "$STUB_POD")"
  case "$verb" in
    get) printf '%s' "$state" ;;
    delete)
      case " $* " in *" --wait=true "*) echo "delete-wait" ;; *) echo "delete-nowait" ;; esac >> "$STUB_LOG"
      # STUB_DELETE: gone (default) | stuck (times out, pod stays) | noop (returns 0, pod stays).
      case "${STUB_DELETE:-gone}" in
        gone) : > "$STUB_POD" ;;
        stuck) echo "error: timed out waiting for the condition" >&2; return 1 ;;
        noop) ;;
      esac ;;
    create)
      cat > "$STUB_MANIFEST"
      if [ -n "$state" ]; then
        echo "create-refused" >> "$STUB_LOG"
        echo "Error from server (AlreadyExists): pods \"$POD\" already exists" >&2
        return 1
      fi
      echo "create" >> "$STUB_LOG"
      printf 'Running/' > "$STUB_POD" ;;
    apply) echo "apply" >> "$STUB_LOG"; cat > "$STUB_MANIFEST"; printf 'Running/' > "$STUB_POD" ;;
    wait) [ "$state" = Running/ ] ;;
    describe) echo "stub describe" ;;
  esac
}

llmb::kc() {
  case "$1" in get | delete | create | apply | wait | describe) stub_pod "$@"; return ;; esac
  # Last argument is the remote sh -c command string.
  local cmd; eval 'cmd=${'"$#"'}'
  if [ "$(cat "$STUB_POD")" != Running/ ]; then
    echo "error: pod $POD is not running" >&2
    return 1
  fi
  case "$cmd" in
    *LLMBZ_*)   # the compression probe
      printf '%s' "$(( $(cat "$STUB_PROBES") + 1 ))" > "$STUB_PROBES"
      case "${STUB_PROBE:-run}" in
        # The real probe script, with every filesystem-writing command on PATH failing.
        run)      PATH="$STUB_NOWRITE:$PATH" sh -c "$cmd" ;;
        nogzip)   echo LLMBZ_NOGZIP ;;
        notarz)   echo LLMBZ_NOTARZ ;;
        execfail) return 1 ;;   # the exec itself fails
      esac
      return 0 ;;
  esac
  # A stream attempt.
  local n; n=$(( $(cat "$STUB_COUNTER") + 1 )); printf '%s' "$n" > "$STUB_COUNTER"
  if [ "$n" = "${STUB_POD_LOST_DURING:-}" ]; then
    # The pod is removed while this stream runs; kubectl exits 137.
    : > "$STUB_POD"
    echo "command terminated with exit code 137" >&2
    return 137
  fi
  if [ "$n" = "${STUB_POD_ENDS_AFTER:-}" ]; then
    # The pod reaches its deadline once this stream has finished.
    sh -c "$cmd"
    printf 'Failed/' > "$STUB_POD"
    return 0
  fi
  case " ${STUB_FAIL_ENTRY:-} " in
    *" $(printf '%s' "$cmd" | sed "s/.*-- '//; s/'.*//") "*)
      echo "stub: entry permanently unavailable" >&2; return 1 ;;
  esac
  if [ -n "${STUB_FAIL_TIMES:-}" ] && [ "$n" -le "$STUB_FAIL_TIMES" ]; then
    # Fail having written nothing.
    echo "stub: remote tar failed (attempt $n)" >&2
    return 1
  fi
  if [ -n "${STUB_TRUNCATE:-}" ]; then
    # Truncate mid-stream and exit 0, as a proxy that hangs up cleanly would. "half" cuts at half the
    # stream's length without its trailing NUL padding, so the cut lands in the archive data whatever the
    # tar implementation pads, and however small the archive compresses.
    if [ "$STUB_TRUNCATE" = half ]; then
      local buf; buf=$(mktemp)
      sh -c "$cmd" 2>/dev/null > "$buf"
      head -c "$(python3 -c 'import sys; print(len(open(sys.argv[1], "rb").read().rstrip(b"\0")) // 2)' "$buf")" "$buf"
      rm -f "$buf"
    else
      sh -c "$cmd" 2>/dev/null | head -c "$STUB_TRUNCATE"
    fi
    return 0
  fi
  sh -c "$cmd"
}

. "$MOUNTER_SNIPPET"
"""

# The slice ends in `exit 1` on a failed entry, so report through the EXIT trap.
HARNESS = STUB + r"""
report() {
  printf 'TAR_Z=%s\n' "${TAR_Z-<unset>}"
  printf 'FAILED=%s\n' "${FAILED-}"
  printf 'ATTEMPTS=%s\n' "$(cat "$STUB_COUNTER")"
  printf 'PROBES=%s\n' "$(cat "$STUB_PROBES")"
  printf 'CREATES=%s\n' "$(grep -cx create "$STUB_LOG" || true)"
}
trap report EXIT

. "$STREAMER_SNIPPET"
"""


def _nowrite_shim(workdir: Path) -> Path:
    """A PATH dir where every filesystem-writing command fails; the remote probe runs under it."""
    shim = workdir / "nowrite"
    shim.mkdir(parents=True, exist_ok=True)
    for name in ("mkdir", "touch", "cp", "dd", "tee", "mktemp", "install"):
        f = shim / name
        f.write_text(f"#!/bin/sh\necho '{name}: Read-only file system' >&2\nexit 1\n")
        f.chmod(0o755)
    return shim


def _pad_then_compress_tar(workdir: Path) -> Path:
    """A PATH dir whose `tar -c...z...` pads the archive and then gzips it, as GNU tar does. bsdtar pads
    the compressed output instead, so on macOS a small archive still streams 10 KiB; this reproduces the
    much shorter compressed streams GNU tar writes."""
    real_tar = shutil.which("tar")
    shim = workdir / "gnu-like-tar"
    shim.mkdir(parents=True, exist_ok=True)
    f = shim / "tar"
    f.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  -c*z*) flags=$(printf %s "$1" | tr -d z); shift\n'
        f'         {real_tar} "$flags" "$@" | gzip -c; exit $? ;;\n'
        "esac\n"
        f'exec {real_tar} "$@"\n'
    )
    f.chmod(0o755)
    return shim


def stub_env(workdir: Path, pod_state: str = "Running/") -> dict:
    """Environment and state files for STUB, with the mounter pod in pod_state."""
    workdir.mkdir(parents=True, exist_ok=True)
    files = {"STUB_POD": pod_state, "STUB_LOG": "", "STUB_MANIFEST": "", "STUB_COUNTER": "0", "STUB_PROBES": "0"}
    env = dict(os.environ)
    for name, text in files.items():
        f = workdir / name.lower()
        f.write_text(text)
        env[name] = str(f)
    mounter = workdir / "mounter.sh"
    mounter.write_text(_extract_mounter())
    env.update(
        MOUNTER_SNIPPET=str(mounter),
        POD="stub-fetch-pod",
        RECIPE_SHORTNAME="stub",
        RUN_ID="stub-run",
        OWNER="stub-owner",
        FETCH_PLACEMENT_YAML="",
        ARTIFACTS_PVC_NAME="stub-artifacts",
        UTIL_IMAGE="busybox:1.36",
    )
    return env


def run_stream(
    workdir: Path,
    remote_dir: Path,
    entries: list[str],
    *,
    with_inputs: bool = False,
    no_compress: bool = False,
    probe: str = "run",
    fail_times: int = 0,
    truncate: int | str = 0,
    fail_entry: str = "",
    path_prefix: str = "",
    pod_lost_during: int = 0,
    pod_ends_after: int = 0,
) -> dict:
    local_dir = workdir / "local"
    stage = local_dir / ".fetch_staging"
    done = local_dir / ".fetch_done"
    for d in (stage, done):
        d.mkdir(parents=True, exist_ok=True)
    snippet = workdir / "streamer.sh"
    snippet.write_text(_extract_streamer())
    harness = workdir / "harness.sh"
    harness.write_text(HARNESS)

    env = dict(
        stub_env(workdir),
        STREAMER_SNIPPET=str(snippet),
        STUB_PROBE=probe,
        STUB_NOWRITE=str(_nowrite_shim(workdir)),
        STUB_FAIL_TIMES=str(fail_times) if fail_times else "",
        STUB_TRUNCATE=str(truncate) if truncate else "",
        STUB_FAIL_ENTRY=fail_entry,
        STUB_POD_LOST_DURING=str(pod_lost_during) if pod_lost_during else "",
        STUB_POD_ENDS_AFTER=str(pod_ends_after) if pod_ends_after else "",
        NAMESPACE="stub-ns",
        PARTIAL="0",
        REMOTE_RUN_DIR=str(remote_dir),
        local_dir=str(local_dir),
        STAGE_DIR=str(stage),
        DONE_DIR=str(done),
        MODE_TAG="withinputs" if with_inputs else "noinputs",
        MAX_TRIES="3",
        ENTRIES="\n".join(entries),
        TAR_EXCLUDES="" if with_inputs else "--exclude=inputs.json",
    )
    if no_compress:
        env["LLMB_FETCH_NO_COMPRESS"] = "1"
    else:
        env.pop("LLMB_FETCH_NO_COMPRESS", None)
    if path_prefix:
        env["PATH"] = path_prefix + os.pathsep + env["PATH"]

    p = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True)
    out = dict(
        rc=p.returncode,
        stdout=p.stdout,
        stderr=p.stderr,
        local_dir=local_dir,
        tar_z="",
        failed="",
        attempts=0,
        probes=0,
        creates=0,
    )
    for line in p.stdout.splitlines():
        if line.startswith("TAR_Z="):
            out["tar_z"] = line.split("=", 1)[1]
        elif line.startswith("FAILED="):
            out["failed"] = line.split("=", 1)[1].strip()
        elif line.startswith("ATTEMPTS="):
            out["attempts"] = int(line.split("=", 1)[1] or 0)
        elif line.startswith("PROBES="):
            out["probes"] = int(line.split("=", 1)[1] or 0)
        elif line.startswith("CREATES="):
            out["creates"] = int(line.split("=", 1)[1] or 0)
    return out


# ── fixture: a run tree shaped like the real one (symlinked inputs.json included) ────────────────────────
def make_remote_tree(root: Path) -> None:
    """concurrency_16/ + concurrency_32/ + _shared/, with inputs.json both real and symlinked, plus a
    loose top-level file, as laid out on the PVC."""
    shared = root / "_shared"
    shared.mkdir(parents=True)
    (shared / "inputs.json").write_text("PRETOKENIZED " * 400)
    (shared / "shared_notes.txt").write_text("shared\n")
    for c in (16, 32):
        d = root / f"concurrency_{c}"
        (d / "sub").mkdir(parents=True)
        (d / "metrics_summary.csv").write_text("metric,value\nttft_ms,%d\n" % c)
        (d / "profile_export.json").write_text('{"c": %d}\n' % c)
        (d / "sub" / "inputs.json").write_text("NESTED " * 100)
        # The per-step inputs.json is a symlink into _shared.
        os.symlink("../_shared/inputs.json", d / "inputs.json")
        os.symlink("../_shared", d / "shared_link")
    (root / "server_initial.id").write_text("srv-1\n")


def manifest(root: Path) -> str:
    """Path, type, mode, mtime, size and content hash for every entry. The fetch's bookkeeping directories
    are excluded, as `local_stats` excludes them, because they are stamped per fetch."""
    import hashlib
    import stat as st_mod

    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in (".fetch_done", ".fetch_staging"))
        for name in sorted(dirnames + filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(root)
            st = p.lstat()
            mode = oct(st_mod.S_IMODE(st.st_mode))
            if p.is_symlink():
                rows.append(f"{rel}\tLNK\t{mode}\t{int(st.st_mtime)}\t{os.readlink(p)}")
            elif p.is_dir():
                rows.append(f"{rel}\tDIR\t{mode}\t{int(st.st_mtime)}")
            else:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                rows.append(f"{rel}\tREG\t{mode}\t{int(st.st_mtime)}\t{st.st_size}\t{h}")
    return "\n".join(sorted(rows))


ENTRIES = ["_shared", "concurrency_16", "concurrency_32", "server_initial.id"]


# ── C: the compression decision ─────────────────────────────────────────────────────────────────────────
print("fetch stream — COMPRESSION DECISION (real bash, stubbed cluster)")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    remote = td / "remote"
    make_remote_tree(remote)

    r = run_stream(td / "c1", remote, ENTRIES)
    check("C1 mounter can gzip → compressed stream", r["tar_z"] == "z", f"TAR_Z={r['tar_z']!r}")
    check("C1 the fetch still succeeds", r["rc"] == 0 and not r["failed"], r["stderr"][-300:])
    check("C1 logs the compressed stream", "gzip-compressed remote-side" in r["stderr"], r["stderr"][-300:])
    check("C1 the probe ran exactly once", r["probes"] == 1, f"probes={r['probes']}")
    # The probe ran with every writing command disabled, so a "z" verdict also shows it needs no scratch space.
    check("C1 the probe reached its verdict with no writable filesystem", r["tar_z"] == "z")

    r = run_stream(td / "c2", remote, ENTRIES, probe="nogzip")
    check("C2 mounter has no gzip → uncompressed, not a failure", r["tar_z"] == "" and r["rc"] == 0)
    check(
        "C2 warns, naming the image and the missing gzip",
        "UNCOMPRESSED" in r["stderr"] and "busybox:1.36" in r["stderr"] and "no gzip binary" in r["stderr"],
        r["stderr"][-400:],
    )
    check("C2 still fetches every entry", not r["failed"], r["failed"])

    r = run_stream(td / "c2b", remote, ENTRIES, probe="notarz")
    check("C2b gzip present but tar refuses -z → uncompressed, distinct message", r["tar_z"] == "")
    check(
        "C2b names the actual fault (tar, not the missing binary)",
        "refuses -z" in r["stderr"] and "no gzip binary" not in r["stderr"],
        r["stderr"][-400:],
    )

    # C3: shim a local tar that rejects `-cz`. Matching the generic flag rather than the probe's exact
    # arguments keeps the test valid if the probe changes.
    shim = Path(tempfile.mkdtemp())
    real_tar = shutil.which("tar")
    (shim / "tar").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *-cz*) echo 'tar: unknown option -- z' >&2; exit 2 ;;\n"
        "esac\n"
        f'exec {real_tar} "$@"\n'
    )
    (shim / "tar").chmod(0o755)
    r = run_stream(td / "c3", remote, ENTRIES, path_prefix=str(shim))
    check("C3 local tar cannot gzip → uncompressed, not a failure", r["tar_z"] == "" and r["rc"] == 0)
    check("C3 warns about the local tar", "local" in r["stderr"] and "UNCOMPRESSED" in r["stderr"])
    check("C3 does not probe the mounter after the local end fails", r["probes"] == 0, f"probes={r['probes']}")
    shutil.rmtree(shim, ignore_errors=True)

    # C4 counts probe calls: a clean stderr would also be consistent with probing before honouring the var.
    r = run_stream(td / "c4", remote, ENTRIES, no_compress=True)
    check("C4 LLMB_FETCH_NO_COMPRESS=1 → uncompressed", r["tar_z"] == "" and r["rc"] == 0)
    check("C4 the remote is never probed at all", r["probes"] == 0, f"probes={r['probes']}")

    # C5 checks the source: a gzip-less tar killing its container cannot be reproduced locally.
    fn = _extract_streamer()
    fn = fn[: fn.index("entry_stats() {")]
    # Only the remote probe script; the local probe also contains `tar -cz`.
    remote_probe = fn[fn.index("remote_probe='") + len("remote_probe='") :]
    remote_probe = remote_probe[: remote_probe.index("'\n")]
    check(
        "C5 the remote probe checks for gzip before running tar -cz",
        "command -v gzip" in remote_probe
        and "tar -cz" in remote_probe
        and remote_probe.index("command -v gzip") < remote_probe.index("tar -cz"),
        "a bare `tar -cz` probe can kill the mounter on a gzip-less image",
    )
    check(
        "C5b the remote probe writes nothing to the mounter's filesystem",
        not any(w in remote_probe for w in ("mkdir", "touch", "mktemp", "/tmp")),
        remote_probe,
    )

    # C6: the exec itself fails, which says nothing about the image.
    r = run_stream(td / "c6", remote, ENTRIES, probe="execfail")
    check("C6 an unrunnable probe → uncompressed", r["tar_z"] == "")
    check("C6 reported as UNKNOWN", "UNKNOWN" in r["stderr"], r["stderr"][-400:])
    check(
        "C6 it does not blame UTIL_IMAGE",
        "no gzip binary" not in r["stderr"] and "refuses -z" not in r["stderr"],
        r["stderr"][-400:],
    )
    check("C6 it heals and retries before concluding", r["probes"] == 2, f"probes={r['probes']}")


# ── E: compressed == uncompressed, byte for byte ────────────────────────────────────────────────────────
print("fetch stream — EQUIVALENCE")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    remote = td / "remote"
    make_remote_tree(remote)

    for mode, with_inputs in (("noinputs", False), ("withinputs", True)):
        gz = run_stream(td / f"e-{mode}-gz", remote, ENTRIES, with_inputs=with_inputs)
        pl = run_stream(td / f"e-{mode}-plain", remote, ENTRIES, with_inputs=with_inputs, no_compress=True)
        check(f"E1 {mode}: both fetches succeed", gz["rc"] == 0 and pl["rc"] == 0, gz["stderr"][-300:])
        check(f"E1 {mode}: the compressed path was used", gz["tar_z"] == "z" and pl["tar_z"] == "")
        mg, mp = manifest(gz["local_dir"]), manifest(pl["local_dir"])
        check(
            f"E1 {mode}: trees are byte-identical (content+mode+mtime+type)",
            mg == mp and mg != "",
            f"{len(mg.splitlines())} vs {len(mp.splitlines())} entries",
        )

    gz_tree = run_stream(td / "e2", remote, ENTRIES)["local_dir"]
    link = gz_tree / "concurrency_16" / "shared_link"
    check("E2 -h dereferenced the symlinked dir into real content", link.is_dir() and not link.is_symlink())
    check("E2 the dereferenced dir carries its files", (link / "shared_notes.txt").is_file())

    kept = sorted(p.relative_to(gz_tree).as_posix() for p in gz_tree.rglob("inputs.json"))
    check("E3 default run drops every inputs.json, at every depth", kept == [], str(kept))
    check("E3 the results themselves are still there", (gz_tree / "concurrency_16" / "metrics_summary.csv").is_file())

    wi_tree = run_stream(td / "e3", remote, ENTRIES, with_inputs=True)["local_dir"]
    kept = sorted(p.relative_to(wi_tree).as_posix() for p in wi_tree.rglob("inputs.json"))
    check(
        "E3 --with-inputs keeps them, including the dereferenced symlink",
        "concurrency_16/inputs.json" in kept
        and "_shared/inputs.json" in kept
        and "concurrency_16/sub/inputs.json" in kept,
        str(kept),
    )


# ── P: pipefail still catches a bad stream ──────────────────────────────────────────────────────────────
print("fetch stream — INTEGRITY")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    remote = td / "remote"
    make_remote_tree(remote)

    for label, kw in (("compressed", {}), ("uncompressed", {"no_compress": True})):
        # P1: remote fails having written zero bytes. Without pipefail the local tar accepts 0 bytes,
        # exits 0, and the fetch reports success over an empty directory.
        r = run_stream(td / f"p1-{label}", remote, ENTRIES, fail_times=99, **kw)
        check(f"P1 {label}: zero-byte failed stream → non-zero exit", r["rc"] != 0, f"rc={r['rc']}")
        check(
            f"P1 {label}: every entry recorded as FAILED", sorted(r["failed"].split()) == sorted(ENTRIES), r["failed"]
        )
        check(
            f"P1 {label}: nothing was promoted into the local tree",
            not [p for p in r["local_dir"].iterdir() if not p.name.startswith(".fetch_")],
            str(list(r["local_dir"].iterdir())),
        )
        check(
            f"P1 {label}: no staging debris left behind",
            not list((r["local_dir"] / ".fetch_staging").iterdir()),
        )

        # P2: the stream is cut mid-transfer. The cut is relative to each entry's archive data, since a
        # fixed offset can exceed a small compressed archive.
        r = run_stream(td / f"p2-{label}", remote, ENTRIES, truncate="half", **kw)
        check(f"P2 {label}: truncated stream → non-zero exit", r["rc"] != 0, f"rc={r['rc']}")
        check(
            f"P2 {label}: every truncated entry is recorded in FAILED",
            sorted(r["failed"].split()) == sorted(ENTRIES),
            r["failed"],
        )
        promoted = sorted(p.name for p in r["local_dir"].iterdir() if not p.name.startswith(".fetch_"))
        check(
            f"P2 {label}: no half-extracted entry was promoted",
            all(n not in promoted for n in r["failed"].split()),
            f"promoted={promoted} failed={r['failed']}",
        )

    # P3: the same cut with a tar that pads before compressing (GNU tar), whose streams are far shorter.
    r = run_stream(td / "p3-gnu-like", remote, ENTRIES, truncate="half", path_prefix=str(_pad_then_compress_tar(td)))
    check("P3 pad-then-compress tar: compression was negotiated", r["tar_z"] == "z", f"TAR_Z={r['tar_z']!r}")
    check("P3 pad-then-compress tar: truncated stream → non-zero exit", r["rc"] != 0, f"rc={r['rc']}")
    check(
        "P3 pad-then-compress tar: every entry is recorded in FAILED",
        sorted(r["failed"].split()) == sorted(ENTRIES),
        r["failed"],
    )


# ── R: the retry budget still works ─────────────────────────────────────────────────────────────────────
print("fetch stream — RETRY BUDGET")
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    remote = td / "remote"
    make_remote_tree(remote)

    # The first entry fails twice, then everything succeeds — inside MAX_TRIES=3.
    r = run_stream(td / "r1", remote, ENTRIES, fail_times=2)
    check("R1 an entry that fails twice still lands on attempt 3", r["rc"] == 0 and not r["failed"], r["failed"])
    check("R1 retried twice", r["attempts"] == len(ENTRIES) + 2, f"attempts={r['attempts']}")
    check(
        "R1 the whole tree landed",
        (r["local_dir"] / "concurrency_32" / "metrics_summary.csv").is_file(),
    )

    # One entry is permanently broken; the others must still be fetched.
    r = run_stream(td / "r2", remote, ENTRIES, fail_entry="concurrency_16")
    check("R2 one bad entry does not abort the others", r["failed"].strip() == "concurrency_16", r["failed"])
    check("R2 the good entries landed", (r["local_dir"] / "concurrency_32" / "metrics_summary.csv").is_file())
    check("R2 the run still exits non-zero", r["rc"] != 0, f"rc={r['rc']}")
    check(
        "R2 the bad entry used exactly MAX_TRIES attempts",
        r["attempts"] == (len(ENTRIES) - 1) + 3,
        f"attempts={r['attempts']}",
    )


# ── L: the mounter pod's lifecycle ──────────────────────────────────────────────────────────────────────
print("fetch stream — MOUNTER POD LIFECYCLE")
SPAWN = STUB + "spawn_mounter && echo SPAWN_RC=0 || echo SPAWN_RC=$?\n"


def run_spawn(workdir: Path, pod_state: str, delete: str = "gone") -> dict:
    env = dict(stub_env(workdir, pod_state), STUB_DELETE=delete)
    harness = workdir / "spawn.sh"
    harness.write_text(SPAWN)
    p = subprocess.run(["bash", str(harness)], env=env, capture_output=True, text=True)
    return dict(
        ok="SPAWN_RC=0" in p.stdout,
        stderr=p.stderr,
        log=Path(env["STUB_LOG"]).read_text().split(),
        state=Path(env["STUB_POD"]).read_text(),
        manifest=Path(env["STUB_MANIFEST"]).read_text(),
    )


TERMINATING = "Running/2026-01-01T00:00:00Z"
with tempfile.TemporaryDirectory() as td:
    td = Path(td)

    # L1: the previous fetch's cleanup deletes without waiting, so its pod can still be terminating.
    r = run_spawn(td / "l1", TERMINATING)
    check("L1 spawn over a terminating pod succeeds", r["ok"] and r["state"] == "Running/", r["stderr"][-300:])
    check("L1 it waits for the deletion, then creates", r["log"] == ["delete-wait", "create"], str(r["log"]))

    r = run_spawn(td / "l2", TERMINATING, delete="stuck")
    check("L2 a pod that will not go away fails the spawn", not r["ok"], r["stderr"][-300:])
    check(
        "L2 nothing is created or applied over it", "create" not in r["log"] and "apply" not in r["log"], str(r["log"])
    )

    r = run_spawn(td / "l3", TERMINATING, delete="noop")
    check(
        "L3 a pod still present after delete is never reused", not r["ok"] and "create" not in r["log"], str(r["log"])
    )
    check("L3 create refused it", "create-refused" in r["log"], str(r["log"]))

    # L4: the pod's lifetime comes from activeDeadlineSeconds, and its PID 1 exits 0 on TERM.
    manifest = run_spawn(td / "l4", "")["manifest"]
    deadline = re.search(r"^  activeDeadlineSeconds: (\d+)$", manifest, re.M)
    check("L4 the pod is bounded by activeDeadlineSeconds", bool(deadline) and int(deadline.group(1)) > 0, manifest)
    check("L4 restartPolicy stays Never", "restartPolicy: Never" in manifest)
    found = re.search(r"^\s*command: (\[.*\])$", manifest, re.M)
    command = json.loads(found.group(1)) if found else ["sh", "-c", "exit 99"]

    def term_after(delay: float, path_prefix: str = "") -> tuple[bool, int | None, float]:
        """Run the pod command, TERM its shell after `delay`, and return (alive at TERM, exit code, seconds)."""
        env = dict(os.environ)
        if path_prefix:
            env["PATH"] = path_prefix + os.pathsep + env["PATH"]
        proc = subprocess.Popen(command, env=env, start_new_session=True)
        try:
            time.sleep(delay)
            alive = proc.poll() is None
            t0 = time.monotonic()
            proc.send_signal(signal.SIGTERM)
            try:
                rc = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                rc = None
            return alive, rc, time.monotonic() - t0
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass

    alive, rc, secs = term_after(0.5)
    check("L4 TERM ends the pod's command promptly with exit 0", alive and rc == 0 and secs < 2, f"rc={rc} {secs:.1f}s")
    fast_sleep = td / "fast-sleep"
    fast_sleep.mkdir()
    (fast_sleep / "sleep").write_text(f"#!/bin/sh\nexec {shutil.which('sleep')} 0.05\n")
    (fast_sleep / "sleep").chmod(0o755)
    alive, rc, _ = term_after(0.5, str(fast_sleep))
    check("L4 the command outlives its sleeps, so only the deadline ends it", alive and rc == 0, f"rc={rc}")

    remote = td / "remote"
    make_remote_tree(remote)
    # L5: the pod is removed while an entry streams; the retry starts a new pod.
    r = run_stream(td / "l5", remote, ENTRIES, pod_lost_during=2)
    check("L5 pod lost mid-stream → the fetch still succeeds", r["rc"] == 0 and not r["failed"], r["stderr"][-400:])
    check("L5 one new pod was started", r["creates"] == 1, f"creates={r['creates']}")
    check(
        "L5 only the interrupted attempt was repeated", r["attempts"] == len(ENTRIES) + 1, f"attempts={r['attempts']}"
    )

    # L6: the pod reaches its deadline between entries; the next entry starts a new pod before streaming.
    r = run_stream(td / "l6", remote, ENTRIES, pod_ends_after=2)
    check("L6 pod past its deadline → the fetch still succeeds", r["rc"] == 0 and not r["failed"], r["stderr"][-400:])
    check("L6 one new pod was started", r["creates"] == 1, f"creates={r['creates']}")
    check("L6 no attempt was spent on the stopped pod", r["attempts"] == len(ENTRIES), f"attempts={r['attempts']}")
    check("L6 the replacement is logged", "is Failed; starting a new one" in r["stderr"], r["stderr"][-400:])

    r = run_stream(td / "l7", remote, ENTRIES)
    check("L7 a running pod is never replaced", r["rc"] == 0 and r["creates"] == 0, f"creates={r['creates']}")


# ── S: the run_meta.json cluster stamp ──────────────────────────────────────────────────────────────────
print("fetch stream — RUN_META CLUSTER STAMP")


def stamp(meta: Path, cluster: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", "-", str(meta), cluster], input=_extract_cluster_stamp(), capture_output=True, text=True
    )


with tempfile.TemporaryDirectory() as td:
    meta = Path(td) / "run_meta.json"
    # Written the way the bench Job writes it.
    source = (
        json.dumps({"cluster": "cluster-a", "run_id": "r1", "started_at_utc": "x"}, indent=2, sort_keys=True) + "\n"
    )
    meta.write_text(source)
    p = stamp(meta, "cluster-a")
    check("S1 a run that already records its cluster keeps the source's bytes", meta.read_text() == source, p.stderr)
    stamp(meta, "other")
    check("S1 an existing cluster is never overwritten", meta.read_text() == source)

    meta.write_text(json.dumps({"run_id": "r1", "started_at_utc": "x"}, indent=2, sort_keys=True) + "\n")
    p = stamp(meta, "cluster-a")
    text = meta.read_text()
    check(
        "S2 a missing cluster is stamped",
        p.returncode == 0 and json.loads(text).get("cluster") == "cluster-a",
        p.stderr,
    )
    check("S2 the stamped file keeps its other fields", json.loads(text).get("run_id") == "r1")
    check("S2 the stamped file ends with a newline, as the source does", text.endswith("}\n"), repr(text[-5:]))


print()
if fails:
    print(f"selftest_fetch_stream FAILED ({len(fails)}): " + "; ".join(fails))
    raise SystemExit(1)
print("selftest_fetch_stream OK")
