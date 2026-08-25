#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""Cancel exactly one active detached LLMB run by run-id.

The run-id is resolved through the same local/cluster-side index used by status/logs/collect.  Before deleting
anything, this command fetches the exact Job and fail-closes unless its run-id and LLMB ownership labels match.
Deleting that Job triggers Kubernetes owner-reference GC for only its serving Deployment/Service.  The durable
index ConfigMap and control-PVC status are retained so the cancelled run remains diagnosable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import profile_resolver as pr  # noqa: E402
import resilient_status as rs  # noqa: E402

MANAGED_BY = "llmb-recipe"


def validate_job_identity(job: dict, run_id: str) -> list[str]:
    """Pure fail-closed identity check for the destructive target."""
    meta = job.get("metadata", {}) or {}
    labels = meta.get("labels", {}) or {}
    issues = []
    if labels.get("llmb.nvidia.com/run-id") != run_id:
        issues.append(f"run-id label is {labels.get('llmb.nvidia.com/run-id')!r}, expected {run_id!r}")
    if labels.get("app.kubernetes.io/managed-by") != MANAGED_BY:
        issues.append(f"managed-by label is {labels.get('app.kubernetes.io/managed-by')!r}, expected {MANAGED_BY!r}")
    if job.get("kind") not in (None, "Job"):
        issues.append(f"resolved object kind is {job.get('kind')!r}, expected 'Job'")
    return issues


def run_owner_ref(job: dict) -> tuple[str, str] | None:
    """Return the controller run-owner Job reference for a run-owner-backed lane."""
    refs = (job.get("metadata", {}) or {}).get("ownerReferences", []) or []
    matches = [
        (str(ref.get("name") or ""), str(ref.get("uid") or ""))
        for ref in refs
        if ref.get("kind") == "Job" and ref.get("controller") is True
    ]
    matches = [(name, uid) for name, uid in matches if name and uid]
    if len(matches) > 1:
        raise ValueError("multiple controller Job ownerReferences; refusing to guess a lifecycle root")
    return matches[0] if matches else None


def validate_run_owner(owner: dict, expected_name: str, expected_uid: str, run_id: str) -> list[str]:
    """Fail closed before deleting the lifecycle root instead of only its driver child."""
    meta = owner.get("metadata", {}) or {}
    labels = meta.get("labels", {}) or {}
    issues = []
    owner_kind = owner.get("kind")
    owner_name = meta.get("name")
    owner_uid = meta.get("uid")
    owner_run_id = labels.get("llmb.nvidia.com/run-id")
    if owner_kind not in (None, "Job"):
        issues.append(f"run-owner object kind is {owner_kind!r}, expected Job")
    if meta.get("name") != expected_name:
        issues.append(f"run-owner name is {owner_name!r}, expected {expected_name!r}")
    if meta.get("uid") != expected_uid:
        issues.append(f"run-owner uid is {owner_uid!r}, expected {expected_uid!r}")
    if labels.get("app.kubernetes.io/managed-by") != MANAGED_BY:
        issues.append("run-owner is not managed by llmb-recipe")
    if labels.get("app.kubernetes.io/component") != "run-owner":
        issues.append("controller owner is not labelled component=run-owner")
    if labels.get("llmb.nvidia.com/run-id") != run_id:
        issues.append(f"run-owner run-id label is {owner_run_id!r}, expected {run_id!r}")
    return issues


def select_run_owner(items: list[dict], run_id: str) -> dict | None:
    """Pure: resolve exactly one labelled lifecycle root after its driver Job is already gone."""
    matches = []
    for item in items:
        meta = item.get("metadata", {}) or {}
        labels = meta.get("labels", {}) or {}
        if (
            item.get("kind") in (None, "Job")
            and labels.get("llmb.nvidia.com/run-id") == run_id
            and labels.get("app.kubernetes.io/managed-by") == MANAGED_BY
            and labels.get("app.kubernetes.io/component") == "run-owner"
        ):
            matches.append(item)
    if len(matches) > 1:
        names = sorted(str((item.get("metadata", {}) or {}).get("name") or "?") for item in matches)
        raise ValueError(f"multiple run-owner Jobs match run-id {run_id!r}: {', '.join(names)}")
    if not matches:
        return None
    meta = matches[0].get("metadata", {}) or {}
    if not meta.get("name") or not meta.get("uid"):
        raise ValueError(f"run-owner Job matching run-id {run_id!r} lacks a name or UID")
    return matches[0]


def owned_targets(jobs: dict | list[dict], deployments: list[dict], pods: list[dict]) -> tuple[list[str], list[str]]:
    """Pure: deployments owned by any Job in the lifecycle chain and their current Pods."""
    if isinstance(jobs, dict):
        jobs = [jobs]
    owners = {
        ((job.get("metadata", {}) or {}).get("name", ""), (job.get("metadata", {}) or {}).get("uid", ""))
        for job in jobs
    }
    owned = []
    selectors = []
    for dep in deployments:
        refs = (dep.get("metadata", {}) or {}).get("ownerReferences", []) or []
        if any(
            r.get("kind") == "Job"
            and any(r.get("name") == name and (not uid or r.get("uid") == uid) for name, uid in owners)
            for r in refs
        ):
            owned.append((dep.get("metadata", {}) or {}).get("name", ""))
            sel = ((dep.get("spec", {}) or {}).get("selector", {}) or {}).get("matchLabels", {}) or {}
            if sel:
                selectors.append(sel)
    pod_names = []
    for pod in pods:
        labels = (pod.get("metadata", {}) or {}).get("labels", {}) or {}
        if any(all(labels.get(k) == v for k, v in sel.items()) for sel in selectors):
            pod_names.append((pod.get("metadata", {}) or {}).get("name", ""))
    return sorted(n for n in owned if n), sorted(n for n in pod_names if n)


def run(base: list[str], *args: str, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*base, *args], capture_output=True, text=True, timeout=timeout)


def decoded_items(result: subprocess.CompletedProcess[str], resource: str) -> list[dict]:
    """Decode a Kubernetes list response; failed or malformed discovery is never an empty list."""
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown kubectl error").strip()[:240]
        raise ValueError(f"cannot list {resource}: {detail}")
    try:
        payload = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode {resource} list response: {exc}") from exc
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"{resource} list response has no items array")
    return items


def gone(base: list[str], kind: str, name: str) -> bool:
    r = run(base, "get", kind, name, "-o", "name")
    return r.returncode != 0 and ("NotFound" in r.stderr or "not found" in r.stderr.lower())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="llmb-k8s cancel",
        description="Cancel exactly one detached LLMB run; owner-reference GC removes only its server.",
    )
    ap.add_argument("run_id")
    ap.add_argument("--cluster", default=None, help="profile name; optional when the local submit record has it")
    ap.add_argument("--yes", "-y", action="store_true", help="confirm non-interactively")
    ap.add_argument("--dry-run", action="store_true", help="resolve and print the exact targets; delete nothing")
    ap.add_argument(
        "--wait-seconds",
        type=int,
        default=180,
        help="wait for Job/server Pods to disappear (default: 180; graceful shutdown may take 120s)",
    )
    args = ap.parse_args(argv)
    if args.wait_seconds < 0:
        ap.error("--wait-seconds must be >= 0")

    rec = rs._resolve(args.run_id, args.cluster)
    profile = rec.get("profile") or args.cluster
    if not profile:
        sys.exit("cancel: no profile resolved; pass --cluster <profile>")
    prof = pr._read_env(pr.profile_env_path(profile))
    ns = (rec.get("namespace") or prof.get("NAMESPACE") or "").strip()
    job_name = (rec.get("job_name") or "").strip()
    if not ns or not job_name:
        sys.exit("cancel: the run record lacks namespace/job_name; refusing an unscoped deletion")
    base = rs._kbase(prof, ns)

    driver_present = True
    jr = run(base, "get", "job", job_name, "-o", "json")
    if jr.returncode != 0:
        if "NotFound" in jr.stderr or "not found" in jr.stderr.lower():
            driver_present = False
            roots_result = run(
                base,
                "get",
                "jobs",
                "-l",
                (
                    f"app.kubernetes.io/managed-by={MANAGED_BY},"
                    f"app.kubernetes.io/component=run-owner,llmb.nvidia.com/run-id={args.run_id}"
                ),
                "-o",
                "json",
            )
            if roots_result.returncode != 0:
                sys.exit(
                    f"cancel: driver job/{job_name} is absent, but run-owner discovery failed: "
                    f"{roots_result.stderr.strip()[:240]}"
                )
            try:
                lifecycle_job = select_run_owner(
                    json.loads(roots_result.stdout or '{"items":[]}').get("items", []), args.run_id
                )
            except ValueError as exc:
                sys.exit(f"cancel: REFUSING — {exc}")
            if lifecycle_job is None:
                print(
                    f"cancel: run {args.run_id} is already stopped — driver job/{job_name} and its run-owner "
                    f"are absent in {ns}"
                )
                return 0
            lifecycle_meta = lifecycle_job.get("metadata", {}) or {}
            lifecycle_name = str(lifecycle_meta.get("name") or "")
            lifecycle_uid = str(lifecycle_meta.get("uid") or "")
            lifecycle_is_run_owner = True
            owner_issues = validate_run_owner(lifecycle_job, lifecycle_name, lifecycle_uid, args.run_id)
            if owner_issues:
                sys.exit(
                    "cancel: REFUSING — discovered run-owner failed identity checks:\n  - "
                    + "\n  - ".join(owner_issues)
                )
            job = {}
        else:
            sys.exit(f"cancel: cannot read job/{job_name}: {jr.stderr.strip()[:240]}")
    else:
        job = json.loads(jr.stdout)
        issues = validate_job_identity(job, args.run_id)
        if issues:
            sys.exit("cancel: REFUSING — resolved Job failed identity checks:\n  - " + "\n  - ".join(issues))

        lifecycle_job = job
        lifecycle_name = job_name
        lifecycle_uid = str((job.get("metadata", {}) or {}).get("uid") or "")
        lifecycle_is_run_owner = False
        if not lifecycle_uid:
            sys.exit(f"cancel: REFUSING — resolved driver job/{job_name} has no UID")
        try:
            owner_ref = run_owner_ref(job)
        except ValueError as exc:
            sys.exit(f"cancel: REFUSING — {exc}")
        if owner_ref:
            owner_name, owner_uid = owner_ref
            owner_result = run(base, "get", "job", owner_name, "-o", "json")
            if owner_result.returncode != 0:
                sys.exit(
                    f"cancel: REFUSING — driver job/{job_name} names run-owner job/{owner_name}, "
                    f"but it cannot be read: {owner_result.stderr.strip()[:240]}"
                )
            lifecycle_job = json.loads(owner_result.stdout)
            owner_issues = validate_run_owner(lifecycle_job, owner_name, owner_uid, args.run_id)
            if owner_issues:
                sys.exit(
                    "cancel: REFUSING — resolved run-owner failed identity checks:\n  - " + "\n  - ".join(owner_issues)
                )
            lifecycle_name = owner_name
            lifecycle_uid = owner_uid
            lifecycle_is_run_owner = True

    dr = run(base, "get", "deployments", "-l", "app.kubernetes.io/managed-by=llmb-recipe", "-o", "json")
    prun = run(base, "get", "pods", "-o", "json")
    try:
        deployments = decoded_items(dr, "Deployments")
        pods = decoded_items(prun, "Pods")
    except ValueError as exc:
        sys.exit(f"cancel: REFUSING — resource discovery failed before deletion: {exc}")
    # Depending on the launcher generation, server Deployments may be owned directly by the run-owner or by
    # its driver child. Track both forms so cancellation never reports success before GPU descendants leave.
    ownership_jobs = [lifecycle_job]
    if driver_present:
        ownership_jobs.append(job)
    dep_names, pod_names = owned_targets(ownership_jobs, deployments, pods)

    print(f"cancel: run-id {args.run_id}")
    print(f"  profile/namespace: {profile} / {ns}")
    print(f"  exact driver Job:  {job_name}{'' if driver_present else ' (already absent)'}")
    print(f"  lifecycle root:    job/{lifecycle_name}")
    print(f"  owned deployments: {', '.join(dep_names) if dep_names else '(none found)'}")
    print(f"  current server pods: {', '.join(pod_names) if pod_names else '(none found)'}")
    print("  preserved:         artifacts PVC, control-PVC logs/status, run-id index ConfigMap")
    if args.dry_run:
        print("cancel: dry-run only — nothing deleted")
        return 0

    if not args.yes:
        if not sys.stdin.isatty():
            print(
                "cancel: confirmation required in non-interactive mode; inspect above, then re-run with --yes",
                file=sys.stderr,
            )
            return 2
        answer = input(f"Cancel only run {args.run_id}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("cancel: cancelled by user; nothing deleted")
            return 0

    # Re-read immediately before the destructive call. Names are reusable in Kubernetes; without this check a
    # delete-by-name could remove a replacement Job created after discovery. This is intentionally fail-closed.
    current_result = run(base, "get", "job", lifecycle_name, "-o", "json")
    if current_result.returncode != 0:
        print(
            f"cancel: REFUSING — lifecycle root job/{lifecycle_name} changed or became unreadable before deletion: "
            f"{current_result.stderr.strip()[:240]}",
            file=sys.stderr,
        )
        return 1
    try:
        current_lifecycle = json.loads(current_result.stdout)
    except json.JSONDecodeError as exc:
        print(f"cancel: REFUSING — cannot decode lifecycle root before deletion: {exc}", file=sys.stderr)
        return 1
    if lifecycle_is_run_owner:
        current_issues = validate_run_owner(current_lifecycle, lifecycle_name, lifecycle_uid, args.run_id)
    else:
        current_issues = validate_job_identity(current_lifecycle, args.run_id)
        current_meta = current_lifecycle.get("metadata", {}) or {}
        if current_meta.get("name") != lifecycle_name:
            current_issues.append(f"driver name is {current_meta.get('name')!r}, expected {lifecycle_name!r}")
        if current_meta.get("uid") != lifecycle_uid:
            current_issues.append(f"driver uid is {current_meta.get('uid')!r}, expected {lifecycle_uid!r}")
    if current_issues:
        print(
            "cancel: REFUSING — lifecycle root identity changed before deletion:\n  - " + "\n  - ".join(current_issues),
            file=sys.stderr,
        )
        return 1

    deleted = run(base, "delete", "job", lifecycle_name, "--wait=true", "--timeout=60s", timeout=70)
    if deleted.returncode != 0 and "NotFound" not in deleted.stderr:
        print(f"cancel: failed to delete job/{lifecycle_name}: {deleted.stderr.strip()[:240]}", file=sys.stderr)
        return 1
    print(f"cancel: deleted lifecycle root job/{lifecycle_name}; waiting for owner-reference cleanup …")

    deadline = time.monotonic() + args.wait_seconds
    while True:
        remaining = []
        if lifecycle_name != job_name and not gone(base, "job", lifecycle_name):
            remaining.append(f"job/{lifecycle_name}")
        if not gone(base, "job", job_name):
            remaining.append(f"job/{job_name}")
        remaining += [f"deployment/{n}" for n in dep_names if not gone(base, "deployment", n)]
        remaining += [f"pod/{n}" for n in pod_names if not gone(base, "pod", n)]
        if not remaining:
            print(f"cancel: complete — run {args.run_id} and its owned server are gone")
            print(f"  verify capacity: llmb-k8s capacity {profile}")
            return 0
        if time.monotonic() >= deadline:
            print(
                "cancel: Job deletion succeeded, but graceful server shutdown is still pending: "
                + ", ".join(remaining),
                file=sys.stderr,
            )
            print(f"  re-check: llmb-k8s capacity {profile}", file=sys.stderr)
            return 2
        time.sleep(3)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
