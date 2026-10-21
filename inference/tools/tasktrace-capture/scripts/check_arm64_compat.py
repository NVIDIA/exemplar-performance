#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Check which Terminal Bench 2.0 task environment images support linux/arm64.

Usage:
    # Pass a list of FROM image refs (one per line) on stdin:
    grep -rh "^FROM" /tmp/tb2_tasks/*/environment/Dockerfile | sort -u \
        | python3 scripts/check_arm64_compat.py

    # Or pass image refs as arguments:
    python3 scripts/check_arm64_compat.py ubuntu:22.04 jefzda/sweap-images:sometag

Output:
    arm64-compat.txt  — tasks whose base image supports linux/arm64
    arm64-no.txt      — tasks whose base image is amd64-only
    arm64-unknown.txt — could not determine (private registry, auth needed, etc.)
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def get_dockerhub_token(repo: str) -> str:
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())["token"]


def check_image_arm64(image_ref: str) -> tuple[bool | None, str]:
    """
    Returns (is_arm64_compatible, detail_string).
    is_arm64_compatible is None when we can't determine.
    """
    image_ref = image_ref.strip()
    if not image_ref or image_ref.startswith("#"):
        return None, "skipped"

    # Strip FROM keyword if passed raw FROM lines
    if image_ref.upper().startswith("FROM "):
        image_ref = image_ref[5:].strip()

    # Strip multi-stage alias: "ubuntu:24.04 AS build" -> "ubuntu:24.04"
    parts = image_ref.split()
    if len(parts) >= 3 and parts[1].upper() == "AS":
        image_ref = parts[0]
    elif len(parts) > 1:
        image_ref = parts[0]  # also covers --platform= as first token

    # Ignore --platform= flags (FROM --platform=linux/amd64 image:tag)
    if image_ref.startswith("--"):
        return None, f"--platform flag without image ref: {image_ref}"

    # Ignore build-arg substitutions like $BASE_IMAGE
    if "$" in image_ref:
        return None, f"build-arg variable, cannot resolve: {image_ref}"

    # Parse repo and tag/digest
    if "@" in image_ref:
        repo_tag, _, digest = image_ref.partition("@")
    else:
        repo_tag = image_ref
        digest = None

    if ":" in repo_tag:
        repo, _, tag = repo_tag.rpartition(":")
    else:
        repo = repo_tag
        tag = "latest"

    # Handle official images (no slash → library/)
    if "/" not in repo:
        repo = f"library/{repo}"

    # Only Docker Hub is handled; other registries need separate auth
    registry = "https://registry-1.docker.io"
    ref = digest or tag

    try:
        token = get_dockerhub_token(repo)
    except Exception as e:
        return None, f"auth failed: {e}"

    headers = {
        "Authorization": f"Bearer {token}",
        # Accept both manifest list (multi-arch) and OCI index
        "Accept": (
            "application/vnd.docker.distribution.manifest.list.v2+json,"
            "application/vnd.oci.image.index.v1+json,"
            "application/vnd.docker.distribution.manifest.v2+json,"
            "application/vnd.oci.image.manifest.v1+json"
        ),
    }

    url = f"{registry}/v2/{repo}/manifests/{ref}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as r:
            content_type = r.headers.get("Content-Type", "")
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return None, f"request failed: {e}"

    # Multi-arch manifest list
    if "list" in content_type or "index" in content_type:
        platforms = [m.get("platform", {}) for m in data.get("manifests", [])]
        arm64 = [p for p in platforms if p.get("architecture") == "arm64"]
        if arm64:
            return True, f"multi-arch, arm64 found: {arm64}"
        else:
            archs = [f"{p.get('os')}/{p.get('architecture')}" for p in platforms]
            return False, f"multi-arch but no arm64; available: {archs}"

    # Single-arch manifest — fetch config blob for architecture
    config_digest = data.get("config", {}).get("digest")
    if not config_digest:
        return None, "no config digest in manifest"

    blob_url = f"{registry}/v2/{repo}/blobs/{config_digest}"
    req2 = urllib.request.Request(blob_url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req2) as r:
            cfg = json.loads(r.read())
    except Exception as e:
        return None, f"config blob fetch failed: {e}"

    arch = cfg.get("architecture", "unknown")
    os_ = cfg.get("os", "unknown")
    if arch == "arm64":
        return True, f"single-arch: {os_}/{arch}"
    else:
        return False, f"single-arch: {os_}/{arch}"


def main():
    if len(sys.argv) > 1:
        lines = sys.argv[1:]
    else:
        lines = [line.strip() for line in sys.stdin if line.strip()]

    compat, incompat, unknown = [], [], []

    for line in lines:
        image = line.strip()
        if not image or image.startswith("#"):
            continue
        result, detail = check_image_arm64(image)
        status = "YES  " if result is True else ("NO   " if result is False else "??   ")
        print(f"{status} {image}\n       → {detail}")
        if result is True:
            compat.append(image)
        elif result is False:
            incompat.append(image)
        else:
            unknown.append(image)

    print("\n=== Summary ===")
    print(f"arm64 compatible : {len(compat)}")
    print(f"amd64 only       : {len(incompat)}")
    print(f"unknown          : {len(unknown)}")

    Path("arm64-compat.txt").write_text("\n".join(compat) + "\n" if compat else "")
    Path("arm64-no.txt").write_text("\n".join(incompat) + "\n" if incompat else "")
    Path("arm64-unknown.txt").write_text("\n".join(unknown) + "\n" if unknown else "")
    print("\nResults written to arm64-compat.txt / arm64-no.txt / arm64-unknown.txt")


if __name__ == "__main__":
    main()
