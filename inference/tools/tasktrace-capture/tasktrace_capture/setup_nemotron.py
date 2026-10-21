# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Download Nemotron-Terminal-Synthetic-Tasks for RL workload benchmarking."""

import argparse
import os
import sys
import tarfile
from pathlib import Path

REPO_ID = "nvidia/Nemotron-Terminal-Synthetic-Tasks"
DOMAINS = [
    "data_processing",
    "data_science",
    "debugging",
    "file_operations",
    "scientific_computing",
    "security",
]


def _check_token():
    """Check for HF token and return it, or print instructions and exit."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Check if huggingface-cli login was used (token cached on disk)
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    # No token found — try without auth first (public dataset, but rate-limited)
    return None


def _download_domain(domain, dest, token):
    """Download and extract one domain tarball."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    extract_dir = dest / "skill_based" / "mixed" / domain
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  Skipping {domain} (already extracted at {extract_dir})")
        return True

    filename = f"skill_based/mixed/{domain}.tar.gz"
    print(f"  Downloading {domain}...")

    try:
        local_path = hf_hub_download(
            REPO_ID,
            filename,
            repo_type="dataset",
            local_dir=str(dest),
            token=token,
        )
    except HfHubHTTPError as e:
        if "401" in str(e) or "403" in str(e):
            print(f"\n  ERROR: Authentication required to download {REPO_ID}.")
            print("  Set your Hugging Face token via one of:")
            print("    export HF_TOKEN=hf_...")
            print("    huggingface-cli login")
            print("  Get a token at: https://huggingface.co/settings/tokens")
            return False
        if "429" in str(e):
            print("\n  ERROR: Rate limited by Hugging Face (too many anonymous requests).")
            print("  Set a HF token to avoid rate limits:")
            print("    export HF_TOKEN=hf_...")
            print("    huggingface-cli login")
            print("  Get a token at: https://huggingface.co/settings/tokens")
            return False
        raise

    tarball = Path(local_path)
    if not tarball.exists():
        # hf_hub_download may return a symlink to cache; find the actual file
        tarball = dest / filename
    if not tarball.exists():
        print(f"  ERROR: Expected tarball not found at {tarball}")
        return False

    print(f"  Extracting {domain}...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:gz") as tar:
        tar.extractall(path=extract_dir.parent)

    # Clean up tarball (the cache still has it)
    try:
        tarball.unlink()
    except Exception:
        pass

    # Strip private docker_image references from task.toml files.
    # The dataset ships with docker_image pointing to a private container
    # registry. Removing it forces Harbor to build from the Dockerfile instead,
    # which uses a public base image (ghcr.io/laude-institute/t-bench/...).
    _strip_private_docker_images(extract_dir)

    return True


_PUBLIC_REGISTRIES = frozenset(
    {
        "docker.io",
        "registry.hub.docker.com",
        "ghcr.io",
        "gcr.io",
        "public.ecr.aws",
        "quay.io",
        "mcr.microsoft.com",
    }
)


def _is_private_image(val: str) -> bool:
    """Return True if val looks like a private-registry docker_image."""
    if "<private-registry>" in val:
        return True
    if "/" not in val:
        # Bare image name like "ubuntu:22.04" — Docker Hub (public)
        return False
    host = val.split("/")[0]
    if "." not in host and ":" not in host:
        # Docker Hub user/image (no dots in host part)
        return False
    # Dotted hostname — check against the public allowlist
    return host not in _PUBLIC_REGISTRIES


def _strip_private_docker_images(extract_dir):
    """Remove private docker_image lines from task.toml files.

    Strips any docker_image line that references a non-public registry
    or the '<private-registry>' placeholder. Public registries (Docker Hub,
    ghcr.io, gcr.io, quay.io, etc.) are preserved.
    """
    count = 0
    for toml_file in extract_dir.rglob("task.toml"):
        try:
            text = toml_file.read_text()
            changed = False
            new_lines = []
            for line in text.splitlines(keepends=True):
                stripped = line.strip()
                if stripped.startswith("docker_image"):
                    val = stripped.split("=", 1)[-1].strip().strip('"').strip("'")
                    if _is_private_image(val):
                        changed = True
                        continue
                new_lines.append(line)
            if changed:
                toml_file.write_text("".join(new_lines))
                count += 1
        except Exception:
            continue
    if count:
        print(f"  Patched {count} task.toml files (removed private docker_image references)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Nemotron-Terminal-Synthetic-Tasks for RL workload benchmarking.",
        epilog="Synthetic tasks for RL workload benchmarking.",
    )
    parser.add_argument(
        "dest",
        nargs="?",
        default="./nemotron_tasks",
        help="Destination directory (default: ./nemotron_tasks)",
    )
    parser.add_argument(
        "--domain",
        choices=DOMAINS,
        default=None,
        help="Download only one domain (default: all)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    targets = [args.domain] if args.domain else DOMAINS
    token = _check_token()

    if not token:
        print("  Note: No HF token found. Downloading without auth (may be rate-limited).")
        print("  To avoid rate limits: export HF_TOKEN=hf_... or run: huggingface-cli login")
        print()

    print(f"Downloading Nemotron-Terminal-Synthetic-Tasks to {dest}/")
    print(f"Domains: {', '.join(targets)}")
    print()

    downloaded = 0
    failed = 0
    skipped = 0

    for domain in targets:
        result = _download_domain(domain, dest, token)
        if result:
            extract_dir = dest / "skill_based" / "mixed" / domain
            if any(extract_dir.iterdir()):
                count = sum(1 for d in extract_dir.iterdir() if d.is_dir())
                if count > 0:
                    downloaded += 1
                else:
                    skipped += 1
            else:
                skipped += 1
        else:
            failed += 1
            if not token:
                # Auth/rate-limit error without token — stop trying
                print("\n  Stopping. Set HF_TOKEN and retry.")
                break

    print()
    print("=" * 50)
    print(f"  Downloaded: {downloaded} domain(s)")
    if skipped:
        print(f"  Skipped:    {skipped} domain(s) (already extracted)")
    if failed:
        print(f"  Failed:     {failed} domain(s)")
    print()

    # Show available domains
    any_found = False
    for domain in targets:
        extract_dir = dest / "skill_based" / "mixed" / domain
        if extract_dir.exists():
            count = sum(1 for d in extract_dir.iterdir() if d.is_dir())
            if count > 0:
                print(f"  {extract_dir}/ ({count} tasks)")
                any_found = True

    if any_found:
        first_domain = targets[0]
        first_dir = dest / "skill_based" / "mixed" / first_domain
        first_task = next((d.name for d in sorted(first_dir.iterdir()) if d.is_dir()), "<task_name>")
        print()
        print("Example commands:")
        print()
        print("  # Single task")
        print(
            f"  uv run tasktrace-capture --task {first_dir}/{first_task} --concurrency 1,2,4 --api-base http://host:8000/v1"
        )
        print()
        print("  # 10 tasks from domain")
        print(
            f"  uv run tasktrace-capture -d {first_dir} --n-tasks 10 --concurrency 1,4,16 --api-base http://host:8000/v1"
        )

    if failed:
        sys.exit(1)
