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

"""Resolve a cluster name to a validated cluster profile.

Cluster-aware commands resolve the profile before accessing Kubernetes. `KUBE_CONTEXT` pins every command to
the intended context; when it is absent, the current kubectl context is used and reachability is reported as
unpinned. Hardware compatibility is checked from recipe GPU type and architecture against profile values.

CLI:
  profile_resolver.py resolve <cluster> [--current-context CTX]
  profile_resolver.py list
  profile_resolver.py context <cluster>
  profile_resolver.py validate <cluster>
  profile_resolver.py compat <cluster> <cell-dir>
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = ROOT / "cluster-profiles"

# Resolution statuses.
OK = "OK"  # profile found + reachable (or no context pinned) → proceed
UNREACHABLE = "UNREACHABLE"  # profile found, KUBE_CONTEXT set, cluster not reachable → stop
NOT_FOUND = "NOT_FOUND"  # no profile by that name → stop, suggest init
AMBIGUOUS = "AMBIGUOUS"  # no --cluster, >1 candidate → ask the caller to pick
NONE = "NONE"  # no --cluster, nothing to auto-select → stop


# Current cluster-profile schema. A profile written by the wizard carries a
# `# LLMB_PROFILE_SCHEMA=<N>` header comment; an absent header = schema 0 (every profile written
# before the wizard shipped). Bumped only when a field is renamed/moved (see normalize_profile).
PROFILE_SCHEMA_VERSION = 1
SCHEMA_HEADER = "LLMB_PROFILE_SCHEMA"
import re as _re  # module-level so _SCHEMA_RE can compile once

_SCHEMA_RE = _re.compile(r"^#\s*LLMB_PROFILE_SCHEMA\s*=\s*(\d+)\s*$")


def normalize_profile(env: dict) -> dict:
    """Up-migrate a parsed profile dict to the CURRENT schema, in memory (the wizard contract §5, Q7).

    Shared by every verb (it lives in _read_env, not in the wizard) so old verbs stay migration-aware.
    An absent header = schema 0. Phase-1 has no field renames, so migration is a version-stamp
    passthrough — but the hook exists so a future rename can migrate old on-disk profiles in memory
    (e.g. `if ver < 2: env['NEW'] = env.pop('OLD', '')`) instead of breaking readers. Idempotent.
    """
    ver = 0
    raw = str(env.get(SCHEMA_HEADER, "") or "").strip()
    if raw.isdigit():
        ver = int(raw)
    # (future field-rename migrations go here, gated on `ver < N`, each bumping `ver`.)
    env[SCHEMA_HEADER] = str(max(ver, 0))
    return env


def _read_env(path: Path) -> dict:
    """Minimal .env parser (KEY=value; quoted or bare-with-inline-comment). Mirrors preflight.parse_env.

    Also reads the `# LLMB_PROFILE_SCHEMA=<N>` header comment (recorded under the SCHEMA_HEADER key) and
    runs normalize_profile() so every reader sees a current-schema dict (Q7)."""
    import re

    out: dict[str, str] = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        m = _SCHEMA_RE.match(ln)
        if m:
            out[SCHEMA_HEADER] = m.group(1)
            continue
        # `export KEY=value` assigns when the file is SOURCED — and deploy.sh does exactly that with the
        # same file. Without this, splitting on the first `=` yields a key literally named `export KEY`,
        # so the real key reads as absent and `profile validate` reports a var the operator plainly set.
        if ln.startswith("export "):
            ln = ln[len("export ") :].lstrip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'"):
            q = v[0]
            j = v.find(q, 1)
            v = v[1:j] if j != -1 else v[1:]
        else:
            v = re.split(r"\s+#", v, 1)[0].strip()
        out[k.strip()] = v
    return normalize_profile(out)


def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    """Names of real profiles: cluster-profiles/<name>.env, excluding *.example and _template*."""
    d = Path(profiles_dir)
    if not d.is_dir():
        return []
    names = []
    for p in d.glob("*.env"):
        if p.name.endswith(".example") or p.name.startswith("_template"):
            continue
        names.append(p.stem)
    return sorted(names)


def profile_env_path(name: str, profiles_dir: Path = PROFILES_DIR) -> Path:
    return Path(profiles_dir) / f"{name}.env"


def profile_context(name: str, profiles_dir: Path = PROFILES_DIR) -> str | None:
    """The KUBE_CONTEXT a profile pins kubectl to, or None if it doesn't pin one."""
    p = profile_env_path(name, profiles_dir)
    if not p.exists():
        return None
    ctx = _read_env(p).get("KUBE_CONTEXT", "").strip()
    return ctx or None


def default_probe(context: str) -> bool:
    """Reachability: can we reach the cluster behind this kube context? (fast, side-effect-free)."""
    try:
        p = subprocess.run(
            ["kubectl", "--context", context, "--request-timeout=5s", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=12,
        )
        return p.returncode == 0
    except Exception:
        return False


def connect_hint(profile: dict, context: str = "") -> str:
    """The EXACT copy/paste command to (re)authenticate to a cluster whose apiserver won't answer — the
    interactive SSO/Teleport/VPN login that can't be safely automated. Precedence: the profile's CONNECT_CMD
    (set verbatim by the operator at `init`; shown by `fleet` too) → a derived `tsh kube login <context>` when
    a context is pinned → a generic instruction. Never bakes credentials; it only echoes back the command the
    operator told us. Pure — shared by resolve() (UNREACHABLE) and preflight so the guidance can't drift.
    """
    cmd = (str(profile.get("CONNECT_CMD") or "")).strip()
    if cmd:
        return cmd
    ctx = (context or str(profile.get("KUBE_CONTEXT") or "")).strip()
    if ctx:
        return f"tsh kube login {ctx}    # or your cluster's SSO/VPN login — pin it as CONNECT_CMD in the profile"
    return (
        "re-authenticate to the cluster (tsh/Teleport/VPN login) — pin the exact command as CONNECT_CMD in the profile"
    )


def _norm_gpu(s: str) -> str:
    """Normalize a GPU name so recipe `gpu_type` and profile `GPU_PRODUCT` compare cleanly:
    'NVIDIA-GB200' → 'GB200', 'GB200' → 'GB200'."""
    return (s or "").strip().upper().replace("NVIDIA-", "").replace("NVIDIA_", "").replace(" ", "")


def profile_gpu_type(profile: dict) -> str:
    """The GPU a profile's cluster provides. Prefer an explicit GPU_TYPE override; else derive from
    GPU_PRODUCT (the node label). Escape hatch for clusters whose product label isn't the standard form.
    """
    if (profile.get("GPU_TYPE") or "").strip():
        return _norm_gpu(profile["GPU_TYPE"])
    return _norm_gpu(profile.get("GPU_PRODUCT", ""))


def check_target_compat(envelope: dict, profile: dict) -> list[str]:
    """Static (no cluster) guard against running a recipe on the wrong hardware target.

    A recipe carries hardware-specific serving flags (e.g. GB-class arm64 needs
    `--disable-custom-all-reduce`); running it on the wrong GPU is a silently-invalid result at best and a
    crash at worst. The live arch guard can't distinguish GB200 from GB300 (both arm64), so we compare the
    recipe's declared target to the cluster's declared target directly. Returns a list of blocking issues
    (empty = compatible)."""
    issues: list[str] = []
    want_gpu = _norm_gpu(envelope.get("gpu_type", ""))
    have_gpu = profile_gpu_type(profile)
    if want_gpu and have_gpu and want_gpu != have_gpu:
        issues.append(
            f"GPU target mismatch — recipe is built for {envelope.get('gpu_type')} but the cluster provides "
            f"{profile.get('GPU_PRODUCT') or have_gpu}. Recipe serving flags may be hardware-specific."
        )

    want_arch = (envelope.get("arch") or "").strip()
    have_arch = (profile.get("ARCH") or "").strip()
    if want_arch and have_arch and want_arch != have_arch:
        issues.append(f"arch mismatch — recipe arch={want_arch} but cluster ARCH={have_arch}.")

    return issues


def compat_message(issues: list[str], cluster: str) -> str:
    body = "\n".join(f"    ✗ {i}" for i in issues)
    return (
        f"── Target compatibility ({cluster}) ──────────────────────────────\n{body}\n\n"
        f"  This recipe targets different hardware than '{cluster}'. Pick a matching cluster, or a\n"
        f"  recipe built for this one:  llmb-k8s status --cluster {cluster}"
    )


# Profile completeness. Required = the deploy will silently fail to schedule/mount without these.
# Recommended = optional but they unlock reachability checks / target-compat / downloads.
REQUIRED_PROFILE_VARS = [
    "NAMESPACE",
    "GPU_PRODUCT",
    "IMAGE_PULL_SECRET",
    "MODEL_CACHE_PVC",
]
# ...except MODEL_CACHE_PVC is satisfied by ANY per-model key too: a cluster whose every model has its own
# claim (MODEL_CACHE_PVC_<MODEL>=...) is fully configured and must not fail `profile validate` for lacking
# a cluster-wide default. install's per-cell gate is what actually decides; this is only completeness.
_CACHE_KEY_PREFIX = "MODEL_CACHE_PVC_"


def _cache_configured(prof: dict) -> bool:
    """PURE — does this profile name a model-cache claim at all (default or per-model)?"""
    return bool(
        (prof.get("MODEL_CACHE_PVC") or "").strip()
        or any(k.startswith(_CACHE_KEY_PREFIX) and (v or "").strip() for k, v in prof.items())
    )


RECOMMENDED_PROFILE_VARS = [
    "KUBE_CONTEXT",
    "ARCH",
    "ARTIFACTS_STORAGE_CLASS",
    "HF_SECRET",
]


def validate_profile(name: str, *, profiles_dir: Path = PROFILES_DIR, probe=default_probe) -> tuple[bool, list[str]]:
    """Is a profile complete and (if it pins KUBE_CONTEXT) reachable? Returns (ok, lines).
    Pure given `probe`. Missing REQUIRED vars or an unreachable pinned context → not ok.
    """
    if name not in list_profiles(profiles_dir):
        return False, [
            f"✗ no profile '{name}' — create one:  llmb-k8s init --cluster {name}  "
            f"(scriptable engine: llmb-k8s profile init --cluster {name})"
        ]
    env = _read_env(profile_env_path(name, profiles_dir))
    lines: list[str] = []
    ok = True
    missing = [v for v in REQUIRED_PROFILE_VARS if not (env.get(v) or "").strip()]
    # A profile that gives every model its own claim (MODEL_CACHE_PVC_<MODEL>=...) and no cluster-wide
    # default is fully configured — don't report the default as missing.
    if "MODEL_CACHE_PVC" in missing and _cache_configured(env):
        missing.remove("MODEL_CACHE_PVC")
    if missing:
        ok = False
        lines.append(f"✗ missing required vars: {', '.join(missing)}  (deploy would silently fail to schedule/mount)")
    else:
        lines.append(f"✓ required vars present: {', '.join(REQUIRED_PROFILE_VARS)}")
    rec_missing = [v for v in RECOMMENDED_PROFILE_VARS if not (env.get(v) or "").strip()]
    if rec_missing:
        lines.append(f"• recommended vars unset (optional): {', '.join(rec_missing)}")
    ctx = (env.get("KUBE_CONTEXT") or "").strip()
    if not ctx:
        lines.append("• no KUBE_CONTEXT pinned — reachability unknown (runs on the current kubectl context)")
    elif probe(ctx):
        lines.append(f"✓ context {ctx} is reachable")
    else:
        ok = False
        lines.append(f"✗ context {ctx} unreachable — VPN/Teleport? (kubectl config get-contexts)")
    return ok, lines


@dataclass
class Resolution:
    status: str
    name: str | None
    env_path: Path | None
    context: str | None
    message: str

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def exit_code(self) -> int:
        return 0 if self.status == OK else (2 if self.status in (AMBIGUOUS, NONE) else 1)


def _panel(title: str, rows: list[tuple[str, str]]) -> str:
    bar = "─" * max(4, 66 - len(title))
    out = [f"── {title} {bar}"]
    for label, val in rows:
        out.append(f"  {label:<12} {val}")
    return "\n".join(out)


def resolve(
    cluster: str | None,
    *,
    profiles_dir: Path = PROFILES_DIR,
    current_context: str | None = None,
    probe=default_probe,
) -> Resolution:
    """Resolve a cluster name (or auto-select) to a profile. Pure given `probe`. See module docstring."""
    profiles = list_profiles(profiles_dir)

    if cluster:
        if cluster not in profiles:
            existing = ", ".join(profiles) if profiles else "(none yet)"
            msg = (
                f"  No profile found for '{cluster}'.\n\n"
                f"  Create one (front door):\n    llmb-k8s init --cluster {cluster}\n"
                f"  (or the scriptable engine:  llmb-k8s profile init --cluster {cluster})\n\n"
                f"  Existing profiles:  {existing}"
            )
            return Resolution(NOT_FOUND, None, None, None, msg)

        env = profile_env_path(cluster, profiles_dir)
        ctx = profile_context(cluster, profiles_dir)
        if ctx is None:
            msg = _panel(
                f"Profile: {cluster}",
                [
                    (
                        "Context",
                        "(none pinned — proceeding on current kubectl context)",
                    ),
                    ("Profile", str(env)),
                ],
            ) + (
                "\n\n  Tip: set KUBE_CONTEXT in the profile to pin kubectl to this cluster "
                "(safe for concurrent multi-cluster use)."
            )
            return Resolution(OK, cluster, env, None, msg)

        if probe(ctx):
            return Resolution(
                OK,
                cluster,
                env,
                ctx,
                _panel(
                    f"Profile: {cluster}",
                    [
                        ("Context", f"{ctx}   ✓ reachable"),
                        ("Profile", str(env)),
                    ],
                ),
            )

        # Detect-and-guide (SSO/Teleport can't be auto-logged-in safely): print the EXACT copy/paste login
        # command up front — from the profile's CONNECT_CMD, or a derived `tsh kube login <ctx>` — so an
        # expired session is a one-line fix, not a downstream kubectl auth error.
        login = connect_hint(_read_env(env), ctx)
        msg = _panel(f"Profile: {cluster}", [("Context", f"{ctx}   ✗ unreachable")]) + (
            f"\n\n  Cannot reach {ctx} (VPN down · SSO/Teleport session expired · context misconfigured).\n"
            f"  Log in, then retry:\n    {login}\n\n"
            f"  Re-check after reconnecting:\n    llmb-k8s profile validate --cluster {cluster}"
        )
        return Resolution(UNREACHABLE, cluster, env, ctx, msg)

    # No --cluster: try to auto-select from the current kubectl context.
    if current_context:
        matches = [p for p in profiles if profile_context(p, profiles_dir) == current_context]
        if len(matches) == 1:
            name = matches[0]
            env = profile_env_path(name, profiles_dir)
            msg = (
                f"  No --cluster given. Current context {current_context} matches profile {name}.\n"
                f"  Using {name}. Pass --cluster <name> to override."
            )
            return Resolution(OK, name, env, current_context, msg)

    if not profiles:
        return Resolution(
            NONE,
            None,
            None,
            None,
            "  No cluster profiles found.\n  Create one:\n    llmb-k8s profile init --cluster <name>",
        )

    rows = "\n".join(f"    {p:<20} {profile_context(p, profiles_dir) or '(no context pinned)'}" for p in profiles)
    msg = (
        f"  No --cluster given and no unambiguous match for the current context.\n\n"
        f"  Available profiles:\n{rows}\n\n  Re-run with:  llmb-k8s <verb> --cluster <name> ..."
    )
    return Resolution(AMBIGUOUS, None, None, None, msg)


def _current_context() -> str | None:
    try:
        p = subprocess.run(
            ["kubectl", "config", "current-context"],
            capture_output=True,
            text=True,
            timeout=8,
        )
        return p.stdout.strip() or None if p.returncode == 0 else None
    except Exception:
        return None


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "list":
        for n in list_profiles():
            print(n)
        return 0
    if cmd == "context":
        if len(argv) < 2:
            sys.exit("usage: profile_resolver.py context <cluster>")
        print(profile_context(argv[1]) or "")
        return 0
    if cmd == "validate":
        if len(argv) < 2:
            sys.exit("usage: profile_resolver.py validate <cluster> [--fast] [--recipe <cell-dir>]")
        # Parse extra flags: --fast skips the LIVE readiness battery (reachability-only), --recipe scopes the
        # pull-secret check to a specific recipe's images (per-image org access, not just registry auth).
        cluster = None
        fast = False
        recipe_cell = None
        rest = argv[1:]
        i = 0
        while i < len(rest):
            a = rest[i]
            if a == "--fast":
                fast = True
                i += 1
                continue
            if a == "--recipe" and i + 1 < len(rest):
                recipe_cell = rest[i + 1]
                i += 2
                continue
            if not a.startswith("-") and cluster is None:
                cluster = a
            i += 1
        if not cluster:
            sys.exit("usage: profile_resolver.py validate <cluster> [--fast] [--recipe <cell-dir>]")
        ok, lines = validate_profile(cluster)
        title = f"profile: {cluster}"
        print(f"── {title} {'─' * max(4, 64 - len(title))}")
        for l in lines:
            print(f"  {l}")
        # An unpinned profile is complete but NOT unconditionally "ready to run" — it targets whatever the
        # ambient kubectl context happens to be, which in a multi-cluster shell is the wrong cluster (GB200
        # round-2). Don't print a clean bill of health; name the risk. Still exit 0 — legacy single-cluster
        # profiles pin their cluster via tsh/KUBE_CLUSTER, so unpinned is a caveat, not a hard failure.
        pinned = bool((_read_env(profile_env_path(cluster)).get("KUBE_CONTEXT") or "").strip())
        if not ok:
            print("  → INCOMPLETE/UNREACHABLE — fix the ✗ items above (skipping live readiness probes)")
            return 1
        if not pinned:
            print(
                "  → completeness OK, but UNPINNED — will run on your CURRENT kubectl context, not a "
                "profile-pinned one. In a multi-cluster shell this can hit the WRONG cluster; set "
                "KUBE_CONTEXT in the profile before any live/mutating verb (deploy/run)."
            )
        # ── DEFAULT-ROBUST readiness battery (cluster self-test) ──────────────────────────────
        # Complete and reachable are necessary but not sufficient. Also verify image pulls,
        # artifact storage, and file transfer before scheduling GPU workloads. Run the
        # no-GPU readiness checks by default; --fast skips them.
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import cluster_readiness as _cr

            prof = _read_env(profile_env_path(cluster))
            recipe = None
            if recipe_cell:
                import yaml

                recipe = yaml.safe_load((Path(recipe_cell) / "recipe.yaml").read_text()) or {}
            checks = _cr.run_battery(prof, recipe=recipe, recipe_cell=recipe_cell, fast=fast)
            print()
            print(_cr.format_battery(cluster, checks))
            battery_ok, _ = _cr.verdict(checks)
            return 0 if battery_ok else 1
        except Exception as _e:
            # The battery must never turn a fine cluster into a hard failure on its own bug — safe-degrade.
            print(f"\n  · readiness battery skipped ({_e}) — completeness+reachability verdict stands")
            return 0
    if cmd == "compat":
        if len(argv) < 3:
            sys.exit("usage: profile_resolver.py compat <cluster> <cell-dir>")
        cluster, cell = argv[1], argv[2]
        import yaml  # optional dep; only needed for this subcommand

        env = (yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}).get("envelope") or {}
        issues = check_target_compat(env, _read_env(profile_env_path(cluster)))
        if issues:
            print(compat_message(issues, cluster))
            return 1
        print(f"  ✓ recipe target {env.get('gpu_type', '?')} is compatible with cluster {cluster}")
        return 0
    if cmd == "resolve":
        cluster = None
        current = None
        rest = argv[1:]
        i = 0
        while i < len(rest):
            if rest[i] == "--current-context" and i + 1 < len(rest):
                current = rest[i + 1]
                i += 2
                continue
            if not rest[i].startswith("-") and cluster is None:
                cluster = rest[i]
            i += 1
        if current is None:
            current = _current_context()
        r = resolve(cluster, current_context=current)
        print(r.message)
        return r.exit_code
    sys.exit(f"profile_resolver.py: unknown command '{cmd}' (resolve | list | context)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
