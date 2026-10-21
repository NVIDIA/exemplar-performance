# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace downloads and offline preparation.

This module provides functions to download and verify HuggingFace assets
for offline use during workload execution.
"""

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, TypeVar

from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Rate limit retry defaults
_MAX_RETRIES = 5
_INITIAL_BACKOFF_SECONDS = 30
_MAX_BACKOFF_SECONDS = 300


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Check if an exception (or any exception in its chain) indicates a 429 rate limit.

    HuggingFace often wraps HTTP errors inside higher-level exceptions like
    LocalEntryNotFoundError. The top-level message may say "model or snapshot
    folder unavailable" while the actual 429 is buried in __cause__ or __context__.
    This function walks the full chain to catch those cases.
    """
    seen: Set[int] = set()
    stack: List[BaseException] = [exc]

    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))

        # Check response.status_code (requests/httpx style exceptions)
        resp = getattr(cur, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            if status == 429:
                return True

        # Check string representation as fallback
        msg = str(cur).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True

        # Walk the exception chain
        if cur.__cause__ is not None:
            stack.append(cur.__cause__)
        if cur.__context__ is not None:
            stack.append(cur.__context__)

    return False


def _format_exception_chain(exc: BaseException, max_depth: int = 5) -> str:
    """Format an exception chain into a compact, readable string.

    Useful for logging HuggingFace errors where the real cause (e.g., 429, 401)
    is buried under wrapper exceptions.
    """
    parts: List[str] = []
    cur: Optional[BaseException] = exc
    depth = 0
    seen: Set[int] = set()

    while cur is not None and depth < max_depth and id(cur) not in seen:
        seen.add(id(cur))

        # Include HTTP status if available
        resp = getattr(cur, "response", None)
        status = getattr(resp, "status_code", None) if resp else None
        status_str = f" [HTTP {status}]" if status else ""

        # Truncate long messages
        msg = str(cur).strip()
        if len(msg) > 200:
            msg = msg[:200] + "..."

        parts.append(f"{cur.__class__.__name__}{status_str}: {msg}" if msg else f"{cur.__class__.__name__}{status_str}")

        cur = cur.__cause__ or cur.__context__
        depth += 1

    return " → ".join(parts) if parts else "Unknown error"


def _with_retry(
    fn: Callable[[], T],
    operation: str,
    max_retries: int = _MAX_RETRIES,
    initial_backoff: int = _INITIAL_BACKOFF_SECONDS,
    max_backoff: int = _MAX_BACKOFF_SECONDS,
) -> T:
    """Execute a HuggingFace operation with retry logic for 429 rate limits.

    HuggingFace applies IP-based rate limits before checking authentication,
    so even valid tokens can hit 429 on shared IPs (e.g., cluster environments).
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise

            last_exception = e
            if attempt < max_retries - 1:
                wait_time = min(initial_backoff * (2**attempt), max_backoff)
                wait_time = round(wait_time * (0.8 + random.random() * 0.4))  # Add jitter (80-120%)
                logger.warning(
                    f"HuggingFace {operation} rate limited (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time}s..."
                )
                print(
                    f"  ⏳ Rate limited during {operation}. Waiting {wait_time}s (attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)

    raise RuntimeError(
        f"HuggingFace {operation} failed after {max_retries} retries due to rate limiting. "
        f"Try again later or from a different network. Last error: {last_exception}"
    )


# Deterministic HuggingFace download patterns (non-weight files only)
# NOTE: *.py is required for models with custom code (e.g., DeepSeek V3, Qwen3)
# that use auto_map in config.json to reference custom Python modules like
# configuration_*.py and modeling_*.py. Without these, trust_remote_code=True
# will fail in offline mode even though trust_remote_code=False verification passes.
HF_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.yaml",
    "*.md",
    "*.py",
]

# Some tokenizers reference chat templates stored outside tokenizer_config.json.
# Keep these patterns scoped to tokenizer assets so config-only cache entries do
# not download unrelated chat-template files.
HF_TOKENIZER_ALLOW_PATTERNS = [
    "*.jinja",
    "chat_template*",
    "chat_templates/*",
]

HF_IGNORE_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.ckpt",
    "*.pth",
    "*.onnx",
    "*.engine",
    "*.plan",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
]

HF_ASSET_TYPES = {"tokenizer", "config"}
HF_REPO_TYPES = {"model", "dataset"}
HF_REPO_TARGETS = {"workload", "shared"}

_CACHE_ITEM_KEYS = {"repo_id", "revision", "assets"}
_REPO_ITEM_KEYS = {"repo_id", "repo_type", "revision", "target", "name", "include", "exclude", "when"}


@dataclass(frozen=True)
class HuggingFaceAssetSpec:
    """A shared-cache download: tokenizer/config files into $LLMB_INSTALL/.cache/huggingface."""

    repo_id: str
    revision: Optional[str] = None
    assets: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class HuggingFaceRepoSpec:
    """A repo-contents download into an LLMB-managed workload or shared dataset directory."""

    workload_key: str
    repo_id: str
    repo_type: str
    revision: Optional[str] = None
    target: str = "workload"
    name: str = ""
    include: Optional[Tuple[str, ...]] = None
    exclude: Optional[Tuple[str, ...]] = None

    def source_key(self) -> Tuple[Any, ...]:
        return (
            self.repo_id,
            self.repo_type,
            self.revision,
            self.include,
            self.exclude,
        )


@dataclass(frozen=True)
class HuggingFaceDownloadPlan:
    cache: Tuple[HuggingFaceAssetSpec, ...] = ()
    repos: Tuple[HuggingFaceRepoSpec, ...] = ()

    def empty(self) -> bool:
        return not self.cache and not self.repos


def _normalize_hf_token(token: str) -> Optional[str]:
    """Normalize a HuggingFace token for reliable HTTP auth.

    Tokens from files or environment variables often have trailing newlines,
    whitespace, or accidental quoting that causes auth failures with cryptic
    errors like "Invalid header value".

    Returns:
        Normalized token, or None if token is empty/whitespace-only.
    """
    t = token.strip()
    # Strip common accidental quoting
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        t = t[1:-1].strip()
    return t or None


# Conservative HuggingFace download parallelism. Peak download memory scales with total
# concurrent Xet streams (max_workers x per-file concurrency), so we pin both low to stay
# under tight per-job/per-user memory caps (`ulimit -v` or cgroup `memory.max`). Raise on
# memory-rich hosts: LLMB_HF_MAX_WORKERS (files) and HF_XET_FIXED_DOWNLOAD_CONCURRENCY (streams).
_DEFAULT_HF_MAX_WORKERS = 4


def _hf_max_workers() -> int:
    """Max number of concurrent download streams. Override with LLMB_HF_MAX_WORKERS."""
    raw = os.environ.get("LLMB_HF_MAX_WORKERS")
    if raw is None:
        return _DEFAULT_HF_MAX_WORKERS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid LLMB_HF_MAX_WORKERS=%r (expected integer); using %d", raw, _DEFAULT_HF_MAX_WORKERS
        )
        return _DEFAULT_HF_MAX_WORKERS
    if value < 1:
        logger.warning("Ignoring LLMB_HF_MAX_WORKERS=%d (must be >= 1); using %d", value, _DEFAULT_HF_MAX_WORKERS)
        return _DEFAULT_HF_MAX_WORKERS
    return value


def set_hf_environment(cache_dir: str) -> None:
    """Set HuggingFace environment variables.

    CRITICAL: Must be called before any 'import transformers' or 'import huggingface_hub'
    statements, as HF_HOME is read and cached at module import time.

    Args:
        cache_dir: Base cache directory (e.g., $LLMB_INSTALL/.cache/huggingface)
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
    # Bound peak download memory (see _DEFAULT_HF_MAX_WORKERS); setdefault keeps user overrides.
    os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "1")


def _hf_hub_cache_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, "hub")


def _check_item_keys(item: Dict[str, Any], allowed: Set[str], where: str) -> None:
    unknown = set(item) - allowed
    if not unknown:
        return
    hint = ""
    if allowed == _CACHE_ITEM_KEYS and unknown & (_REPO_ITEM_KEYS | {"snapshot"}):
        hint = " Repo-contents downloads belong in 'downloads.huggingface.repos'."
    raise ValueError(f"{where} entry has unsupported key(s) {sorted(unknown)}. Allowed keys: {sorted(allowed)}.{hint}")


def _parse_repo_id(item: Dict[str, Any], where: str) -> str:
    repo_id = item.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"{where} entry has invalid 'repo_id': expected non-empty string.")
    return repo_id


def _parse_repo_type(item: Dict[str, Any], repo_id: str, where: str) -> str:
    repo_type = item.get("repo_type")
    if repo_type is None:
        raise ValueError(
            f"{where} entry for repo '{repo_id}' is missing required 'repo_type'. "
            f"Allowed values are 'model' and 'dataset'."
        )
    if not isinstance(repo_type, str) or repo_type not in HF_REPO_TYPES:
        raise ValueError(
            f"{where} entry for repo '{repo_id}' has invalid 'repo_type' value '{repo_type}'. "
            f"Allowed values are 'model' and 'dataset'."
        )
    return repo_type


def _parse_revision(item: Dict[str, Any], where: str) -> Optional[str]:
    revision = item.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError(
            f"{where} entry has invalid 'revision': expected string or null, got {type(revision).__name__}."
        )
    return revision


def _parse_assets(item: Dict[str, Any], where: str) -> FrozenSet[str]:
    assets = item.get("assets")
    if assets is None:
        return frozenset({"tokenizer", "config"})
    if not isinstance(assets, list):
        raise ValueError(f"{where} entry has invalid 'assets': expected list, got {type(assets).__name__}.")
    if not assets:
        raise ValueError(f"{where} entry has invalid 'assets': list must not be empty.")
    asset_set = set()
    for asset in assets:
        if not isinstance(asset, str):
            raise ValueError(f"{where} entry has invalid 'assets' entry: expected string, got {type(asset).__name__}.")
        if asset not in HF_ASSET_TYPES:
            raise ValueError(
                f"{where} entry has invalid 'assets' value '{asset}'. Allowed values are 'tokenizer' and 'config'."
            )
        asset_set.add(asset)
    return frozenset(asset_set)


def _repo_basename(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def _parse_patterns(value: Any, field_name: str, where: str) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{where} entry has invalid '{field_name}': expected list, got {type(value).__name__}.")
    patterns = []
    for pattern in value:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"{where} entry has invalid '{field_name}' entry: expected non-empty string.")
        patterns.append(pattern)
    return tuple(patterns)


def _parse_when_gpu(item: Dict[str, Any], where: str) -> Optional[List[str]]:
    when = item.get("when")
    if when is None:
        return None
    if not isinstance(when, dict):
        raise ValueError(f"{where} entry has invalid 'when': expected dict, got {type(when).__name__}.")
    unknown = set(when) - {"gpu"}
    if unknown:
        raise ValueError(f"{where} entry has unsupported 'when' key(s) {sorted(unknown)}. Allowed keys: ['gpu'].")
    when_gpu = when.get("gpu")
    if when_gpu is None:
        return None
    if not isinstance(when_gpu, list) or not when_gpu:
        raise ValueError(f"{where} entry has invalid 'when.gpu': expected non-empty list.")
    if not all(isinstance(value, str) and value for value in when_gpu):
        raise ValueError(f"{where} entry has invalid 'when.gpu' entry: expected non-empty string.")
    return when_gpu


def _parse_cache_item(item: Dict[str, Any], where: str) -> Tuple[str, Optional[str], FrozenSet[str]]:
    if not isinstance(item, dict):
        raise ValueError(f"{where} entry is invalid: expected dict, got {type(item).__name__}.")
    _check_item_keys(item, _CACHE_ITEM_KEYS, where)
    repo_id = _parse_repo_id(item, where)
    revision = _parse_revision(item, where)
    return repo_id, revision, _parse_assets(item, where)


def _parse_repo_item(
    item: Dict[str, Any], workload_key: str, where: str
) -> Tuple[HuggingFaceRepoSpec, Optional[List[str]]]:
    """Parse a repo-contents download entry.

    The entry is fully validated before the 'when' filter is evaluated by the
    caller, so a malformed entry fails on every GPU type, not just matching ones.
    """
    if not isinstance(item, dict):
        raise ValueError(f"{where} entry is invalid: expected dict, got {type(item).__name__}.")
    _check_item_keys(item, _REPO_ITEM_KEYS, where)
    repo_id = _parse_repo_id(item, where)
    repo_type = _parse_repo_type(item, repo_id, where)
    revision = _parse_revision(item, where)

    target = item.get("target", "workload")
    if target is None:
        target = "workload"
    if not isinstance(target, str) or target not in HF_REPO_TARGETS:
        raise ValueError(
            f"{where} entry for repo '{repo_id}' has invalid 'target' value '{target}'. "
            f"Allowed values are 'workload' and 'shared'."
        )

    name = item.get("name", _repo_basename(repo_id))
    if name is None:
        name = _repo_basename(repo_id)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where} entry for repo '{repo_id}' has invalid 'name': expected non-empty string.")

    spec = HuggingFaceRepoSpec(
        workload_key=workload_key,
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        target=target,
        name=name,
        include=_parse_patterns(item.get("include"), "include", where),
        exclude=_parse_patterns(item.get("exclude"), "exclude", where),
    )
    return spec, _parse_when_gpu(item, where)


def _split_huggingface_section(section: Any, where: str) -> Tuple[List[Any], List[Any]]:
    """Split downloads.huggingface into (cache_items, repo_items).

    The bare list form is the released shorthand for cache entries; the map form
    holds explicit 'cache' and 'repos' lists.
    """
    if isinstance(section, list):
        return section, []
    if isinstance(section, dict):
        unknown = set(section) - {"cache", "repos"}
        if unknown:
            raise ValueError(f"{where} has unsupported key(s) {sorted(unknown)}. Allowed keys: ['cache', 'repos'].")
        cache_items = section.get("cache") or []
        repo_items = section.get("repos") or []
        if not isinstance(cache_items, list):
            raise ValueError(f"{where}.cache is invalid: expected list, got {type(cache_items).__name__}.")
        if not isinstance(repo_items, list):
            raise ValueError(f"{where}.repos is invalid: expected list, got {type(repo_items).__name__}.")
        return cache_items, repo_items
    raise ValueError(
        f"{where} is invalid: expected list or dict, got {type(section).__name__}. "
        f"Check metadata.yaml - should be:\n"
        f"downloads:\n"
        f"  huggingface:\n"
        f"    - repo_id: 'model/name'\n"
        f"      assets: [tokenizer, config]"
    )


def build_hf_download_plan(
    workloads: Dict[str, Dict[str, Any]], selected_keys: List[str], gpu_type: Optional[str] = None
) -> HuggingFaceDownloadPlan:
    """Build a normalized HuggingFace download plan from workload metadata."""
    asset_requirements: Dict[Tuple[str, Optional[str]], Set[str]] = {}
    repo_specs: Dict[HuggingFaceRepoSpec, None] = {}

    logger.debug(f"Checking {len(selected_keys)} workloads for HF downloads: {selected_keys}")

    for key in selected_keys:
        workload = workloads.get(key)
        if not workload:
            logger.warning(f"Workload '{key}' not found")
            continue

        downloads = workload.get("downloads", {}) or {}

        if not isinstance(downloads, dict):
            raise ValueError(
                f"Workload '{key}' has invalid 'downloads' section: expected dict, got {type(downloads).__name__}. "
                f"Check metadata.yaml - use proper dict structure or omit 'downloads' entirely."
            )

        if "huggingface" in downloads:
            section = downloads.get("huggingface")
            logger.debug(f"Workload '{key}': downloads.huggingface = {section}")
            where = f"Workload '{key}' 'downloads.huggingface'"
            cache_items, repo_items = _split_huggingface_section(section, where)
            cache_where = where if isinstance(section, list) else f"{where}.cache"

            for item in cache_items:
                repo_id, revision, assets = _parse_cache_item(item, cache_where)
                asset_requirements.setdefault((repo_id, revision), set()).update(assets)

            for item in repo_items:
                spec, when_gpu = _parse_repo_item(item, key, f"{where}.repos")
                if when_gpu is not None and gpu_type is not None and gpu_type not in when_gpu:
                    logger.debug(
                        "Skipping HuggingFace repo download for workload '%s' repo '%s' on GPU '%s' (allowed: %s)",
                        key,
                        spec.repo_id,
                        gpu_type,
                        when_gpu,
                    )
                    continue
                repo_specs.setdefault(spec)

    # Normalize optional fields (revision/include/exclude may be None) before sorting:
    # mixing None and str/tuple in a sort key raises TypeError. This happens whenever two
    # entries share a repo_id with inconsistent revision pinning or include/exclude patterns.
    cache = tuple(
        HuggingFaceAssetSpec(repo_id=repo_id, revision=revision, assets=frozenset(assets))
        for (repo_id, revision), assets in sorted(asset_requirements.items(), key=lambda kv: (kv[0][0], kv[0][1] or ""))
    )
    repos = tuple(
        sorted(
            repo_specs,
            key=lambda spec: (
                spec.workload_key,
                spec.repo_id,
                spec.repo_type,
                spec.revision or "",
                spec.target,
                spec.name,
                spec.include or (),
                spec.exclude or (),
            ),
        )
    )

    logger.debug("Total unique HF cache asset specs found: %d", len(cache))
    logger.debug("Total unique HF repo download specs found: %d", len(repos))
    return HuggingFaceDownloadPlan(cache=cache, repos=repos)


def download_huggingface_cache_assets(
    asset_specs: Tuple[HuggingFaceAssetSpec, ...], cache_dir: str, token: Optional[str] = None
) -> List[HuggingFaceAssetSpec]:
    """Download HuggingFace asset repos deterministically (non-weight files only).

    Uses snapshot_download with fixed allow/ignore patterns. No verification is
    performed in this phase.

    Note: On failure, successfully downloaded repos remain in the cache. This is
    intentional - snapshot_download is idempotent, so retrying skips completed
    downloads and resumes faster (especially useful when rate-limited).
    """
    if not asset_specs:
        logger.debug("No HuggingFace repos required for download phase")
        return []

    os.makedirs(cache_dir, exist_ok=True)
    set_hf_environment(cache_dir)

    # Normalize token to handle common issues (trailing newlines, whitespace, quotes)
    # that cause cryptic "Invalid header value" errors.
    normalized_token = _normalize_hf_token(token) if token else None

    if token and not normalized_token:
        logger.warning("HF token provided but empty after normalization - treating as no token")

    if normalized_token:
        # Set HF_TOKEN env var for downstream code (e.g., transformers.AutoTokenizer)
        # that uses implicit token resolution. The explicit token= parameter we pass
        # to snapshot_download takes precedence, but this covers other HF calls.
        os.environ["HF_TOKEN"] = normalized_token
        logger.debug(f"Authenticating with HuggingFace (token: {_obfuscate_token(normalized_token)})")
    else:
        logger.debug("No HF token provided - downloads may be rate limited and gated repos will fail")

    # Import HF libs only after HF_HOME/HF_HUB_CACHE/HF_TOKEN are set.
    from huggingface_hub import snapshot_download

    specs = tuple(sorted(asset_specs, key=lambda item: (item.repo_id, item.revision or "")))

    print("\nDownloading HuggingFace files")
    print("--------------------------------")
    print("\nRequired HuggingFace repos:")
    for spec in specs:
        assets = ", ".join(sorted(spec.assets))
        suffix = f" ({assets})" if assets else ""
        revision = f" @ {spec.revision}" if spec.revision else ""
        print(f"  - {spec.repo_id}{revision}{suffix}")

    print("\nDownloading non-weight files...")
    print("(Progress bars are expected)")

    successful = []
    total = len(specs)
    for idx, spec in enumerate(specs, 1):
        repo_id = spec.repo_id
        logger.debug(f"Snapshotting HuggingFace repo {idx}/{total}: {repo_id}")
        print(f"\n[{idx}/{total}] {repo_id}")
        try:
            allow_patterns = list(HF_ALLOW_PATTERNS)
            if "tokenizer" in spec.assets:
                allow_patterns.extend(HF_TOKENIZER_ALLOW_PATTERNS)
            snapshot_path = _with_retry(
                lambda hf_spec=spec, tok=normalized_token, patterns=allow_patterns: snapshot_download(
                    repo_id=hf_spec.repo_id,
                    revision=hf_spec.revision,
                    allow_patterns=patterns,
                    ignore_patterns=HF_IGNORE_PATTERNS,
                    cache_dir=_hf_hub_cache_dir(cache_dir),
                    token=tok,
                    max_workers=_hf_max_workers(),
                ),
                operation=f"snapshot '{repo_id}'",
            )
        except Exception as exc:
            # Provide actionable error: HF often wraps 401/403/429 in generic errors
            chain = _format_exception_chain(exc)
            logger.error("HuggingFace snapshot failed for '%s': %s", repo_id, chain)
            raise RuntimeError(f"HuggingFace snapshot failed for '{repo_id}'. Error chain: {chain}") from exc
        _maybe_inject_nemotron_config(
            repo_id=repo_id,
            assets=set(spec.assets),
            snapshot_path=snapshot_path,
        )
        # Tokenizer finalization may download additional files (e.g., custom code)
        # and can hit rate limits independently of snapshot_download.
        _with_retry(
            lambda hf_spec=spec, tok=normalized_token, path=snapshot_path: _maybe_finalize_tokenizer_snapshot(
                repo_id=hf_spec.repo_id,
                assets=set(hf_spec.assets),
                snapshot_path=path,
                token=tok,
            ),
            operation=f"tokenizer finalization '{repo_id}'",
        )
        successful.append(spec)

    print(f"\nSuccessfully downloaded {len(successful)} repo(s).")
    print("Download phase complete.")
    return successful


def resolve_huggingface_repo_local_dir(spec: HuggingFaceRepoSpec, install_path: str) -> str:
    """Resolve a repo download destination under LLMB-managed paths."""
    if spec.target == "workload":
        base = Path(install_path) / "workloads" / spec.workload_key
    elif spec.target == "shared":
        base = Path(install_path) / "datasets"
    else:
        raise ValueError(
            f"Invalid HuggingFace repo download target '{spec.target}'. Allowed values are 'workload' and 'shared'."
        )

    base_resolved = base.resolve()
    destination = (base / spec.name).resolve()
    if destination == base_resolved:
        raise ValueError(
            f"Invalid HuggingFace repo download name '{spec.name}' for workload '{spec.workload_key}': "
            f"destination must be a subdirectory of {base_resolved}, not the directory itself."
        )
    try:
        destination.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(
            f"Invalid HuggingFace repo download name '{spec.name}' for workload '{spec.workload_key}': "
            f"destination must stay under {base_resolved}."
        ) from exc
    return str(destination)


def _dedup_and_validate_repo_destinations(
    repo_specs: Tuple[HuggingFaceRepoSpec, ...], install_path: str
) -> Tuple[HuggingFaceRepoSpec, ...]:
    """Collapse specs resolving to the same destination; reject conflicting sources."""
    by_destination: Dict[str, HuggingFaceRepoSpec] = {}
    for spec in repo_specs:
        destination = resolve_huggingface_repo_local_dir(spec, install_path)
        existing = by_destination.get(destination)
        if existing is None:
            by_destination[destination] = spec
        elif existing.source_key() != spec.source_key():
            raise ValueError(
                "Conflicting HuggingFace repo downloads resolve to the same destination: "
                f"{destination}. Existing source is {existing.repo_id}; conflicting source is {spec.repo_id}."
            )
    return tuple(by_destination[destination] for destination in sorted(by_destination))


def _repo_download_kwargs(
    spec: HuggingFaceRepoSpec, local_dir: str, cache_dir: str, token: Optional[str]
) -> Dict[str, Any]:
    # Both HF_HUB_CACHE (via set_hf_environment) and cache_dir pin the cache into the
    # install dir. The env var is the real guard against the ~/.cache fallback; cache_dir
    # is explicit redundancy - important on HPC where /home is size-restricted.
    kwargs: Dict[str, Any] = {
        "repo_id": spec.repo_id,
        "repo_type": spec.repo_type,
        "revision": spec.revision,
        "local_dir": local_dir,
        "cache_dir": _hf_hub_cache_dir(cache_dir),
        "token": token,
        "max_workers": _hf_max_workers(),
    }
    if spec.include is not None:
        kwargs["allow_patterns"] = list(spec.include)
    if spec.exclude is not None:
        kwargs["ignore_patterns"] = list(spec.exclude)
    return kwargs


def download_huggingface_repos(
    repo_specs: Tuple[HuggingFaceRepoSpec, ...],
    install_path: str,
    cache_dir: str,
    token: Optional[str] = None,
) -> List[str]:
    """Download HuggingFace repo contents into LLMB-managed directories."""
    if not repo_specs:
        logger.debug("No HuggingFace repo downloads required")
        return []

    repo_specs = _dedup_and_validate_repo_destinations(repo_specs, install_path)
    os.makedirs(cache_dir, exist_ok=True)
    set_hf_environment(cache_dir)

    normalized_token = _normalize_hf_token(token) if token else None
    if token and not normalized_token:
        logger.warning("HF token provided but empty after normalization - treating as no token")
    if normalized_token:
        os.environ["HF_TOKEN"] = normalized_token
        logger.debug(f"Authenticating with HuggingFace (token: {_obfuscate_token(normalized_token)})")
    else:
        logger.debug("No HF token provided - downloads may be rate limited and gated repos will fail")

    from huggingface_hub import snapshot_download

    specs = tuple(sorted(repo_specs, key=lambda item: (item.target, item.workload_key, item.name, item.repo_id)))

    print("\nDownloading HuggingFace repos")
    print("--------------------------------")
    print("\nRequired HuggingFace repos:")
    for spec in specs:
        local_dir = resolve_huggingface_repo_local_dir(spec, install_path)
        revision = f" @ {spec.revision}" if spec.revision else ""
        repo_type = f" [{spec.repo_type}]" if spec.repo_type != "model" else ""
        print(f"  - {spec.repo_id}{revision}{repo_type} -> {local_dir}")

    print("\nDownloading repo contents...")
    print("(Progress bars are expected)")

    successful = []
    total = len(specs)
    for idx, spec in enumerate(specs, 1):
        local_dir = resolve_huggingface_repo_local_dir(spec, install_path)
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        logger.debug("Downloading HuggingFace repo %d/%d: %s -> %s", idx, total, spec.repo_id, local_dir)
        print(f"\n[{idx}/{total}] {spec.repo_id} -> {local_dir}")
        try:
            _with_retry(
                lambda hf_spec=spec, dest=local_dir, tok=normalized_token: snapshot_download(
                    **_repo_download_kwargs(hf_spec, dest, cache_dir, tok)
                ),
                operation=f"repo download '{spec.repo_id}'",
            )
        except Exception as exc:
            chain = _format_exception_chain(exc)
            logger.error("HuggingFace repo download failed for '%s': %s", spec.repo_id, chain)
            raise RuntimeError(f"HuggingFace repo download failed for '{spec.repo_id}'. Error chain: {chain}") from exc
        successful.append(local_dir)

    print(f"\nSuccessfully downloaded {len(successful)} repo(s).")
    return successful


def _verify_hf_asset(repo_id: str, asset: str, load_fn: Any, revision: Optional[str] = None) -> None:
    """Verify a HuggingFace asset loads locally.

    Uses trust_remote_code=True to match runtime behavior. This is safe because:
    1. Only curated repos from metadata.yaml are downloaded
    2. Runtime (Megatron-Bridge) already uses trust_remote_code=True
    3. Models like DeepSeek V3 and Qwen3 require custom Python modules
    """
    try:
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
        }
        if revision is not None:
            kwargs["revision"] = revision
        load_fn(repo_id, **kwargs)
    except Exception as exc:
        logger.debug(
            "Offline verification failed for repo '%s' (asset: %s): %s",
            repo_id,
            asset,
            exc,
            exc_info=True,
        )
        reason = f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__
        raise RuntimeError(
            f"HuggingFace offline verification failed for repo '{repo_id}' (asset: {asset}). "
            f"Reason: {reason}. Ensure required files (including *.py for custom models) "
            f"are present in the local HF cache."
        ) from exc


def _maybe_inject_nemotron_config(repo_id: str, assets: Set[str], snapshot_path: str) -> None:
    """Inject minimal config.json for tokenizer-only Nemotron repos when needed."""
    if "tokenizer" not in assets or "config" in assets:
        return

    if "nemotron" not in repo_id.lower():
        return

    config_path = Path(snapshot_path) / "config.json"
    if config_path.exists():
        return

    logger.debug("Injecting minimal config.json for tokenizer-only Nemotron repo '%s'", repo_id)
    with open(config_path, "w") as config_file:
        json.dump({"model_type": "nemotron"}, config_file, indent=2)

    print("  → Injected minimal config.json for tokenizer-only Nemotron")


def _maybe_finalize_tokenizer_snapshot(
    repo_id: str, assets: Set[str], snapshot_path: str, token: Optional[str]
) -> None:
    """Write tokenizer metadata files for repos that don't ship them."""
    if "tokenizer" not in assets:
        return

    snapshot_root = Path(snapshot_path)
    tokenizer_config_path = snapshot_root / "tokenizer_config.json"
    if tokenizer_config_path.exists():
        return

    from transformers import AutoTokenizer

    logger.debug(
        "Generating tokenizer metadata for repo '%s' (missing tokenizer_config.json)",
        repo_id,
    )

    def _load_tokenizer(trust_remote_code: bool) -> Any:
        return AutoTokenizer.from_pretrained(
            repo_id,
            token=token,
            trust_remote_code=trust_remote_code,
        )

    try:
        tokenizer = _load_tokenizer(trust_remote_code=False)
    except Exception as exc:
        # Only log full traceback for non-rate-limit errors (rate limits will retry)
        is_rate_limit = _is_rate_limit_error(exc)
        logger.debug(
            "Tokenizer load failed for repo '%s' with trust_remote_code=False: %s",
            repo_id,
            exc,
            exc_info=not is_rate_limit,
        )
        # NOTE: At time of writing, this edge case has only been observed for Nemotron repos.
        if "nemotron" not in repo_id.lower():
            raise
        logger.debug(
            "Retrying tokenizer load for Nemotron repo '%s' with trust_remote_code=True",
            repo_id,
        )
        tokenizer = _load_tokenizer(trust_remote_code=True)

    tokenizer.save_pretrained(snapshot_path)
    print("  → Generated tokenizer metadata files")


def verify_huggingface_cache_assets(asset_specs: Tuple[HuggingFaceAssetSpec, ...], cache_dir: str) -> None:
    """Verify required HuggingFace asset specs load offline using local cache only."""
    if not asset_specs:
        logger.debug("No HuggingFace repos required for verification phase")
        return

    os.makedirs(cache_dir, exist_ok=True)
    set_hf_environment(cache_dir)

    # Import HF libs only after HF_HOME/HF_HUB_CACHE are set.
    from transformers import AutoConfig, AutoTokenizer

    specs = tuple(sorted(asset_specs, key=lambda item: (item.repo_id, item.revision or "")))

    print("\nVerifying HuggingFace assets (offline)")
    print("-------------------------------------")
    for idx, spec in enumerate(specs, 1):
        repo_id = spec.repo_id
        assets = spec.assets
        asset_list = ", ".join(sorted(assets))
        suffix = f" ({asset_list})" if asset_list else ""
        revision = f" @ {spec.revision}" if spec.revision else ""
        print(f"\n[{idx}/{len(specs)}] {repo_id}{revision}{suffix}")

        if "tokenizer" in assets:
            _verify_hf_asset(repo_id, "tokenizer", AutoTokenizer.from_pretrained, revision=spec.revision)
            print("  ✓ tokenizer")

        if "config" in assets:
            _verify_hf_asset(repo_id, "config", AutoConfig.from_pretrained, revision=spec.revision)
            print("  ✓ config")

    print(f"\nSuccessfully verified {len(specs)} repo(s).")


def download_huggingface_files_for_workloads(
    workloads: Dict[str, Dict[str, Any]],
    selected_keys: List[str],
    install_path: str,
    hf_token: Optional[str] = None,
    gpu_type: Optional[str] = None,
) -> None:
    """Download HuggingFace cache assets and repo contents required by selected workloads.

    Cache asset downloads are verified offline. Repo downloads are written into
    LLMB-managed workload or shared dataset directories; a clean HuggingFace
    download exit is the verification for those entries.
    """
    plan = build_hf_download_plan(workloads, selected_keys, gpu_type=gpu_type)
    if plan.empty():
        logger.debug("No HuggingFace downloads required")
        return

    hf_cache_dir = os.path.join(install_path, ".cache", "huggingface")
    os.makedirs(hf_cache_dir, exist_ok=True)

    download_huggingface_cache_assets(plan.cache, hf_cache_dir, hf_token)
    verify_huggingface_cache_assets(plan.cache, hf_cache_dir)
    download_huggingface_repos(plan.repos, install_path, hf_cache_dir, hf_token)


def _obfuscate_token(token: str) -> str:
    """Obfuscate token for safe logging (show first 6 and last 4 characters)."""
    if len(token) <= 12:
        return token[:2] + "*" * (len(token) - 2)
    return f"{token[:6]}...{token[-4:]}"
