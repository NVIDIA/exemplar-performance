#!/usr/bin/env bash
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

# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# parallel_repro.sh <cell-dir> <profile> <N> [--stagger S] [--attach|--foreground] [--dry-run] [-- <run.sh flags>]
#
# Within-cluster reproducibility, IN PARALLEL: run <cell> N times at once on N nodes — instead of N sequential
# Run the same cell N times in parallel. Detached mode records a durable sweep ID; collect later with:
#   llmb-k8s collect --sweep <sweep-id> --cluster <profile>
# Use --attach for short runs that should remain connected to the local orchestrator.
#
# THE ORCHESTRATOR ITSELF SELF-DETACHES (default). Detached LEGS were never the whole story: firing them is a
# SEQUENTIAL loop paced by the FSx stagger (60s base, auto-widened when filesystems < copies), so at N=8 the
# orchestrator lives 16-30 MINUTES locally. A session boundary inside that window killed sweeps that had cloned
# every leg and applied ZERO Jobs — nothing detached existed yet to survive (rc20260802t010113bacb lost 8/8).
# So before doing any work this script re-execs itself into its OWN SESSION (python double-fork + setsid; macOS
# has no setsid(1)) with stdin </dev/null and stdout/stderr to results/.sweeps/<sweep-id>/orchestrator.log. It
# VERIFIES the child reparented (ppid==1) and refuses to continue attached if it did not, then prints the
# sweep-id, log path and detached pid and returns. Follow it with `tail -f <log>`.
#   --foreground (--no-self-detach)  keep the orchestrator inline (CI, debugging); legs are still detached.
#   --attach                         implies --foreground and waits for all legs.
#   --dry-run                        never self-detaches — its plan output is for the caller to read NOW.
#   PARALLEL_REPRO_DETACHED_CHILD=1  sentinel the child carries; it is the re-exec loop guard (never set it).
#
# SAFETY BY DESIGN: it does NOT modify the shared lifecycle scripts (run.sh/deploy.sh/sweep.sh) that the live
# fleet uses. It clones the cell into N scratch copies (detached: durable under results/.sweeps/<id>/legs so a
# later `collect --sweep` can re-read them; --attach: ephemeral under $TMPDIR — NEVER under recipes/) with a
# SUFFIXED envelope.name, so every derived k8s object — server deployment, artifacts PVC, collision label — is
# distinct and the N runs can't collide. Each copy is the SAME benchmark_id as the original (benchmark_id
# excludes name), so its value is a legitimate reproducibility sample; the N values are consolidated into the
# ORIGINAL cell's runs.jsonl (stamped with the original's recipe_hash) and `compare --repro` prints the spread.
#
#   --stagger S   base seconds between launching each copy (default 60). The model-cache is FSx; starting many
#                 550B loads on ONE filesystem contends for its bandwidth and each load runs at 1/N speed.
#                 Stagger so loads don't thunder. The EFFECTIVE stagger is derived from the spread below: with
#                 one filesystem per load it auto-drops to a token 5s (no contention left to spread); when there
#                 are FEWER filesystems than copies it auto-WIDENS to base×ceil(N/npvcs) so copies that pile onto
#                 a shared filesystem stay serialized instead of thrashing it.
#
#   SPREAD IS AUTOMATIC. By default (MODEL_CACHE_PVCS unset) the copies are round-robined across REPLICA model-
#   cache PVCs auto-discovered on the target cluster: the profile's MODEL_CACHE_PVC base plus its Bound
#   <base>-r<N> replicas (e.g. <cell>-model-cache + -r2 -r3 -r4 -r5). Each PVC is its own filesystem with
#   its own throughput, so K replicas give K× aggregate read bandwidth and K loads run at full speed at once —
#   you can no longer forget to opt in and silently thunder one filesystem. Add more <base>-r<N> replicas to
#   fully de-contend; with fewer filesystems than copies the stagger auto-widens (above). The replicas must be
#   populated with the model snapshot (same layout as the base PVC).
#
#   env MODEL_CACHE_PVCS="pvc-a pvc-b pvc-c"   OVERRIDE auto-discovery with an explicit round-robin list.
#   --dry-run     clone + render the N collision-free copies and print the plan; deploy NOTHING (offline-safe).
#
# NOTE: experimental — smoke it with N=2 on one cell before a big fan-out. GPU count is not the only limit;
# FSx model-load bandwidth caps how many servers can load concurrently, so keep N modest and stagger.
# Set KEEP_REPRO_SCRATCH=1 to preserve the scratch copies and their run.log/publish.log after a successful run.
set -euo pipefail

# ---------------------------------------------------------------------------
# Pure helpers for model-cache spread (auto-discovery + round-robin + stagger
# widening). Kept side-effect-free so selftest_parallel_repro_spread.py can
# source this file with PARALLEL_REPRO_LIB_ONLY=1 and exercise them against a
# MOCKED PVC list — no live cluster required.
# ---------------------------------------------------------------------------

# mcp_select_replicas <base> < "NAME PHASE" lines on stdin
# Echo the ordered spread set for a base model-cache PVC: the base itself first
# (only if Bound), then every Bound replica named <base>-r<N> in ascending N.
# Non-Bound and non-matching PVCs are dropped. This is what "auto-spread by
# default" round-robins across when MODEL_CACHE_PVCS is unset.
mcp_select_replicas() {
    local base="$1"
    awk -v base="$base" '
    $2=="Bound" {
      if ($1==base) { has_base=1; next }
      if ($1 ~ "^"base"-r[0-9]+$") { n=$1; sub("^"base"-r","",n); repl[n]=$1 }
    }
    END {
      if (has_base) print base
      k=0; for (i in repl) keys[k++]=i
      for (a=1;a<k;a++){ v=keys[a]; b=a-1; while(b>=0 && keys[b]+0>v+0){keys[b+1]=keys[b];b--}; keys[b+1]=v }
      for (a=0;a<k;a++) print repl[keys[a]]
    }'
}

# mcp_eff_stagger <base_stagger> <N_copies> <N_pvcs>
# Effective inter-launch stagger given how many replica filesystems the N cold
# loads are spread over:
#   npvcs<=0  → unchanged (no spread info; single profile PVC — all contend)
#   npvcs>=N  → 5s token (one dedicated filesystem per load; no FSx contention)
#   0<npvcs<N → base_stagger × ceil(N/npvcs): the busiest filesystem serves
#               d=ceil(N/npvcs) loads, so widen the launch gap by d to serialize
#               the copies that pile onto one shared PVC (fewer replicas → more
#               pile-up → wider stagger). This is the auto-widen that makes it
#               structurally impossible for N copies to thrash one filesystem.
mcp_eff_stagger() {
    local stagger="$1" n="$2" npvcs="$3"
    if [ "$npvcs" -le 0 ]; then
        echo "$stagger"
        return
    fi
    if [ "$npvcs" -ge "$n" ]; then
        echo 5
        return
    fi
    local d=$(((n + npvcs - 1) / npvcs)) # ceil(N/npvcs) = loads on busiest filesystem
    echo $((stagger * d))
}

# ---------------------------------------------------------------------------
# Self-detach primitives (also pure/side-effect-free enough for the selftest to
# source and drive directly — see selftest_parallel_repro_spread.py section D/E).
# ---------------------------------------------------------------------------

# should_self_detach <child_sentinel> <dry> <attach> <foreground> → prints 1 or 0.
# The orchestrator re-execs itself into its own session UNLESS:
#   child_sentinel  — we ARE the re-exec'd child (the loop guard; without this the
#                     child would detach a grandchild, forever).
#   dry             — --dry-run prints a plan the caller must read synchronously.
#   attach          — --attach waits for the legs inline.
#   foreground      — explicit --foreground/--no-self-detach escape hatch (CI/debug).
should_self_detach() {
    local child="${1:-0}" dry="${2:-0}" attach="${3:-0}" fg="${4:-0}"
    if [ "$child" = 1 ] || [ "$dry" = 1 ] || [ "$attach" = 1 ] || [ "$fg" = 1 ]; then echo 0; else echo 1; fi
}

# spawn_detached <logfile> <cmd> [args...] → prints the detached PID; exit 1 if it
# could not be reparented to init.
#
# WHY PYTHON, NOT setsid(1)/nohup: macOS has NO setsid binary (verified: `command -v
# setsid` fails on darwin), and `nohup … & disown` only ignores SIGHUP — it leaves the
# child in the CALLER'S process group, so the harness's group-wide reap (killpg) still
# takes it. python3 is already a hard dependency here and os.setsid() is POSIX, so the
# same code path works on macOS and Linux:
#   fork  → intermediate calls os.setsid()  (new SESSION + new process group: a killpg
#           or SIGHUP aimed at the caller's group can no longer reach us)
#   fork  → grandchild execs the real command with stdin=/dev/null and stdout/stderr
#           appended to <logfile> (no controlling terminal to lose)
#   intermediate _exit(0) → the grandchild is orphaned and the kernel reparents it to
#           init (pid 1 / launchd), which is what we then VERIFY rather than assume.
# The parent waitpid()s the intermediate, so reparenting has already happened when we
# read the pid; the poll is belt-and-braces for a Linux child-subreaper, where ppid
# would settle on the subreaper instead of 1 — in that case we FAIL LOUDLY.
spawn_detached() {
    python3 - "$@" << 'PY'
import os, subprocess, sys, time

# Read another process's parent from procfs on Linux, with a portable fallback for other systems.
_HAVE_PROC = os.path.isdir("/proc/self")

def _ppid_of(pid):
    """PPID of `pid`, or None when it is GONE. None is 'no such process', never 'could not tell' —
    the caller relies on that distinction to decide whether detachment can be claimed."""
    if _HAVE_PROC:
        try:
            data = open("/proc/%d/stat" % pid, "rb").read()
        except OSError:
            return None
        try:
            # field 2 (comm) is parenthesised and may itself contain spaces AND ')', so a naive
            # split() mis-indexes every subsequent field: split after the LAST ')'.
            return int(data[data.rindex(b")") + 2:].split()[1])
        except (ValueError, IndexError):
            return None
    out = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)],
                         capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else None

log, cmd = sys.argv[1], sys.argv[2:]
if not cmd:
    sys.exit("spawn_detached: no command")
r, w = os.pipe()
pid = os.fork()
if pid == 0:                      # intermediate
    os.close(r)
    os.setsid()                   # new session + process group: immune to the caller's killpg/SIGHUP
    pid2 = os.fork()
    if pid2 == 0:                 # grandchild → becomes the detached orchestrator
        os.close(w)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        fo = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fo, 1); os.dup2(fo, 2)
        try:
            os.execvp(cmd[0], cmd)
        except Exception as e:     # noqa: BLE001 — last chance to say why, into the log
            sys.stderr.write(f"spawn_detached: exec failed: {e}\n")
        os._exit(127)
    os.write(w, str(pid2).encode())
    os.close(w)
    os._exit(0)                   # → grandchild orphaned → reparented to init
os.close(w)
os.waitpid(pid, 0)                # reap the intermediate; reparenting is done once this returns
child = int(os.read(r, 32) or 0)
os.close(r)
if child <= 0:
    sys.exit("spawn_detached: no child pid")
state = "unknown"
for _ in range(50):               # ≤5s: confirm the reparent instead of claiming it
    _pp = _ppid_of(child)
    if _pp is None:               # already exited (short-lived child) — it can no longer be group-reaped
        state = "gone"
        break
    out = str(_pp)
    if out.isdigit() and int(out) == 1:
        state = "1"
        break
    state = out
    time.sleep(0.1)
if state not in ("1", "gone"):
    sys.stderr.write(
        f"spawn_detached: child {child} did NOT reparent to init (ppid={state}); "
        "refusing to claim detachment\n")
    try:
        os.kill(child, 15)
    except OSError:
        pass
    sys.exit(1)
print(f"{child} {state}")         # "<pid> 1" (reparented) or "<pid> gone" (already finished)
PY
}

# mcp_assign <N> <pvc...>  → one PVC per copy (round-robin), one per line.
mcp_assign() {
    local n="$1"
    shift
    local npvcs=$#
    local -a pvcs=("$@")
    local i
    for ((i = 0; i < n; i++)); do echo "${pvcs[$((i % npvcs))]}"; done
}

# Sourced by the selftest to reach the pure helpers without running the tool.
if [ -n "${PARALLEL_REPRO_LIB_ONLY:-}" ]; then return 0 2> /dev/null || exit 0; fi

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ORIG_ARGV=("$@") # verbatim argv, replayed into the self-detached child (see the re-exec below)

CELL="${1:?usage: parallel_repro.sh <cell-dir> <profile> <N> [--stagger S] [--attach] [--dry-run] [-- extra run.sh flags]}"
PROFILE="${2:?need profile}"
N="${3:?need N (number of parallel repeats)}"
shift 3
# DEFAULT = DETACHED (ATTACH=0): fire N `run.sh --detach` legs (each does only ~2-min setup then exits), record
# a durable sweep-id, and exit. Harvest later via `llmb-k8s collect --sweep <id>`; the runs continue
# multi-hour run (the reason a repro campaign kept losing its auto fetch/publish to the ~hourly session reap).
# --attach waits for each leg and consolidates results inline (background each leg, wait, publish,
# consolidate, print compare --repro) — fine for SHORT runs where staying attached is acceptable.
STAGGER=60
DRY=0
ATTACH=0
FOREGROUND=0
EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --stagger)
            STAGGER="${2:?--stagger needs seconds}"
            shift 2
            ;;
        # --attach waits for the legs inline and therefore implies --foreground.
        --attach | --wait)
            ATTACH=1
            FOREGROUND=1
            shift
            ;;
        --detach)
            ATTACH=0
            shift
            ;;
        # keep the ORCHESTRATOR inline (legs stay detached) — CI, debugging, and anything reading our stdout live.
        --foreground | --no-self-detach)
            FOREGROUND=1
            shift
            ;;
        --dry-run)
            DRY=1
            shift
            ;;
        --)
            shift
            EXTRA=("$@")
            break
            ;;
        *)
            EXTRA+=("$1")
            shift
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$CELL/recipe.yaml" ] || {
    echo "parallel_repro: no recipe.yaml at $CELL" >&2
    exit 1
}

# ============================================================================================================
# SELF-DETACH (default). Everything below this point — cloning N cells, rendering them, and firing the legs
# one at a time paced by the FSx stagger — is a 16-30 MINUTE loop at N=8. Run it in our OWN SESSION so a
# session/context boundary that reaps this shell's process group cannot take the orchestrator with it and
# strand a sweep with legs cloned but ZERO Jobs applied. The sweep-id is minted HERE so the parent can print
# the durable handle (and the child's log lives inside that sweep dir) before returning.
# ============================================================================================================
if [ "$(should_self_detach "${PARALLEL_REPRO_DETACHED_CHILD:-0}" "$DRY" "$ATTACH" "$FOREGROUND")" = 1 ]; then
    SWEEP_ID="rc$(date -u +%Y%m%dt%H%M%S)$(printf '%04x' "$((($$ ^ $(date +%s)) % 65536))")"
    _sweep_dir="$ROOT/results/.sweeps/$SWEEP_ID"
    mkdir -p "$_sweep_dir"
    _log="$_sweep_dir/orchestrator.log"
    # Stake the INTENT on disk before the child exists, so even a child that dies in its first second (bad
    # profile, render blowup) leaves a sweep dir that explains itself instead of a bare log.
    python3 - "$_sweep_dir/plan.json" "$SWEEP_ID" "$CELL" "$PROFILE" "$N" "$_log" << 'PY' || true
import json, sys, time
p, sweep_id, cell, profile, repeat, log = sys.argv[1:7]
with open(p, "w") as f:
    f.write(json.dumps({
        "sweep_id": sweep_id, "cell": cell, "profile": profile, "repeat": int(repeat),
        "mode": "parallel-detached", "orchestrator_log": log, "state": "spawning",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
PY
    # The child carries the sentinel (loop guard) + the pre-minted sweep-id, and replays our verbatim argv.
    if ! _spawned="$(PARALLEL_REPRO_DETACHED_CHILD=1 PARALLEL_REPRO_SWEEP_ID="$SWEEP_ID" \
        spawn_detached "$_log" bash "$SELF" ${ORIG_ARGV[@]+"${ORIG_ARGV[@]}"})"; then
        echo "parallel_repro: FATAL — could not detach the orchestrator into its own session." >&2
        echo "  Refusing to continue attached: a $N-leg fan-out takes 15-30 min and a session boundary would" >&2
        echo "  leave it partially submitted. Re-run with --foreground to wait for submission. (log: $_log)" >&2
        rm -rf "$_sweep_dir" 2> /dev/null || true
        exit 1
    fi
    _pid="${_spawned%% *}"
    _ppid="${_spawned##* }"
    # Say what we actually observed. "gone" = the child already exited (it can no longer be group-reaped, but
    # it also did NOT get far) — never dress that up as a healthy detachment.
    if [ "$_ppid" = 1 ]; then
        _detach_note="ppid=1 — reparented to init, verified"
    else
        _detach_note="child already exited before we could sample its ppid — CHECK THE LOG"
    fi
    cat << EOF
parallel_repro: 🔌 SELF-DETACHED — the orchestrator now runs in its own session (ppid=${_ppid}); this shell may
                die (session boundary, ^C, closed laptop) without stranding the fan-out.

  sweep-id: $SWEEP_ID
  pid:      $_pid   (${_detach_note})
  log:      $_log
  follow:   tail -f $_log
  progress: $_sweep_dir/plan.json  +  $_sweep_dir/progress.jsonl  (which legs fired, which did not)
  collect:  llmb-k8s collect --sweep $SWEEP_ID --cluster $PROFILE   (after the legs finish)

                Use --foreground to keep it inline instead; --dry-run and --attach never self-detach.
EOF
    exit 0
fi

BASE="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["envelope"]["name"])' "$CELL/recipe.yaml")"

# Scratch location + sweep identity depend on the mode:
#   ATTACH  → ephemeral $TMPDIR scratch (cleaned on success), no sweep-id (consolidated inline).
#   DETACH  → a DURABLE sweep dir results/.sweeps/<sweep-id>/legs that MUST survive this process, because
#             `collect --sweep <id>` (a separate, later invocation) re-reads each leg's scratch cell to fetch,
#             publish, and consolidate. Mirrors scripts/submit.sh --repeat's durable sweep root.
if [ "$ATTACH" = 1 ]; then
    SWEEP_ID=""
    SCRATCH="${TMPDIR:-/tmp}/llmb-repro-${BASE}-$$"
else
    # Self-detached child: reuse the sweep-id the parent already minted + printed to the operator (its
    # orchestrator.log already lives in that dir). Only a --foreground run mints its own here.
    SWEEP_ID="${PARALLEL_REPRO_SWEEP_ID:-rc$(date -u +%Y%m%dt%H%M%S)$(printf '%04x' "$((($$ ^ $(date +%s)) % 65536))")}"
    SCRATCH="$ROOT/results/.sweeps/$SWEEP_ID/legs"
fi
mkdir -p "$SCRATCH"
RUNS_FAILED=0
PUBLISH_FAILED=0

# ── durable progress record (detached mode) ─────────────────────────────────────────────────────────────────
# A self-detached orchestrator has no terminal, so its state must be ON DISK next to the legs. results/.sweeps/
# <id>/ carries: plan.json (the intent — every planned leg tag, the salt that generated them, our pid/log, and a
# state machine cloning→firing→fired) and progress.jsonl (one append-only line per leg the moment it is applied
# or fails). The final results/.sweeps/<id>.json is written ONLY after every leg has been attempted, so:
#   plan.json state=fired + <id>.json present  → the sweep really fired all N legs
#   plan.json state=cloning|firing, no <id>.json → it was KILLED mid-flight; progress.jsonl names exactly which
#                                                  legs made it to the cluster and which never did.
# A half-fired sweep can therefore never masquerade as complete (the failure mode that hid rc20260802t010113bacb).
SWEEP_DIR=""
SWEEP_STATE=""
[ -n "$SWEEP_ID" ] && SWEEP_DIR="$ROOT/results/.sweeps/$SWEEP_ID"
sweep_state() { # <state> [note]
    SWEEP_STATE="$1"
    [ -n "$SWEEP_DIR" ] || return 0
    python3 - "$SWEEP_DIR/plan.json" "$1" "${2:-}" << 'PY' || true
import json, os, sys, time
p, state, note = sys.argv[1], sys.argv[2], sys.argv[3]
d = {}
if os.path.exists(p):
    try:
        d = json.load(open(p))
    except Exception:
        d = {}
d["state"] = state
d["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if note:
    d["note"] = note
tmp = p + ".tmp"
with open(tmp, "w") as f:
    f.write(json.dumps(d, indent=2) + "\n")
os.replace(tmp, p)
PY
}
sweep_progress() { # <run-id> <applied|failed> [detail]
    [ -n "$SWEEP_DIR" ] || return 0
    printf '{"run_id":"%s","state":"%s","detail":"%s","ts":"%s"}\n' \
        "$1" "$2" "${3:-}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SWEEP_DIR/progress.jsonl"
}
on_signal() { # a reap must leave an honest record, never a silent half-fired sweep
    sweep_state interrupted "orchestrator killed by SIG$1 mid-flight; legs already applied are in progress.jsonl and keep running in-cluster"
    echo "parallel_repro: SIG$1 — orchestrator interrupted; state recorded in ${SWEEP_DIR:-<none>}/plan.json" >&2
    exit 143
}
if [ "$ATTACH" = 0 ]; then
    trap 'on_signal TERM' TERM
    trap 'on_signal INT' INT
    trap 'on_signal HUP' HUP
fi

cleanup() {
    if [ "$DRY" = "1" ]; then
        rm -rf "$SCRATCH"
        [ -n "$SWEEP_ID" ] && rm -rf "$ROOT/results/.sweeps/$SWEEP_ID" 2> /dev/null || true
        return
    fi
    # DETACHED: the sweep dir is a durable handle for a LATER `collect --sweep` — never auto-delete it here.
    if [ "$ATTACH" = 0 ]; then
        # Exiting before the fan-out finished (set -e abort, kill -9's follow-up, render failure): say so on disk
        # instead of leaving a plan that reads as if every leg is in flight.
        case "$SWEEP_STATE" in
            "" | fired | interrupted) : ;;
            *) sweep_state aborted "orchestrator exited during '$SWEEP_STATE'; see orchestrator.log + progress.jsonl" ;;
        esac
        return
    fi
    if [ "${KEEP_REPRO_SCRATCH:-0}" = "1" ] || [ "$RUNS_FAILED" -ne 0 ] || [ "$PUBLISH_FAILED" -ne 0 ]; then
        echo "parallel_repro: scratch preserved at $SCRATCH"
        return
    fi
    rm -rf "$SCRATCH"
}
trap cleanup EXIT
RUN_TAG_SALT="$(printf '%04x' "$(($$ % 65536))")"

# Size the clone tag to the TIGHTEST derived k8s name (≤63), per LANE. The coord Job carries the tag TWICE
# (<cell>-<tag>-<kind>-<tag>) and the artifacts PVC adds "-artifacts"; a long base name
# leaves little room, so don't assume the llm-perf "-bench-<rid>" shape — derive the budget from lane.py's kind.
# This lets any lane clone despite a long envelope.name.
KIND="$(python3 "$ROOT/scripts/lane.py" "$CELL" kind 2> /dev/null || echo bench)"
_job_budget=$(((63 - ${#BASE} - ${#KIND} - 3) / 2)) # <cell>-<tag>-<kind>-<tag>  (tag appears twice)
_art_budget=$((63 - ${#BASE} - 11))                 # <cell>-<tag>-artifacts
MAXTAG=$((_job_budget < _art_budget ? _job_budget : _art_budget))
if [ "$MAXTAG" -lt 3 ]; then
    echo "parallel_repro: base cell name '${BASE}' (${#BASE} chars) is too long to clone for lane '$KIND' —" >&2
    echo "  even a minimal clone tag overflows the 63-char k8s name limit (max tag=$MAXTAG). Shorten envelope.name." >&2
    exit 1
fi

# 1) clone N copies with distinct suffixed names, re-rendered so their manifests carry the distinct names.
# Tags are derived (index + per-process salt), so PLAN them first and record the salt: plan.json then names
# every leg this sweep intends to fire, before a single one exists, and the tags are reproducible from it.
TAGS=()
for i in $(seq 1 "$N"); do
    _t="r$(printf '%x' "$i")${RUN_TAG_SALT}" # ≤MAXTAG chars, unique by index
    TAGS+=("${_t:0:MAXTAG}")
done
if [ -n "$SWEEP_DIR" ]; then
    python3 - "$SWEEP_DIR/plan.json" "$SWEEP_ID" "$CELL" "$PROFILE" "$BASE" "$N" "$RUN_TAG_SALT" "$$" \
        "$SWEEP_DIR/orchestrator.log" "${TAGS[@]}" << 'PY'
import json, os, sys, time
(p, sweep_id, cell, profile, recipe, repeat, salt, pid, log), tags = sys.argv[1:10], sys.argv[10:]
rec = {
    "sweep_id": sweep_id, "cell": cell, "profile": profile, "recipe": recipe,
    "repeat": int(repeat), "mode": "parallel-detached", "tag_salt": salt,
    "orchestrator_pid": int(pid), "orchestrator_log": log,
    "self_detached": os.environ.get("PARALLEL_REPRO_DETACHED_CHILD", "0") == "1",
    "planned_run_ids": tags, "state": "cloning",
    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(p, "w") as f:
    f.write(json.dumps(rec, indent=2) + "\n")
PY
    SWEEP_STATE="cloning"
fi
copies=()
for tag in ${TAGS[@]+"${TAGS[@]}"}; do
    cell_name="${BASE}-${tag}"
    for _suf in "-server" "-artifacts" "-${KIND}"; do # non-truncated derived objects must fit
        if [ "$((${#cell_name} + ${#_suf}))" -gt 63 ]; then
            echo "parallel_repro: ${cell_name}${_suf} exceeds the 63-char k8s limit (tag-sizing bug)" >&2
            exit 1
        fi
    done
    dst="$SCRATCH/$tag"
    cp -R "$CELL" "$dst"
    rm -rf "$dst/rendered" "$dst/runs" "$dst/runs.jsonl" "$dst/RESULTS.md" "$dst/record.json" "$dst/results" 2> /dev/null || true
    # Set the suffixed envelope.name (collision-free clone) and, when REPRO_N_ATTEMPTS is set, OVERRIDE
    # n_attempts in the clone BEFORE render. This is the reproducibility-floor knob: it fixes the
    # attempts/task FLOOR so every clone renders the same attempt count. Per-rung the template scales it as
    # max(n_attempts, ceil(attempts_target_waves*rung / n_tasks)) — at c16 with waves=2,n_tasks=16 that floor
    # is 2, so for REPRO_N_ATTEMPTS>=2 the rendered attempts == REPRO_N_ATTEMPTS exactly. Unset => unchanged
    # (fully backward-compatible). Only applied when the cell actually has a bench: block.
    python3 - "$dst/recipe.yaml" "$tag" << 'PY'
import os, yaml, sys
p, tag = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p))
d["envelope"]["name"] = f'{d["envelope"]["name"]}-{tag}'
na = os.environ.get("REPRO_N_ATTEMPTS", "").strip()
if na:
    try:
        n = int(na)
    except ValueError:
        sys.exit(f"parallel_repro: REPRO_N_ATTEMPTS must be an integer, got {na!r}")
    if n < 1:
        sys.exit(f"parallel_repro: REPRO_N_ATTEMPTS must be >=1, got {n}")
    if isinstance(d.get("bench"), dict):
        d["bench"]["n_attempts"] = n
    else:
        sys.exit("parallel_repro: REPRO_N_ATTEMPTS set but cell has no bench: block")
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
    bash "$ROOT/scripts/render.sh" "$dst" > /dev/null
    copies+=("$dst")
done

echo "parallel_repro: ${N}× ${BASE} — collision-free scratch copies:"
for c in "${copies[@]}"; do
    nm="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["envelope"]["name"])' "$c/recipe.yaml")"
    rid="$(basename "$c")"
    echo "   server=${nm}-server   job=${nm}-${KIND}-${rid}   pvc=${nm}-artifacts   ($c)"
done

if [ "$DRY" = "1" ]; then
    if [ "$ATTACH" = 1 ]; then
        echo "parallel_repro: --dry-run (--attach) — rendered $N collision-free copies; would wait for each run.sh"
        echo "                inline (background → wait → publish → consolidate → compare --repro); deployed nothing."
    else
        echo "parallel_repro: --dry-run (DETACHED default) — rendered $N collision-free copies; would fire each as"
        echo "                run.sh --detach (setup-then-exit) then EXIT; deployed nothing."
        echo "                sweep-id would be: $SWEEP_ID"
        echo "                harvest later:     llmb-k8s collect --sweep $SWEEP_ID --cluster $PROFILE"
    fi
    exit 0
fi

# ============================================================================================================
# DETACHED (default): fire each leg as `run.sh --detach` (setup-then-exit), record a durable sweep, then EXIT.
# No inline wait/publish/consolidate — that happens later via `llmb-k8s collect --sweep <id>` (which fetches +
# publishes each leg, consolidates their values into the ORIGINAL cell's runs.jsonl, and prints compare --repro).
# ============================================================================================================
pids=()
rids=()
# Attached-only run.sh flags depend on the local process; detached runs rely
# on the in-cluster run-owner for teardown and have no laptop to host a guard.
run_flags=(--idle-guard --teardown)
if [ "${#EXTRA[@]}" -gt 0 ]; then
    run_flags+=("${EXTRA[@]}")
fi
# Some lanes need a distinct serving node per parallel copy (each run pins coord+server to ONE
# node via SERVING_NODE). Pass SERVING_NODES="node1 node2 …" to assign one per clone in order; falls back to a
# single ambient $SERVING_NODE. Warn if too few nodes given.
read -r -a _serving_nodes <<< "${SERVING_NODES:-}"
if [ "${#_serving_nodes[@]}" -gt 0 ] && [ "${#_serving_nodes[@]}" -lt "$N" ]; then
    echo "parallel_repro: WARN — SERVING_NODES has ${#_serving_nodes[@]} node(s) for $N copies; extras reuse the last/ambient (may collide on one node)." >&2
fi
# Spread the 550B cold loads across REPLICA model-cache PVCs. Each PVC on the
# FSx class is its OWN filesystem with its OWN provisioned throughput, so N loads
# on N filesystems each get full bandwidth instead of 1/N of one shared one.
#
# AUTO-SPREAD BY DEFAULT: when MODEL_CACHE_PVCS is UNSET we auto-discover the
# spread set on the target cluster — the profile's MODEL_CACHE_PVC base plus its
# Bound <base>-r<N> replicas — and round-robin the copies across them. This makes
# de-contention the default: you can no longer forget to opt in and silently
# thunder one filesystem. An explicit MODEL_CACHE_PVCS overrides discovery.
_pvc_src="explicit"
read -r -a _pvcs <<< "${MODEL_CACHE_PVCS:-}"
if [ "${#_pvcs[@]}" -eq 0 ]; then
    _pvc_src="auto"
    # Read kube coords from the profile without polluting our env.
    eval "$(
        _pf="$ROOT/cluster-profiles/${PROFILE}.env"
        if [ -f "$_pf" ]; then
            set -a
            . "$_pf" > /dev/null 2>&1 || true
            set +a
        fi
        printf '_KCTX=%q\n_KNS=%q\n' "${KUBE_CONTEXT:-}" "${NAMESPACE:-}"
    )"
    # THE BASE CLAIM COMES FROM THE RESOLVER, NOT FROM RAW ${MODEL_CACHE_PVC}.
    # This is the same single-rule invariant every other consumer obeys, and it was the last place that did
    # not: reading the profile's global MODEL_CACHE_PVC here means that on a cluster using a per-model key
    # (MODEL_CACHE_PVC_<MODEL_SLUG> — which cluster-profiles/example-gpu-cluster.env.example now ships) the
    # replicas discovered are replicas of the WRONG claim. Each leg would then be handed
    # MODEL_CACHE_PVC_OVERRIDE=<global-claim>-rN and mount a replica of a filesystem that does not hold this
    # model's weights, while the model's actual claim sits unused — model-not-found AFTER a full GPU
    # allocation, on N nodes at once. The override outranks the profile inside the leg, so nothing downstream
    # can catch it.
    # FAIL-OPEN IS NOT AVAILABLE HERE EITHER: if the claim cannot be resolved we spread nothing and say so,
    # rather than falling back to the global claim (each leg then resolves for itself, exactly as a
    # non-parallel run does).
    # stdout ONLY (the claim); the resolver's diagnostics — including the exact profile key to set — go to
    # the terminal. Capturing 2>&1 here would glue a warning banner onto the claim name, which is the
    # fail-OPEN this codebase already paid for once in _model_cache.sh.
    _BASE_PVC="$(python3 "$ROOT/scripts/model_cache.py" resolve "$CELL" "$ROOT/cluster-profiles/${PROFILE}.env" || true)"
    if [ -z "$_BASE_PVC" ]; then
        echo "parallel_repro: WARN — could not resolve this cell's model-cache claim from profile '${PROFILE}'; NOT spreading across replicas. Each leg resolves its own claim (scripts/_model_cache.sh); set MODEL_CACHE_PVCS to spread explicitly." >&2
    fi
    if [ -n "$_BASE_PVC" ] && command -v kubectl > /dev/null 2>&1; then
        _pvc_lines="$(kubectl ${_KCTX:+--context "$_KCTX"} ${_KNS:+-n "$_KNS"} \
            get pvc -o custom-columns=NAME:.metadata.name,PHASE:.status.phase --no-headers 2> /dev/null || true)"
        _pvcs=() # bash 3.2 (macOS) has no mapfile; read the discovered set line-by-line
        while IFS= read -r _l; do [ -n "$_l" ] && _pvcs+=("$_l"); done \
            < <(printf '%s\n' "$_pvc_lines" | mcp_select_replicas "$_BASE_PVC")
    fi
    if [ "${#_pvcs[@]}" -gt 0 ]; then
        echo "parallel_repro: auto-discovered ${#_pvcs[@]} model-cache filesystem(s) for base '${_BASE_PVC}': ${_pvcs[*]}" >&2
    else
        echo "parallel_repro: WARN — could not auto-discover replicas for base '${_BASE_PVC:-<unset>}' (kubectl/profile?); all $N copies fall back to the single profile PVC and WILL contend on its FSx bandwidth. Set MODEL_CACHE_PVCS or add <base>-r2… replicas." >&2
    fi
fi
_npvcs="${#_pvcs[@]}"
# Auto-widen the stagger so copies that pile onto a shared filesystem (copies >
# replicas) are serialized enough not to thrash it; token 5s when replicas>=N.
_eff_stagger="$(mcp_eff_stagger "$STAGGER" "$N" "$_npvcs")"
if [ "$_npvcs" -gt 0 ] && [ "$_npvcs" -lt "$N" ]; then
    _d=$(((N + _npvcs - 1) / _npvcs))
    echo "parallel_repro: ${_npvcs} ${_pvc_src} filesystem(s) < $N copies — busiest serves ${_d} loads; auto-widened stagger ${STAGGER}s→${_eff_stagger}s to serialize same-PVC loads." >&2
elif [ "$_npvcs" -ge "$N" ] && [ "$_npvcs" -gt 0 ]; then
    echo "parallel_repro: ${_npvcs} ${_pvc_src} filesystem(s) >= $N copies — one filesystem per load, no FSx contention; stagger→${_eff_stagger}s" >&2
fi
if [ "$ATTACH" = 0 ]; then
    # ── DETACHED (default) ── fire each leg as run.sh --detach (setup-then-exit), SEQUENTIALLY with the stagger
    # (each returns after applying its Job; the in-cluster run then proceeds unattended, so the
    # stagger paces when each server's weight-load STARTS — the same FSx-contention control as the attach path).
    sweep_state firing
    _idx=0
    for c in "${copies[@]}"; do
        rid="$(basename "$c")"
        rids+=("$rid")
        _node="${_serving_nodes[$_idx]:-${SERVING_NODE:-}}"
        if [ "$_npvcs" -gt 0 ]; then _pvc="${_pvcs[$((_idx % _npvcs))]}"; else _pvc=""; fi
        echo "parallel_repro: [detached] applying $(basename "$c") run-id=$rid node=${_node:-<unset>} pvc=${_pvc:-<profile-default>} …"
        if SERVING_NODE="$_node" MODEL_CACHE_PVC_OVERRIDE="$_pvc" \
            bash "$ROOT/scripts/run.sh" "$c" "$PROFILE" "$rid" --detach ${EXTRA[@]+"${EXTRA[@]}"} > "$c/run.log" 2>&1; then
            sweep_progress "$rid" applied "job applied to cluster; run-owner owns it from here"
            echo "parallel_repro: [detached] leg $rid applied (see $c/run.log)"
        else
            RUNS_FAILED=$((RUNS_FAILED + 1))
            sweep_progress "$rid" failed "run.sh --detach setup failed; see $c/run.log"
            echo "parallel_repro: [detached] leg $rid setup FAILED — see $c/run.log (its run-id stays in the sweep record)" >&2
        fi
        _idx=$((_idx + 1))
        [ "$_idx" -lt "$N" ] && {
            echo "parallel_repro: staggering ${_eff_stagger}s before the next leg…"
            sleep "$_eff_stagger"
        }
    done

    # Durable sweep record (results/.sweeps/<id>.json) + cluster-side sweep ConfigMap (llmb-sweep-<id>; no
    # ownerRef/TTL → survives Job GC) so `collect --sweep <id> [--cluster <profile>]` resolves + consolidates the
    # legs — same schema scripts/submit.sh --repeat writes and scripts/resilient_status.py reads.
    ORIG_BID="$(python3 "$ROOT/scripts/benchmark_id.py" "$CELL" --short 2> /dev/null || echo '?')"
    mkdir -p "$ROOT/results/.sweeps"
    python3 - "$ROOT/results/.sweeps/${SWEEP_ID}.json" "$SWEEP_ID" "$CELL" "$PROFILE" "$BASE" "$N" "$ORIG_BID" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${rids[@]}" << 'PY'
import json, sys
out, sweep_id, cell, profile, recipe, repeat, bid, created = sys.argv[1:9]
tags = sys.argv[9:]
# scratch cell dirs live at results/.sweeps/<id>/legs/<tag> (this parallel_repro run wrote them there).
import os
root = os.path.dirname(os.path.dirname(out))  # results/
legs_root = os.path.join(root, ".sweeps", sweep_id, "legs")
rec = {
    "sweep_id": sweep_id, "cell": cell, "profile": profile, "recipe": recipe,
    "repeat": int(repeat), "mode": "parallel-detached", "benchmark_id": bid,
    "created_utc": created, "run_ids": tags,
    "legs": [{"run_id": t, "scratch_name": f"{recipe}-{t}",
              "scratch_cell": os.path.join(legs_root, t)} for t in tags],
}
open(out, "w").write(json.dumps(rec, indent=2) + "\n")
PY
    # Every leg has now been ATTEMPTED and the durable <id>.json handle exists → the sweep is honestly "fired".
    sweep_state fired "$((N - RUNS_FAILED))/$N legs applied"
    # Apply the cluster-side sweep ConfigMap (best-effort; the local record above is the primary handle).
    (
        _pf="$ROOT/cluster-profiles/${PROFILE}.env"
        if [ -f "$_pf" ]; then
            set -a
            . "$_pf" > /dev/null 2>&1 || true
            set +a
        fi
        _ns="${NAMESPACE:-}"
        _ctx="${KUBE_CONTEXT:-}"
        if [ -n "$_ns" ]; then
            kubectl ${_ctx:+--context "$_ctx"} apply -f - << EOF > /dev/null 2>&1 || true
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmb-sweep-${SWEEP_ID}
  namespace: ${_ns}
  labels:
    app.kubernetes.io/managed-by: llmb-recipe
    llmb.nvidia.com/sweep-id: ${SWEEP_ID}
    llmb.nvidia.com/lifecycle: detached
data:
  sweep_id: "${SWEEP_ID}"
  cell: "${CELL}"
  profile: "${PROFILE}"
  namespace: "${_ns}"
  recipe: "${BASE}"
  repeat: "${N}"
  mode: "parallel-detached"
  benchmark_id: "${ORIG_BID}"
  run_ids: "${rids[*]}"
EOF
        fi
    )

    cat << EOF

parallel_repro: ✅ DETACHED — $((N - RUNS_FAILED))/$N legs applied (each independently in-cluster; the run-owner
                frees each GPU on its Job's terminal state). You may disconnect (laptop + tsh) now.

  sweep-id: $SWEEP_ID
  legs:     ${rids[*]}
  collect:  llmb-k8s collect --sweep $SWEEP_ID --cluster $PROFILE
            (fetch each leg → publish → consolidate into $CELL/runs.jsonl → compare --repro spread)
  watch:    llmb-k8s fleet --watch
  record:   $ROOT/results/.sweeps/$SWEEP_ID/plan.json (state=fired) + progress.jsonl (per-leg apply log)
EOF
    [ "$RUNS_FAILED" -eq 0 ] || {
        echo "parallel_repro: completed with $RUNS_FAILED setup failure(s); inspect the leg run.log(s)." >&2
        exit 1
    }
    exit 0
fi

# ============================================================================================================
# ATTACH (--attach): wait for each run.sh leg and consolidate results inline — background each leg, wait,
# publish, consolidate, compare --repro. Fine for SHORT runs where staying attached is acceptable.
# ============================================================================================================
_idx=0
for c in "${copies[@]}"; do
    rid="$(basename "$c")" # unique short tag per copy, pre-checked against k8s name limits.
    rids+=("$rid")
    _node="${_serving_nodes[$_idx]:-${SERVING_NODE:-}}"
    if [ "$_npvcs" -gt 0 ]; then _pvc="${_pvcs[$((_idx % _npvcs))]}"; else _pvc=""; fi
    (SERVING_NODE="$_node" MODEL_CACHE_PVC_OVERRIDE="$_pvc" bash "$ROOT/scripts/run.sh" "$c" "$PROFILE" "$rid" "${run_flags[@]}" > "$c/run.log" 2>&1) &
    pids+=($!)
    echo "parallel_repro: launched $(basename "$c") run-id=$rid node=${_node:-<unset>} pvc=${_pvc:-<profile-default>} (pid $!); staggering ${_eff_stagger}s"
    _idx=$((_idx + 1))
    [ "$_idx" -lt "$N" ] && sleep "$_eff_stagger"
done
echo "parallel_repro: $N runs launched — watch: llmb-k8s fleet --watch"

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
RUNS_FAILED="$fail"
echo "parallel_repro: $((N - fail))/$N run.sh invocations exited 0 (see each copy's run.log)"

# 3) publish each copy (writes its value + benchmark_id into the copy's runs.jsonl), then consolidate the
#    values into the ORIGINAL cell's runs.jsonl and print the within-cluster spread.
for i in "${!copies[@]}"; do
    c="${copies[$i]}"
    rid="${rids[$i]}"
    if "$ROOT/scripts/llmb-k8s" publish "$c" "$PROFILE" "$ROOT/results/$rid" > "$c/publish.log" 2>&1; then
        echo "parallel_repro: published $(basename "$c")"
    else
        PUBLISH_FAILED=1
        echo "parallel_repro: publish FAILED for $(basename "$c") — see $c/publish.log (run may not have produced a value)"
    fi
done
python3 "$ROOT/scripts/repro_consolidate.py" "$CELL" "${copies[@]}"
python3 "$ROOT/analysis/compare.py" --repro "$CELL" || true
echo "parallel_repro: done. Review $CELL/runs.jsonl + the compare --repro spread above."
if [ "$RUNS_FAILED" -ne 0 ] || [ "$PUBLISH_FAILED" -ne 0 ]; then
    echo "parallel_repro: completed with failures; inspect the preserved scratch logs before retrying." >&2
    exit 1
fi
