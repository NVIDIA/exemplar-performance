# AgentScry Privileged Cross-Check

`tasktrace report compare-agentscry` checks whether CPU time from a
privileged AgentScry process trace agrees with the container cgroup CPU time
for the same offline replay. It is separate from the normal profiler:

| Measurement                                       | Privilege                                                                                 | Scope                                                                                                          |
| ------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `tasktrace profile transcript --strace-mode none` | Docker access only; no `sudo`, `ptrace`, perf, or BPF access                              | Disposable replay container `cpu.stat` across the measured command interval.                                   |
| AgentScry `--exit --cputime`                      | Root-equivalent BPF capability; the comparison procedure uses a privileged Docker sidecar | Host eBPF process exit CPU observations, narrowed to the replay cpuset and attributed to captured descendants. |

## What Is AgentScry?

AgentScry is an external, optional eBPF/BCC-based process-event sidecar. It is
not part of this repository and is not downloaded or installed by TaskTrace.
TaskTrace only consumes AgentScry's filtered CSV output through the
`compare-agentscry` command. Another tool can be used if it produces a
compatible CSV.

This comparison is an independent diagnostic cross-check. The normal
`profile transcript`, `profile atif`, and `profile container` workflows do not
require AgentScry.

## Expected CSV Format

The filtered CSV must contain these columns:

| Column | Type    | Description                                      |
| ------ | ------- | ------------------------------------------------ |
| `PID`  | integer | Process ID.                                      |
| `PPID` | integer | Parent process ID.                               |
| `ARGS` | string  | Command line used to find the replay entrypoint. |
| `USER` | float   | User-mode CPU seconds for the process.           |
| `SYS`  | float   | System-mode CPU seconds for the process.         |

The analyzer also uses these optional columns when present:

| Column     | Type   | Description                                    |
| ---------- | ------ | ---------------------------------------------- |
| `DURATION` | float  | Wall-clock duration apportioned by the filter. |
| `COMM`     | string | Short process name.                            |

When a process has multiple exec rows, such as output from
`asfilter --duration`, the comparison sums every retained row instead of
deduplicating by PID.

Use `--root-command-substring` to identify the measured replay entrypoint in
the `ARGS` column; its default is `/tasktrace-profile-entrypoint.sh`. Use
`--tolerance-fraction` to set the maximum acceptable relative difference
between AgentScry and cgroup CPU totals; its default is `0.05` (5%).

Its available CPU-time mode records `USER`/`SYS` when a process reaches its
exit hook. It can therefore omit a per-process value for descendants that
terminate through a signal rather than the exit hook, such as `grep` producers
killed when a `head` pipeline closes early. The comparison analyzer reports
both numerical CPU agreement and a stricter completeness check so that a close
total is not confused with complete process-level accounting.

## Reproducible Comparison Procedure

Use a single fixed trial, image, fixture checkout, and cpuset for both runs.
The sidecar command below uses
`docker run --privileged --pid=host --cgroupns=host`. This is root-equivalent
instrumentation even though it does not issue an interactive host `sudo`
command. Require explicit approval and handle it accordingly.

```bash
EXPERIMENT=/tmp/tasktrace-offline-agentscry-comparison
TRANSCRIPT=your_transcripts.jsonl
TASK=your_task_id
TRIAL=your_trial_name
IMAGE=your-task-image:tag
SKILL_DIR=/tmp/your-skill-dir
CPUSET=0-3

# Normal offline measurement: no BPF or ptrace privilege.
uv run tasktrace profile transcript "$TRANSCRIPT" \
  --task-id "$TASK" --trial-name "$TRIAL" \
  --image "$IMAGE" --cpuset-cpus "$CPUSET" \
  --source-workspace /path/to/agent_workspace \
  --pinchbench-skill-dir "$SKILL_DIR" \
  --strace-mode none --timeout-sec 180 \
  --output-dir "$EXPERIMENT/no-sudo-baseline"
```

Obtain AgentScry separately, review it, and place `agentscry.py` and
`asfilter.py` outside this repository. Then run the same replay under the
privileged sidecar.

The sidecar needs this host access:

- `--pid=host` exposes host process IDs so AgentScry can observe the replay
  process tree.
- `--cgroupns=host` exposes the host cgroup hierarchy used to scope the replay.
- `-v /:/host:ro` mounts the host filesystem read-only at `/host`.
- `chroot /host` uses the host's Python and BCC installation. The separate
  experiment-directory mount is writable so the tracer can emit its CSV and
  diagnostic files.

```bash
# Place reviewed copies of agentscry.py and asfilter.py in the tools directory.
mkdir -p "$EXPERIMENT/agentscry-tools"
cp /path/to/agentscry/agentscry.py "$EXPERIMENT/agentscry-tools/"
cp /path/to/agentscry/asfilter.py "$EXPERIMENT/agentscry-tools/"

TRACER=offline-agentscry-comparison
docker run -d --name "$TRACER" \
  --privileged --pid=host --cgroupns=host \
  -v /:/host:ro -v "$EXPERIMENT":"/host$EXPERIMENT":rw \
  --entrypoint /usr/sbin/chroot "$IMAGE" \
  /host /bin/sh -c "exec /usr/bin/python3 '$EXPERIMENT/agentscry-tools/agentscry.py' \
    --csv -t -C -e -c --cores '$CPUSET' -p '$EXPERIMENT/agentscry.pid' \
    > '$EXPERIMENT/agentscry_raw.csv' 2> '$EXPERIMENT/agentscry.stderr.txt'"
sleep 3

uv run tasktrace profile transcript "$TRANSCRIPT" \
  --task-id "$TASK" --trial-name "$TRIAL" \
  --image "$IMAGE" --cpuset-cpus "$CPUSET" \
  --source-workspace /path/to/agent_workspace \
  --pinchbench-skill-dir "$SKILL_DIR" \
  --strace-mode none --timeout-sec 180 \
  --output-dir "$EXPERIMENT/agentscry-leg"

docker stop --signal=INT --time 10 "$TRACER"
docker rm "$TRACER"
python3 "$EXPERIMENT/agentscry-tools/asfilter.py" --duration --warn \
  "$EXPERIMENT/agentscry_raw.csv" > "$EXPERIMENT/agentscry_filtered.csv" \
  2> "$EXPERIMENT/asfilter.stderr.txt"

uv run tasktrace report compare-agentscry \
  "$EXPERIMENT/agentscry-leg/pipeline_report.json" \
  "$EXPERIMENT/agentscry_filtered.csv" \
  --output "$EXPERIMENT/agentscry_comparison_report.json"
```

The comparison command:

- Roots attribution at `/tasktrace-profile-entrypoint.sh`.
- Follows only captured descendants of that measured replay container command.
- Sums AgentScry CPU portions on every accounted row because
  `asfilter --duration` proportionally distributes one PID's exit CPU across its
  retained `exec` rows.
- Compares AgentScry `USER + SYS` against the **same run's** cgroup
  `usage_sec`.
- Reports numerical CPU agreement separately from strict equivalence.
- Exits nonzero when CPU differs beyond tolerance or any replay descendant
  lacks AgentScry exit accounting.

## Interpreting the Comparison

A successful comparison shows that the AgentScry descendant `USER + SYS` sum
is within the configured numerical tolerance of the same run's cgroup
measurement. AgentScry's exit-based mode may still omit accounting for
signal-terminated pipeline members, so numerical agreement does not establish
complete process-level attribution. The task-scoped cgroup total remains the
authoritative container CPU measurement.
