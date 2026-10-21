#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse Harbor result.json files and generate timing charts."""

import glob
import json
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _phase_duration(d, key):
    """Safely extract phase duration in seconds, returning 0.0 if phase is None."""
    phase = d.get(key)
    if not phase or not phase.get("started_at") or not phase.get("finished_at"):
        return 0.0
    return (parse_ts(phase["finished_at"]) - parse_ts(phase["started_at"])).total_seconds()


def _parse_trial_file(tf):
    """Parse a single trial result.json into a structured dict. Returns None if invalid."""
    d = json.load(open(tf))
    if not d.get("started_at") or not d.get("finished_at"):
        return None
    started = parse_ts(d["started_at"])
    finished = parse_ts(d["finished_at"])
    total = (finished - started).total_seconds()
    env_s = _phase_duration(d, "environment_setup")
    agent_setup_s = _phase_duration(d, "agent_setup")
    agent_exec_s = _phase_duration(d, "agent_execution")
    verifier_s = _phase_duration(d, "verifier")
    agent_result = d.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    llm_times_ms = metadata.get("api_request_times_msec", [])
    llm_total_s = sum(llm_times_ms) / 1000.0
    n_llm_calls = len(llm_times_ms)
    tool_time_s = max(0, agent_exec_s - llm_total_s)
    return {
        "trial_name": d.get("trial_name", os.path.basename(os.path.dirname(tf))),
        "started_at": started,
        "finished_at": finished,
        "total_s": total,
        "env_setup_s": env_s,
        "agent_setup_s": agent_setup_s,
        "agent_exec_s": agent_exec_s,
        "verifier_s": verifier_s,
        "llm_total_s": llm_total_s,
        "tool_time_s": tool_time_s,
        "n_llm_calls": n_llm_calls,
        "llm_times_ms": llm_times_ms,
        "tokens_in": agent_result.get("n_input_tokens", 0),
        "tokens_out": agent_result.get("n_output_tokens", 0),
    }


def load_trials(sweep_dir):
    """Load per-trial timing data for each concurrency level.

    Supports both the new layout (runs/<timestamp>/ subdirs) and the legacy
    layout (harbor_job__N_agents symlinks).
    """
    concurrency_data = {}

    # New layout: runs/ subdir contains one harbor job dir per concurrency level.
    runs_dir = os.path.join(sweep_dir, "runs")
    if os.path.isdir(runs_dir):
        harbor_dirs = sorted(d for d in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(d))
        # Read concurrency levels from scaling_results.json (ordered by sweep)
        concurrency_levels = []
        scaling_json = os.path.join(sweep_dir, "scaling_results.json")
        if os.path.exists(scaling_json):
            with open(scaling_json) as f:
                sd = json.load(f)
            for r in sd.get("results", []):
                entry = r[0] if isinstance(r, list) else r
                concurrency_levels.append(entry.get("n_concurrent", 0))
        for i, hdir in enumerate(harbor_dirs):
            n = concurrency_levels[i] if i < len(concurrency_levels) else (i + 1)
            trial_files = sorted(glob.glob(os.path.join(hdir, "*__*/result.json")))
            trials = [t for tf in trial_files for t in [_parse_trial_file(tf)] if t]
            if trials:
                concurrency_data[n] = trials
        if concurrency_data:
            return concurrency_data

    # Legacy layout: harbor_job__N_agents symlinks at sweep root.
    for link in sorted(glob.glob(os.path.join(sweep_dir, "harbor_job__*_agents"))):
        name = os.path.basename(link)
        n = int(name.split("__")[1].split("_")[0])
        target = os.path.realpath(link)
        trial_files = sorted(glob.glob(os.path.join(target, "*__*/result.json")))
        trials = [t for tf in trial_files for t in [_parse_trial_file(tf)] if t]
        concurrency_data[n] = trials

    return concurrency_data


def plot_wall_time(concurrency_data, out_dir):
    """Bar chart: total benchmark wall time vs per-agent average per concurrency level."""
    levels = sorted(concurrency_data.keys())
    batch_wall = []
    agent_avg = []
    agent_max = []
    agent_min = []

    for n in levels:
        trials = concurrency_data[n]
        earliest = min(t["started_at"] for t in trials)
        latest = max(t["finished_at"] for t in trials)
        batch_wall.append((latest - earliest).total_seconds())
        times = [t["total_s"] for t in trials]
        agent_avg.append(np.mean(times))
        agent_max.append(max(times))
        agent_min.append(min(times))

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2,
        batch_wall,
        width,
        label="Batch wall time (first start → last finish)",
        color="#2196F3",
        edgecolor="white",
    )
    bars2 = ax.bar(x + width / 2, agent_avg, width, label="Avg per-agent wall time", color="#FF9800", edgecolor="white")

    yerr_low = [agent_avg[i] - agent_min[i] for i in range(len(levels))]
    yerr_high = [agent_max[i] - agent_avg[i] for i in range(len(levels))]
    ax.errorbar(
        x + width / 2, agent_avg, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=5, capthick=1.5
    )

    ax.set_xlabel("Concurrent Agents", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Total Benchmark Time vs. Agent Time by Concurrency", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in levels])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.0f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.0f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "timing_wall_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_phase_breakdown(concurrency_data, out_dir):
    """Stacked bar chart: average time per phase for each concurrency level."""
    levels = sorted(concurrency_data.keys())
    phases = ["env_setup_s", "agent_setup_s", "agent_exec_s", "verifier_s"]
    labels = ["Env Setup", "Agent Setup\n(pip install, etc.)", "Agent Execution", "Verifier"]
    colors = ["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"]

    avgs = {p: [] for p in phases}
    for n in levels:
        trials = concurrency_data[n]
        for p in phases:
            avgs[p].append(np.mean([t[p] for t in trials]))

    x = np.arange(len(levels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(levels))
    for phase, label, color in zip(phases, labels, colors, strict=False):
        vals = avgs[phase]
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, edgecolor="white")
        for i, v in enumerate(vals):
            if v > 2:
                ax.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{v:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
        bottom += np.array(vals)

    ax.set_xlabel("Concurrent Agents", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Average Per-Agent Time Breakdown by Phase", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in levels])
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "timing_phase_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_llm_vs_tool(concurrency_data, out_dir):
    """During agent execution: LLM wait time vs tool/CPU time."""
    levels = sorted(concurrency_data.keys())

    avg_llm = []
    avg_tool = []
    for n in levels:
        trials = concurrency_data[n]
        avg_llm.append(np.mean([t["llm_total_s"] for t in trials]))
        avg_tool.append(np.mean([t["tool_time_s"] for t in trials]))

    x = np.arange(len(levels))
    width = 0.5
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x, avg_tool, width, label="Tool / CPU time (local)", color="#4CAF50", edgecolor="white")
    ax.bar(x, avg_llm, width, bottom=avg_tool, label="LLM inference (remote GPU)", color="#F44336", edgecolor="white")

    for i in range(len(levels)):
        total = avg_tool[i] + avg_llm[i]
        pct_llm = (avg_llm[i] / total * 100) if total > 0 else 0
        ax.text(x[i], total + 0.2, f"{pct_llm:.0f}% LLM", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if avg_tool[i] > 0.3:
            ax.text(
                x[i],
                avg_tool[i] / 2,
                f"{avg_tool[i]:.1f}s",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        if avg_llm[i] > 0.3:
            ax.text(
                x[i],
                avg_tool[i] + avg_llm[i] / 2,
                f"{avg_llm[i]:.1f}s",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

    ax.set_xlabel("Concurrent Agents", fontsize=13)
    ax.set_ylabel("Time (seconds)", fontsize=13)
    ax.set_title("Agent Execution: LLM Wait vs. Tool/CPU Time", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in levels])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "timing_llm_vs_tool.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_agent_timeline(concurrency_data, out_dir):
    """Gantt-style chart showing each agent's lifecycle phases."""
    levels = sorted(concurrency_data.keys())
    fig, axes = plt.subplots(len(levels), 1, figsize=(14, 3 * len(levels)), squeeze=False)

    phase_colors = {
        "env_setup": "#4CAF50",
        "agent_setup": "#FF9800",
        "agent_exec": "#2196F3",
        "verifier": "#9C27B0",
    }

    for idx, n in enumerate(levels):
        ax = axes[idx, 0]
        trials = concurrency_data[n]
        earliest = min(t["started_at"] for t in trials)

        for i, t in enumerate(sorted(trials, key=lambda x: x["started_at"])):
            offset = (t["started_at"] - earliest).total_seconds()
            phases = [
                ("env_setup", t["env_setup_s"]),
                ("agent_setup", t["agent_setup_s"]),
                ("agent_exec", t["agent_exec_s"]),
                ("verifier", t["verifier_s"]),
            ]
            left = offset
            for pname, dur in phases:
                ax.barh(i, dur, left=left, height=0.6, color=phase_colors[pname], edgecolor="white")
                left += dur

        ax.set_yticks(range(len(trials)))
        ax.set_yticklabels([f"Agent {i+1}" for i in range(len(trials))], fontsize=9)
        ax.set_xlabel("Time (seconds from batch start)", fontsize=10)
        ax.set_title(f"{n} Concurrent Agent{'s' if n > 1 else ''}", fontsize=12, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Env Setup"),
        mpatches.Patch(color="#FF9800", label="Agent Setup"),
        mpatches.Patch(color="#2196F3", label="Agent Execution"),
        mpatches.Patch(color="#9C27B0", label="Verifier"),
    ]
    axes[0, 0].legend(handles=legend_patches, loc="upper right", fontsize=9)

    fig.suptitle("Agent Lifecycle Timeline by Concurrency Level", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "timing_agent_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_call_latency(concurrency_data, out_dir):
    """Box plot of individual LLM call latencies by concurrency level."""
    levels = sorted(concurrency_data.keys())
    all_latencies = []
    labels = []

    for n in levels:
        trials = concurrency_data[n]
        calls = []
        for t in trials:
            calls.extend(t["llm_times_ms"])
        all_latencies.append(calls)
        labels.append(str(n))

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(all_latencies, tick_labels=labels, patch_artist=True, widths=0.5)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for patch, color in zip(bp["boxes"], colors[: len(levels)], strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(all_latencies):
        jitter = np.random.normal(0, 0.04, len(data))
        ax.scatter(
            np.full(len(data), i + 1) + jitter,
            data,
            alpha=0.6,
            s=30,
            zorder=3,
            color=colors[i % len(colors)],
            edgecolors="white",
            linewidth=0.5,
        )

    ax.set_xlabel("Concurrent Agents", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("Per-LLM-Call Latency Distribution", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "timing_llm_latency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary_table(concurrency_data, out_dir):
    """Summary table as an image."""
    levels = sorted(concurrency_data.keys())
    rows = []
    for n in levels:
        trials = concurrency_data[n]
        earliest = min(t["started_at"] for t in trials)
        latest = max(t["finished_at"] for t in trials)
        batch_s = (latest - earliest).total_seconds()
        avg_total = np.mean([t["total_s"] for t in trials])
        avg_exec = np.mean([t["agent_exec_s"] for t in trials])
        avg_llm = np.mean([t["llm_total_s"] for t in trials])
        avg_tool = np.mean([t["tool_time_s"] for t in trials])
        pct_llm = (avg_llm / avg_exec * 100) if avg_exec > 0 else 0
        avg_calls = np.mean([t["n_llm_calls"] for t in trials])
        avg_tok_in = np.mean([t["tokens_in"] for t in trials])
        avg_tok_out = np.mean([t["tokens_out"] for t in trials])

        rows.append(
            [
                str(n),
                f"{batch_s:.1f}",
                f"{avg_total:.1f}",
                f"{avg_exec:.1f}",
                f"{avg_llm:.1f}",
                f"{avg_tool:.1f}",
                f"{pct_llm:.0f}%",
                f"{avg_calls:.1f}",
                f"{avg_tok_in:.0f}",
                f"{avg_tok_out:.0f}",
            ]
        )

    col_labels = [
        "Agents",
        "Batch\nWall (s)",
        "Avg Agent\nWall (s)",
        "Avg Agent\nExec (s)",
        "Avg LLM\nTime (s)",
        "Avg Tool\nTime (s)",
        "% LLM",
        "Avg LLM\nCalls",
        "Avg Tok\nIn",
        "Avg Tok\nOut",
    ]

    fig, ax = plt.subplots(figsize=(14, 2 + 0.5 * len(levels)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ECEFF1" if row % 2 == 0 else "white")

    ax.set_title("Timing Summary by Concurrency Level", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    path = os.path.join(out_dir, "timing_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def generate_timing_charts(sweep_dir):
    """Generate timing charts for a sweep directory. Can be called as a library."""
    out_dir = os.path.join(sweep_dir, "timing_charts")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading trial data from {sweep_dir}...")
    data = load_trials(sweep_dir)
    if not data:
        print("No harbor_job__*_agents symlinks found. Skipping timing charts.")
        return False

    print(f"Found concurrency levels: {sorted(data.keys())}")
    for n, trials in sorted(data.items()):
        print(f"  {n} agents: {len(trials)} trials")

    print("\nGenerating timing charts...")
    plot_wall_time(data, out_dir)
    plot_phase_breakdown(data, out_dir)
    plot_llm_vs_tool(data, out_dir)
    plot_agent_timeline(data, out_dir)
    plot_per_call_latency(data, out_dir)
    plot_summary_table(data, out_dir)
    return True


def main():
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    if not generate_timing_charts(sweep_dir):
        sys.exit(1)


if __name__ == "__main__":
    main()
