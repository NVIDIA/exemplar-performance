# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Matplotlib chart generation for benchmark results.

Reads the scaling_results.json data dict and produces PNG figures:
  fig1_cores.png       -- CPU cores + processes vs concurrent agents
  fig1b_proctypes.png  -- process type breakdown (horizontal bar)
  fig3_bursts.png      -- CPU utilization sparklines per concurrency level
  summary_table.png    -- summary metrics table
  fig4_time_split.png  -- per-agent CPU vs LLM time split
  fig6_disk.png        -- disk read/write throughput
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(data, output_dir):
    """Generate all benchmark charts from a results data dict."""
    results = data["results"]
    cpu_info = data["cpu_info"]
    if not results:
        return
    n_cores = results[0].get("n_cores", 96)

    agents = [r["n_concurrent"] for r in results]
    avg_cores = [r["avg_cores_active"] for r in results]
    avg_procs = [r["avg_procs"] for r in results]

    x = np.arange(len(agents))

    _plot_cores(x, agents, avg_cores, avg_procs, n_cores, output_dir, results=results)
    _plot_proc_breakdown(results, output_dir)
    _plot_cpu_bursts(results, data.get("dataset", "benchmark"), output_dir)
    _plot_cores_timeline(results, data.get("dataset", "benchmark"), output_dir)
    _plot_summary_table(results, cpu_info, n_cores, output_dir)
    _plot_time_split(x, agents, data.get("time_splits", {}), output_dir)
    _plot_disk_io(x, agents, results, output_dir)

    print(f"\n  All figures saved to {output_dir}/")


# ─── Individual chart functions ──────────────────────────────────────────────


def _plot_cores(x, agents, avg_cores, avg_procs, n_cores, output_dir, results=None):
    p95_cores = []
    peak_cores = []
    if results:
        for r in results:
            core_samples = [s.get("cores_active", 0) for s in r.get("samples", [])]
            if core_samples:
                p95_cores.append(float(np.percentile(core_samples, 95)))
                peak_cores.append(float(max(core_samples)))
            else:
                p95_cores.append(0.0)
                peak_cores.append(0.0)
    else:
        p95_cores = [0.0] * len(agents)
        peak_cores = [0.0] * len(agents)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    w = 0.22
    ax1.bar(x - w, avg_cores, w, color="#4a90d9", edgecolor="#2a5a8a", linewidth=1, label="Avg")
    ax1.bar(x, p95_cores, w, color="#e07020", edgecolor="#b05010", linewidth=1, label="P95")
    ax1.bar(x + w, peak_cores, w, color="#d9534f", edgecolor="#a03030", linewidth=1, label="Peak")

    for i, (a, p95, pk) in enumerate(zip(avg_cores, p95_cores, peak_cores, strict=False)):
        ax1.text(i - w, a + 0.5, f"{a:.1f}", ha="center", fontsize=8, fontweight="bold", color="#2a5a8a")
        ax1.text(i, p95 + 0.5, f"{p95:.0f}", ha="center", fontsize=8, fontweight="bold", color="#b05010")
        ax1.text(i + w, pk + 0.5, f"{pk:.0f}", ha="center", fontsize=8, fontweight="bold", color="#a03030")

    ax1.set_xlabel("Concurrent Agents", fontsize=12)
    ax1.set_ylabel("CPU Cores", fontsize=12)
    ax1.set_title("CPU Cores (Avg / P95 / Peak) vs Concurrent Agents", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    all_vals = avg_cores + p95_cores + peak_cores
    ax1.set_ylim(0, max(max(all_vals) * 1.25, 10))
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=10, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "fig1_cores.png", dpi=150)
    print(f"  Saved {output_dir / 'fig1_cores.png'}")
    plt.close()


def _plot_proc_breakdown(results, output_dir):
    cat_colors = {
        "python": "#a78bfa",
        "bash/sh": "#3fb950",
        "pip/install": "#f0883e",
        "grep": "#4da6ff",
        "git": "#d2a8ff",
        "apt/dpkg": "#ffa657",
        "compile": "#f85149",
        "curl/wget": "#79c0ff",
        "node/npm": "#56d364",
        "echo/tee": "#8b949e",
        "sed/awk": "#79c0ff",
        "other": "#999",
    }
    last_run = results[-1]
    cats = dict(last_run.get("avg_proc_categories", {}))
    sorted_cats = sorted(cats.items(), key=lambda kv: kv[1], reverse=True)
    sorted_cats = [(k, v) for k, v in sorted_cats if v > 0.3]
    if not sorted_cats:
        return

    labels = [k for k, _ in sorted_cats]
    values = [v for _, v in sorted_cats]
    colors = [cat_colors.get(k, "#999") for k in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels) + 1)))
    ax.barh(range(len(labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Avg Active Processes", fontsize=12)
    ax.set_title(
        f"Process Breakdown — {last_run['n_concurrent']} Concurrent Agent(s)",
        fontsize=14,
        fontweight="bold",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1b_proctypes.png", dpi=150)
    print(f"  Saved {output_dir / 'fig1b_proctypes.png'}")
    plt.close()


def _plot_cpu_bursts(results, dataset_name, output_dir):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.2 * n)), sharex=False)
    if n == 1:
        axes = [axes]

    for i, r in enumerate(results):
        ax = axes[i]
        samples = [s for s in r.get("samples", []) if s.get("n_containers", 0) > 0]
        if not samples:
            samples = r.get("samples", [])
        t_vals = [s["t"] for s in samples]
        cpu_vals = [s["cpu_total"] for s in samples]

        ax.fill_between(t_vals, cpu_vals, alpha=0.4, color="#4a90d9")
        ax.plot(t_vals, cpu_vals, color="#2a5a8a", linewidth=0.8)
        ax.set_ylim(0, max(50, max(cpu_vals) * 1.2) if cpu_vals else 50)
        ax.set_ylabel("CPU %", fontsize=9)
        ax.set_title(
            f'{r["n_concurrent"]} concurrent agent{"s" if r["n_concurrent"] > 1 else ""}',
            fontsize=11,
            fontweight="bold",
            loc="left",
        )
        ax.grid(axis="both", alpha=0.2)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(f"CPU Utilization — {dataset_name}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_bursts.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'fig3_bursts.png'}")
    plt.close()


def _plot_cores_timeline(results, dataset_name, output_dir):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3, 2.2 * n)), sharex=False)
    if n == 1:
        axes = [axes]

    n_cores = results[0].get("n_cores", 128)

    for i, r in enumerate(results):
        ax = axes[i]
        samples = [s for s in r.get("samples", []) if s.get("n_containers", 0) > 0]
        if not samples:
            samples = r.get("samples", [])
        t_vals = [s["t"] for s in samples]
        core_vals = [s.get("cores_active", 0) for s in samples]

        ax.fill_between(t_vals, core_vals, alpha=0.4, color="#e07020")
        ax.plot(t_vals, core_vals, color="#b05010", linewidth=0.8)
        peak = max(core_vals) if core_vals else 1
        ax.set_ylim(0, max(peak * 1.3, 8))
        ax.axhline(y=n_cores, color="#ccc", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Cores", fontsize=9)
        label = f'{r["n_concurrent"]} concurrent agent{"s" if r["n_concurrent"] > 1 else ""}'
        avg_c = np.mean(core_vals) if core_vals else 0
        ax.set_title(f"{label}  (avg {avg_c:.1f}, peak {peak})", fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="both", alpha=0.2)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    fig.suptitle(f"Active CPU Cores Over Time — {dataset_name}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_cores_timeline.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'fig2_cores_timeline.png'}")
    plt.close()


def _plot_summary_table(results, cpu_info, n_cores, output_dir):
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.4 * len(results)))
    ax.axis("off")
    table_data = []
    for r in results:
        table_data.append(
            [
                str(r["n_concurrent"]),
                f'{r["avg_cores_active"]:.1f}',
                f'{r["peak_cores_active"]}/{n_cores}',
                f'{r["avg_cpu"]:.1f}%',
                f'{r["peak_cpu"]:.1f}%',
                f'{r["avg_procs"]:.0f}',
            ]
        )
    table = ax.table(
        cellText=table_data,
        colLabels=["Agents", "Avg Cores", "Peak Cores", "Avg CPU", "Peak CPU", "Avg Procs"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4a90d9")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if row % 2 == 0 else "#fff")
    ax.set_title(f'{cpu_info["model"]} — {n_cores} cores', fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(output_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'summary_table.png'}")
    plt.close()


def _plot_time_split(x, agents, time_splits, output_dir):
    if not time_splits:
        return

    def _val(v, default=0):
        if v is None:
            return default
        if isinstance(v, dict):
            return v.get("median", default)
        return v

    pct_cpu_real = []
    pct_llm_real = []
    for a in agents:
        key = str(a)
        if key in time_splits:
            cpu = _val(time_splits[key].get("pct_cpu"), 0)
            llm = _val(time_splits[key].get("pct_llm"), 0)
            pct_cpu_real.append(cpu)
            pct_llm_real.append(llm)
        else:
            pct_cpu_real.append(0)
            pct_llm_real.append(0)

    if all(v == 0 for v in pct_cpu_real) and all(v == 0 for v in pct_llm_real):
        return  # no valid timing data to plot

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x,
        pct_cpu_real,
        0.5,
        color="#e07040",
        edgecolor="#a04020",
        linewidth=1,
        label="CPU time (tool execution, parsing)",
    )
    ax.bar(
        x,
        pct_llm_real,
        0.5,
        bottom=pct_cpu_real,
        color="#a0c4e8",
        edgecolor="#4a90d9",
        linewidth=1,
        label="LLM time (GPU inference)",
    )
    for i, v in enumerate(pct_cpu_real):
        if v > 5:
            ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    for i, v in enumerate(pct_llm_real):
        if v > 5:
            ax.text(
                i,
                pct_cpu_real[i] + v / 2,
                f"{v:.0f}%",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#2a5a8a",
            )
    ax.set_xlabel("Concurrent Agents", fontsize=12)
    ax.set_ylabel("% of Per-Agent Wall-Clock Time", fontsize=12)
    ax.set_title("Per-Agent Time: CPU (Tool Execution) vs LLM (GPU Inference)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_time_split.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {output_dir / 'fig4_time_split.png'}")
    plt.close()


def _plot_disk_io(x, agents, results, output_dir):
    avg_read = [r.get("avg_disk_read_mb_s", 0) for r in results]
    avg_write = [r.get("avg_disk_write_mb_s", 0) for r in results]
    if not any(v > 0.1 for v in avg_read + avg_write):
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, avg_read, 0.35, color="#4a90d9", edgecolor="#2a5a8a", linewidth=1, label="Avg Read (MB/s)")
    ax.bar(x + 0.2, avg_write, 0.35, color="#e07040", edgecolor="#a04020", linewidth=1, label="Avg Write (MB/s)")
    for i, v in enumerate(avg_read):
        if v > 0.1:
            ax.text(i - 0.2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold", color="#2a5a8a")
    for i, v in enumerate(avg_write):
        if v > 0.1:
            ax.text(i + 0.2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold", color="#a04020")
    ax.set_xlabel("Concurrent Agents", fontsize=12)
    ax.set_ylabel("Disk I/O (MB/s)", fontsize=12)
    ax.set_title("Disk Read/Write Throughput vs Concurrent Agents", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig6_disk.png", dpi=150)
    print(f"  Saved {output_dir / 'fig6_disk.png'}")
    plt.close()
