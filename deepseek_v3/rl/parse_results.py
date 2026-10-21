#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import json
import os
import statistics

import matplotlib.pyplot as plt

METRICS = {
    'timing/train/total_step_time': 'Total step time (s)',
    'performance/tokens_per_sec_per_gpu': 'Tokens/sec/GPU',
    'train/mean_total_tokens_per_sample': 'G-Avg seq len',
}

CONFIG_KEYS = [
    'Algo',
    'Policy',
    'T-Max Seq len',
    '#-GPUs',
    'G-GBS',
    'T-GBS',
    'G-TP/PP',
    'T-TP/CP/EP/PP/VPP',
]

MAX_STEPS = 10
START_STEP = 2
END_STEP = 6  # inclusive

TIMING_THRESHOLD = 5.0  # (seconds)
TIMING_EXCLUDE_KEYS = {"timing/train/valid_tokens_per_sec_per_gpu"}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--prefix", type=str, default="timing/")
    parser.add_argument("--exclude_border_steps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def find_config(raw):
    megatron = raw['policy']['megatron_cfg']
    vllm = raw['policy']['generation']['vllm_cfg']
    grpo = raw['grpo']

    tp = megatron['tensor_model_parallel_size']
    ep = megatron['expert_model_parallel_size']
    pp = megatron['pipeline_model_parallel_size']
    cp = megatron['context_parallel_size']
    vpp = megatron.get('virtual_pipeline_model_parallel_size', 1)

    return {
        'Algo': 'GRPO',
        'Policy': "Off policy" if grpo['async_grpo']['enabled'] else "On policy",
        'T-Max Seq len': raw['policy']['max_total_sequence_length'],
        '#-GPUs': raw['cluster']['num_nodes'] * raw['cluster']['gpus_per_node'],
        'G-GBS': grpo['num_generations_per_prompt'] * grpo['num_prompts_per_step'],
        'T-GBS': raw['policy']['train_global_batch_size'],
        'G-TP/PP': f"{vllm['tensor_parallel_size']}/{vllm['pipeline_parallel_size']}",
        'T-TP/CP/EP/PP/VPP': f'{tp}/{cp}/{ep}/{pp}/{vpp}',
    }


# Extract the config from the log file
def read_config_from_log(file_path):
    config_lines = []
    in_config_block = False
    with open(file_path, 'r') as f:
        for line in f:
            if in_config_block:
                config_lines.append(line)
            if "Final config:" in line:
                in_config_block = True
            if "train_micro_batch_size" in line:
                break
    if not config_lines:
        return None
    return find_config(ast.literal_eval("".join(config_lines)))


def get_metrics_from_data(data):
    result = {}
    for key in METRICS:
        raw = data.get(key)
        if raw is None or len(raw) != MAX_STEPS:
            return None
        window = list(raw.values())[START_STEP : END_STEP + 1]
        result[f"{key}_avg"] = statistics.mean(window)
        result[f"{key}_std"] = statistics.stdev(window)
    return result


def _print_table(all_cols, rows):
    col_widths = [len(h) for h in all_cols]
    for row in rows:
        col_widths[:] = [max(w, len(str(cell))) for w, cell in zip(col_widths, row, strict=True)]

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(col_widths[i]) for i, c in enumerate(cells)) + " |"

    print(fmt_row(all_cols))
    print("| " + " | ".join("-" * w for w in col_widths) + " |")
    for row in rows:
        print(fmt_row(row))


def print_markdown_results(results):
    print("## Performance Results Summary")
    all_cols = ["Experiment"] + CONFIG_KEYS + list(METRICS.values())

    rows = []
    for experiment, values in sorted(results.items()):
        cfg = values.get('config') or {}
        metrics = values.get('metrics') or {}
        config_cells = [cfg.get(k, 'N/A') for k in CONFIG_KEYS]
        metric_cells = [
            f"{metrics[f'{k}_avg']:.4f} (+/-{metrics[f'{k}_std']:.4f})" if f"{k}_avg" in metrics else 'N/A'
            for k in METRICS
        ]
        rows.append([experiment] + config_cells + metric_cells)

    _print_table(all_cols, rows)


def get_steps(exclude_border_steps=True):
    start_step = 2 if exclude_border_steps else 0
    end_step = MAX_STEPS - 1 if exclude_border_steps else MAX_STEPS
    return start_step, end_step


def get_timing_perf(data, prefix, exclude_border_steps, output_dir):
    timing_keys = [k for k in data.keys() if k.startswith(prefix) and k not in TIMING_EXCLUDE_KEYS]

    start_step, end_step = get_steps(exclude_border_steps)

    def _plot_timing_metrics():
        _, ax = plt.subplots(figsize=(14, 7))

        for key in timing_keys:
            vals = [data[key].get(str(i)) for i in range(start_step, end_step + 1)]
            if not any(v is not None and v > TIMING_THRESHOLD for v in vals):
                continue
            label = key.replace(prefix, "")
            ax.plot(range(start_step, end_step + 1), vals, marker="o", label=label)

        ax.set_title("RL Training — Iteration Timing Breakdown", fontsize=13, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, fontsize=7, loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "timing.png"), dpi=150, bbox_inches="tight")

    def _print_table_of_timing_metrics():
        total_key = prefix + "train/total_step_time"
        iters = [str(s) for s in range(0, MAX_STEPS + 1)]

        col_w = 10
        key_w = max(len(k) for k in timing_keys + [total_key])
        experiment_name = os.path.basename(output_dir)
        key_w = max(key_w, len(experiment_name))

        header = f"{experiment_name:<{key_w}}" + "".join(f"{'Step ' + i:>{col_w}}" for i in iters)
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)

        print(
            f"{total_key:<{key_w}}"
            + "".join(
                f"{data[total_key][i]:>{col_w}.2f}" if data[total_key].get(i) is not None else f"{'—':>{col_w}}"
                for i in iters
            )
        )

        for key in timing_keys:
            vals = [data[key].get(i) for i in iters]
            if not any(isinstance(v, (int, float)) and not isinstance(v, bool) and v > TIMING_THRESHOLD for v in vals):
                continue
            row = f"{key:<{key_w}}"
            for i in iters:
                val = data[key].get(i)
                row += (
                    f"{val:>{col_w}.2f}"
                    if isinstance(val, (int, float)) and not isinstance(val, bool)
                    else f"{'—':>{col_w}}"
                )
            print(row)
        print()
        print()

    _plot_timing_metrics()
    _print_table_of_timing_metrics()


def get_perf_data(data, prefix, exclude_border_steps, output_dir):
    if prefix == "timing/":
        return get_timing_perf(data, prefix, exclude_border_steps, output_dir)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = get_args()
    results = {}
    for dirname, _dirs, files in sorted(os.walk(args.experiment_dir)):
        if 'ray-driver.log' in files:
            experiment_name = os.path.basename(os.path.dirname(dirname))
            results.setdefault(experiment_name, {})['config'] = read_config_from_log(
                os.path.join(dirname, 'ray-driver.log')
            )
        if 'metrics.json' in files:
            experiment_name = os.path.basename(dirname)
            with open(os.path.join(dirname, 'metrics.json')) as f:
                data = json.load(f)
            metrics = get_metrics_from_data(data)
            results.setdefault(experiment_name, {})['metrics'] = metrics
            if metrics and args.debug:
                get_perf_data(data, args.prefix, args.exclude_border_steps, output_dir=dirname)

    print_markdown_results(results)
