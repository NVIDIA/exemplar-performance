#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import json
import os
import re
import statistics

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ray/wandb sometimes flush other log lines into the middle of the pretty-printed
# config write (no newline in between), so a naive "drop the whole line" filter
# silently deletes real config content. Match the noise and strip only that part.
CONFIG_LOG_NOISE_PATTERNS = [
    re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+\s+INFO worker\.py:.*'),
    re.compile(r'wandb:.*'),
]


def read(file_path):
    with open(file_path) as f:
        return f.read().strip().split('\n')


def extract_experiment_config(experiment_job_id, exp_name, config):
    """Print a one-row markdown table summarizing the experiment config."""
    cluster = config.get('cluster', {})
    policy = config.get('policy', {})
    num_gpus = cluster.get('gpus_per_node', 0) * cluster.get('num_nodes', 0)
    generation_resources = policy.get('generation', {}).get('colocated', {}).get('resources', {})
    g_gpus = generation_resources.get('gpus_per_node', 0) * generation_resources.get('num_nodes', 0)
    t_gpus = num_gpus - g_gpus

    grpo = config.get('grpo', {})
    g_gbs = grpo.get('num_prompts_per_step', 0) * grpo.get('num_generations_per_prompt', 0)
    t_gbs = policy.get('train_global_batch_size')

    max_seqlen = policy.get('max_total_sequence_length')
    megatron = policy.get('megatron_cfg', {})
    vllm_cfg = policy.get('generation', {}).get('vllm_cfg', {})
    t_parallelism = (
        f"TP{megatron.get('tensor_model_parallel_size', 1)}_"
        f"PP{megatron.get('pipeline_model_parallel_size', 1)}_"
        f"CP{megatron.get('context_parallel_size', 1)}_"
        f"EP{megatron.get('expert_model_parallel_size', 1)}"
    )
    g_parallelism = (
        f"TP{vllm_cfg.get('tensor_parallel_size', 1)}_"
        f"PP{vllm_cfg.get('pipeline_parallel_size', 1)}_"
        f"EP{vllm_cfg.get('expert_parallel_size', 1)}"
    )
    return {
        'num_gpus': num_gpus,
        'g_gpus': g_gpus,
        't_gpus': t_gpus,
        'g_gbs': g_gbs,
        't_gbs': t_gbs,
        'max_seqlen': max_seqlen,
        't_parallelism': t_parallelism,
        'g_parallelism': g_parallelism,
    }


def read_full_config_from_log(log_lines, stage="swe1"):
    # currently last line is hardcoded
    if stage == "swe1":
        last_line = "\'train_micro_batch_size\':"
    elif stage == "swe2":
        last_line = "\'sif_dir\':"
    else:
        raise ValueError(f"Unknown stage: {stage}")

    config_lines = []
    in_config_block = False
    for line in log_lines:
        if in_config_block:
            for pattern in CONFIG_LOG_NOISE_PATTERNS:
                line = pattern.sub("", line)
            config_lines.append(line)
        if "Final config:" in line:
            in_config_block = True
        if last_line in line:
            break
    if config_lines == []:
        return None
    return ast.literal_eval("".join(config_lines))


BULLET_LINE_PATTERN = re.compile(r"\s*[•-]\s*(.+?):\s*([-+\d.eE]+)")
REQUIRED_STEP_FIELDS = (
    'Avg Reward',
    'Total step time',
    'policy_training',
    'weight_sync',
    'policy_and_reference_logprobs',
    'logprob_inference_prep',
    'training_prep',
    'exposed_generation',
    'Training Worker Group (Tokens/sec/gpu)',
    'Generation Worker Group (Tokens/sec/gpu)',
)


def extract_results(log_lines):
    """Single pass over the log, one dict per training step (keyed by 1-based
    step number) bounded by consecutive 'Training Results:' headers, keeping
    only REQUIRED_STEP_FIELDS."""
    results = {}
    current = None
    step = 0
    for line in log_lines:
        if 'Training Results:' in line:
            if current is not None:
                results[step] = current
            step += 1
            current = {}
            continue

        if current is None:
            continue

        m = BULLET_LINE_PATTERN.match(line)
        if m:
            key = m.group(1).strip()
            if key in REQUIRED_STEP_FIELDS:
                value = m.group(2).strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                current[key] = value

    if current is not None:
        results[step] = current
    return results


def dump_json(data, output_path):
    """Write data (config and/or step_metrics) to a JSON file."""
    with open(output_path, 'w') as f:
        print(f"Writing to {output_path}")
        json.dump(data, f, indent=2)


TIMING_PLOT_FIELDS = (
    'Total step time',
    'policy_training',
    'weight_sync',
    'policy_and_reference_logprobs',
    'logprob_inference_prep',
    'training_prep',
    'exposed_generation',
)

THROUGHPUT_FIELDS = (
    'Training Worker Group (Tokens/sec/gpu)',
    'Generation Worker Group (Tokens/sec/gpu)',
)

STATS_TABLE_FIELDS = TIMING_PLOT_FIELDS + THROUGHPUT_FIELDS


def plot_timing_vs_iterations(step_metrics, experiment_job_id, output_path):
    """Plot each timing field vs. iteration (log-scale) and save to output_path."""
    iterations = sorted(step_metrics.keys())

    fig, ax = plt.subplots(figsize=(14, 7))
    for field in TIMING_PLOT_FIELDS:
        values = [step_metrics[i].get(field) for i in iterations]
        ax.plot(iterations, values, marker='o', markersize=3, linewidth=1.2, label=field)

    ax.set_xlabel('Iteration (step)')
    ax.set_ylabel('seconds')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(
        f'Timing Breakdown per Iteration (n={len(iterations)} steps) - job {experiment_job_id}',
        fontsize=13,
        fontweight='bold',
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_total_step_time_ecdf(jobs_step_metrics, output_path):
    """Plot the empirical CDF of 'Total step time' for each job, overlaid."""
    field = 'Total step time'

    marked_percentiles = (50, 60, 75, 80, 90, 99)

    fig, ax = plt.subplots(figsize=(10, 6))
    table_rows = []
    row_colors = []
    for job_id, step_metrics in sorted(jobs_step_metrics.items(), key=lambda kv: int(kv[0])):
        values = sorted(m[field] for m in step_metrics.values() if isinstance(m.get(field), (int, float)))
        if not values:
            continue
        n = len(values)
        y = [(i + 1) / n for i in range(n)]
        (line,) = ax.step(values, y, where='post', marker='.', markersize=4, linewidth=1.2)

        color = line.get_color()
        table_rows.append([str(job_id), str(n)] + [f'{_percentile(values, p):.1f}' for p in marked_percentiles])
        row_colors.append(color)

    # Render percentile summaries as a table (axes-fraction coords) in empty
    # plot space, rather than labeling each point directly, since the jobs'
    # percentile values are too close together on this scale to label
    # in-place without overlapping.
    if table_rows:
        col_labels = ['job_id', 'n'] + [f'p{p}' for p in marked_percentiles]
        table = ax.table(
            cellText=table_rows,
            colLabels=col_labels,
            cellLoc='center',
            colLoc='center',
            bbox=[0.60, 0.15, 0.38, 0.05 + 0.06 * len(table_rows)],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for row_index, color in enumerate(row_colors, start=1):
            for col_index in range(len(col_labels)):
                table[row_index, col_index].get_text().set_color(color)

    ax.set_xlabel('Total step time (s)')
    ax.set_ylabel('ECDF')
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 2, 3, 5, 7)))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title('ECDF of Total Step Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    print(f"Saving plot to {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _percentile(values, p):
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method='inclusive')[p - 1]


def compute_timing_stats(jobs_step_metrics):
    """Build {field: {job_id: {num_iter, min, max, mean, std, p50, p60, p75, p80, p90, p99}}} for
    each field in STATS_TABLE_FIELDS, across all jobs."""
    stats = {}
    for field in STATS_TABLE_FIELDS:
        field_stats = {}
        for job_id, step_metrics in jobs_step_metrics.items():
            values = [m[field] for m in step_metrics.values() if isinstance(m.get(field), (int, float))]
            if not values:
                continue
            field_stats[job_id] = {
                'num_iter': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'p50': _percentile(values, 50),
                'p60': _percentile(values, 60),
                'p75': _percentile(values, 75),
                'p80': _percentile(values, 80),
                'p90': _percentile(values, 90),
                'p99': _percentile(values, 99),
            }
        stats[field] = field_stats
    return stats


TIMING_STATS_TABLE_COLS = ('job_id', 'num_iter', 'min', 'max', 'mean', 'p50', 'p60', 'p75', 'p80', 'p90', 'p99')


def build_timing_stats_rows(field_stats):
    """Build rows (job_id, num_iter, min, max, mean, p50, p60, p75, p80, p90, p99) for one
    timing field, with trailing 'mean' and 'std' summary rows across jobs. Numeric cells are
    left as numbers (rounded to 2 decimal places) so callers can format them for display or
    write them into a spreadsheet."""
    numeric_cols = TIMING_STATS_TABLE_COLS[1:]

    def r(x):
        return x if isinstance(x, int) else round(x, 2)

    rows = []
    for job_id, s in field_stats.items():
        rows.append((str(job_id),) + tuple(r(s[c]) for c in numeric_cols))

    if rows:
        job_values = list(field_stats.values())
        rows.append(('mean',) + tuple(r(statistics.mean(s[c] for s in job_values)) for c in numeric_cols))
        rows.append(
            ('std',)
            + tuple(r(statistics.stdev(s[c] for s in job_values)) if len(job_values) > 1 else 0.0 for c in numeric_cols)
        )
    return rows


def print_timing_stats_tables(timing_stats):
    """Print one box-drawing table per timing field, one row per job_id:
    job_id | num_iter | min | max | mean | p50 | p60 | p75 | p80 | p90 | p99."""
    cols = TIMING_STATS_TABLE_COLS

    for field, field_stats in timing_stats.items():
        rows = [
            tuple(v if c == 'job_id' else f"{v:.2f}" for c, v in zip(cols, row, strict=True))
            for row in build_timing_stats_rows(field_stats)
        ]

        widths = {c: len(c) + 2 for c in cols}
        for row in rows:
            for c, v in zip(cols, row, strict=True):
                widths[c] = max(widths[c], len(v) + 2)

        def hline(left, mid, right, widths=widths):
            return left + mid.join('─' * widths[c] for c in cols) + right

        def fmt_row(values, widths=widths):
            return '│' + '│'.join(f'{v:^{widths[c]}}' for c, v in zip(cols, values, strict=True)) + '│'

        lines = [f'{field}:', hline('┌', '┬', '┐'), fmt_row(cols), hline('├', '┼', '┤')]
        for i, row in enumerate(rows):
            lines.append(fmt_row(row))
            if i < len(rows) - 1:
                lines.append(hline('├', '┼', '┤'))
        lines.append(hline('└', '┴', '┘'))
        print('\n'.join(lines))
        print()


SUMMARY_TABLE_ROWS = (
    ('Total num iterations (mean per job)', 'Total step time', 'num_iter'),
    ('Total step mean', 'Total step time', 'mean'),
    ('Total step p50', 'Total step time', 'p50'),
    ('Policy training p50', 'policy_training', 'p50'),
    ('weight_sync p50', 'weight_sync', 'p50'),
    ('policy_and_reference_logprobs p50', 'policy_and_reference_logprobs', 'p50'),
    ('logprob_inference_prep p50', 'logprob_inference_prep', 'p50'),
    ('training_prep p50', 'training_prep', 'p50'),
    ('exposed_generation p50', 'exposed_generation', 'p50'),
    ('Training Worker Group (Tokens/sec/gpu) mean', 'Training Worker Group (Tokens/sec/gpu)', 'mean'),
    ('Generation Worker Group (Tokens/sec/gpu) mean', 'Generation Worker Group (Tokens/sec/gpu)', 'mean'),
)


def build_summary_table(timing_stats, label):
    """Build (Metric, label) rows for SUMMARY_TABLE_ROWS, each value averaged across jobs
    (mean of each job's stat, i.e. the same cross-job 'mean' summary row used elsewhere),
    rounded to 2 decimal places (num_iter is kept as an int)."""
    rows = [('Metric', label)]
    for row_label, field, stat in SUMMARY_TABLE_ROWS:
        field_stats = timing_stats.get(field, {})
        job_values = list(field_stats.values())
        if not job_values:
            value = None
        elif stat == 'num_iter':
            value = round(statistics.mean(s[stat] for s in job_values))
        else:
            value = round(statistics.mean(s[stat] for s in job_values), 2)
        rows.append((row_label, value))
    return rows


def print_summary_table(timing_stats, label='Value'):
    """Print the final Metric/label summary table."""
    rows = build_summary_table(timing_stats, label)
    fmt_rows = [rows[0]] + [
        (r, f'{v:.2f}' if isinstance(v, float) else str(v) if v is not None else '') for r, v in rows[1:]
    ]

    widths = [max(len(str(r[0])) for r in fmt_rows) + 2, max(len(str(r[1])) for r in fmt_rows) + 2]

    def hline(left, mid, right):
        return left + mid.join('─' * w for w in widths) + right

    def fmt_row(values):
        return '│' + '│'.join(f'{v:^{w}}' for v, w in zip(values, widths, strict=True)) + '│'

    lines = [hline('┌', '┬', '┐'), fmt_row(fmt_rows[0]), hline('├', '┼', '┤')]
    for i, row in enumerate(fmt_rows[1:]):
        lines.append(fmt_row(row))
        if i < len(fmt_rows) - 2:
            lines.append(hline('├', '┼', '┤'))
    lines.append(hline('└', '┴', '┘'))
    print('\n'.join(lines))
    print()


def dump_xlsx_tables(timing_stats, output_path, summary_label='Value'):
    """Write all timing field tables to a single worksheet, stacked one after another
    with a field-name header row and a blank row between tables. Each table holds
    job_id | num_iter | min | max | mean | p50 | p60 | p75 | p80 | p90 | p99.
    A final Metric/summary_label table is appended at the end."""
    import openpyxl

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'timing_stats'
    for field, field_stats in timing_stats.items():
        sheet.append((field,))
        sheet.append(TIMING_STATS_TABLE_COLS)
        for row in build_timing_stats_rows(field_stats):
            sheet.append(row)
        sheet.append(())

    sheet.append(('Summary',))
    for row in build_summary_table(timing_stats, summary_label):
        sheet.append(row)

    print(f"Writing timing stats tables to {output_path}")
    wb.save(output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--plot_step_metrics', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--summary_label', type=str, default='Value')
    return parser.parse_args()


def main():
    args = get_args()
    jobs_step_metrics = {}
    for root, _dirs, files in os.walk(args.input_dir):
        if 'ray-driver.log' in files:
            experiment_dir = os.path.dirname(os.path.dirname(os.path.dirname(root)))
            experiment_job_id = root.split('/')[-1].split('-')[0]
            ray_driver_log_path = os.path.join(root, 'ray-driver.log')
            log_lines = read(ray_driver_log_path)
            exp_name = os.path.basename(os.path.dirname(root))
            stage = exp_name.rsplit('-', 1)[-1]

            # Parse experiment config from ray-driver.log file
            try:
                config = read_full_config_from_log(log_lines, stage)
            except (SyntaxError, ValueError) as e:
                print(f"Warning: failed to parse config block in {ray_driver_log_path} ({e}), skipping this job.")
                continue
            if config is None:
                continue
            config = extract_experiment_config(experiment_job_id, exp_name, config)
            # Parse training results from ray-driver.log file
            step_metrics = extract_results(log_lines)
            jobs_step_metrics[experiment_job_id] = step_metrics

            # Dump step metrics to a JSON file
            json_data = {
                'config': config,
                'step_metrics': step_metrics,
            }
            json_output_path = os.path.join(experiment_dir, f'results_{experiment_job_id}.json')
            dump_json(json_data, json_output_path)

            if args.plot_step_metrics:
                # Dump timing-vs-iterations plot
                plot_output_path = os.path.join(experiment_dir, f'timing_vs_iterations_{experiment_job_id}.png')
                plot_timing_vs_iterations(step_metrics, experiment_job_id, plot_output_path)

    timing_stats = compute_timing_stats(jobs_step_metrics)
    print_timing_stats_tables(timing_stats)
    print_summary_table(timing_stats, args.summary_label)

    timing_stats_output_path = os.path.join(args.input_dir, 'timing_stats_summary.json')
    dump_json(timing_stats, timing_stats_output_path)

    xlsx_output_path = os.path.join(args.input_dir, 'timing_stats_tables.xlsx')
    dump_xlsx_tables(timing_stats, xlsx_output_path, args.summary_label)

    if args.plot_step_metrics:
        ecdf_output_path = os.path.join(args.input_dir, 'total_step_time_ecdf.png')
        plot_total_step_time_ecdf(jobs_step_metrics, ecdf_output_path)


if __name__ == '__main__':
    main()
