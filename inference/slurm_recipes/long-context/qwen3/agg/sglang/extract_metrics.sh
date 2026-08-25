#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Extract specific (Metric, column) pairs from per-job aiperf CSV files.
#
# Usage:
#   ./extract_metrics.sh                          # all jobs in CWD
#   ./extract_metrics.sh <jobid>                   # single job in CWD
#   ./extract_metrics.sh <jobid> <jobid2> ...        # multiple jobs in CWD
#   ./extract_metrics.sh -d path/to/results       # all jobs under that dir
#   ./extract_metrics.sh -d path/to/results 1234  # specific jobs under that dir

set -u

ROOT="."
while [ $# -gt 0 ]; do
    case "$1" in
        -d | --dir)
            ROOT="$2"
            shift 2
            ;;
        -h | --help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *) break ;;
    esac
done

if [ ! -d "$ROOT" ]; then
    echo "Directory not found: $ROOT" >&2
    exit 1
fi

# ---- Configure target metrics here: "Metric Name|column" ----
METRICS=(
    "request_latency_p50|mean"
    "effective_total_throughput_avg|mean"
)
# -------------------------------------------------------------

CSV_REL="logs/aiperf/aggregate/profile_export_aiperf_aggregate.csv"
CONFIG_REL="config.yaml"

# Extract `--concurrency N` from inside the `benchmark:` block of config.yaml.
concurrency() {
    local cfg="$1"
    [ -f "$cfg" ] || {
        echo "NO_CONFIG"
        return
    }
    awk '
        # Enter the benchmark block on a top-level "benchmark:" line.
        /^benchmark:[[:space:]]*$/ { in_b = 1; next }
        # Leave on the next top-level key (non-indented, non-comment, non-blank).
        in_b && /^[^[:space:]#]/ { in_b = 0 }
        in_b {
            if (match($0, /--concurrency[[:space:]]+[0-9]+/)) {
                s = substr($0, RSTART, RLENGTH)
                sub(/--concurrency[[:space:]]+/, "", s)
                print s
                exit
            }
        }
    ' "$cfg"
}

extract() {
    local csv="$1" metric="$2" column="$3"
    awk -v metric="$metric" -v column="$column" '
        BEGIN { FS="," }
        # Header rows always start with "Metric," or "Endpoint,"
        /^Metric,/ || /^Endpoint,/ || /^metric,/ {
            delete idx
            for (i = 1; i <= NF; i++) idx[$i] = i
            next
        }
        $1 == metric {
            if (column in idx) {
                v = $(idx[column])
                if (v == "") v = "(empty)"
                print v
                exit
            } else {
                print "(no column \"" column "\" in this section)"
                exit
            }
        }
        END {
            # Nothing matched
        }
    ' "$csv"
}

# Collect jobids
if [ $# -gt 0 ]; then
    JOBS=("$@")
else
    JOBS=()
    for d in "$ROOT"/*/; do
        d="${d%/}"
        name="${d##*/}"
        [[ $name =~ ^[0-9]+$ ]] && [ -f "$d/$CSV_REL" ] && JOBS+=("$name")
    done
fi

if [ ${#JOBS[@]} -eq 0 ]; then
    echo "No jobs found in $ROOT." >&2
    exit 1
fi

# Collect everything into a TSV buffer, then pad columns at the end.
SEP=$'\t'
buffer=""

# Header
row="concurrency"
for pair in "${METRICS[@]}"; do
    metric="${pair%%|*}"
    column="${pair##*|}"
    row+="${SEP}${metric} [${column}]"
done
row+="${SEP}jobid"
buffer+="${row}"$'\n'

# Data rows
for job in "${JOBS[@]}"; do
    csv="$ROOT/$job/$CSV_REL"
    conc="$(concurrency "$ROOT/$job/$CONFIG_REL")"
    [ -z "$conc" ] && conc="NOT_FOUND"
    row="$conc"
    if [ ! -f "$csv" ]; then
        for _ in "${METRICS[@]}"; do row+="${SEP}NO_FILE"; done
    else
        for pair in "${METRICS[@]}"; do
            metric="${pair%%|*}"
            column="${pair##*|}"
            val="$(extract "$csv" "$metric" "$column")"
            [ -z "$val" ] && val="NOT_FOUND"
            row+="${SEP}${val}"
        done
    fi
    row+="${SEP}${job}"
    buffer+="${row}"$'\n'
done

# Pad to per-column max width, two-space gutter, right-align everything
# except the last column.
printf "%s" "$buffer" | awk -F'\t' '
    { rows[NR] = $0; nrows = NR
      for (i = 1; i <= NF; i++) {
          if (length($i) > w[i]) w[i] = length($i)
          if (i > ncols) ncols = i
      }
    }
    END {
        for (r = 1; r <= nrows; r++) {
            n = split(rows[r], f, "\t")
            for (i = 1; i <= n; i++) {
                if (i == n) printf "%s",  f[i]
                else        printf "%-*s  ", w[i], f[i]
            }
            printf "\n"
        }
    }
'
