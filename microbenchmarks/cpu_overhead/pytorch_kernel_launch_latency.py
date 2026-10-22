# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

import argparse
import csv
import time

import torch


def benchmark_gemm(size, num_iters=10_000):
    """
    Measures kernel launch latency for a given GEMM size.

    Args:
        size (int): Size of the square matrix (size x size).
        num_iters (int): Number of iterations.

    Returns:
        float: Average kernel launch latency in microseconds.
    """
    device = "cuda"

    # Initialize random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Clear caching allocator and synchronize before timing
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_time = time.time()

    # Run GEMMs with synchronization after each one
    for _ in range(num_iters):
        torch.matmul(a, b)  # Small GEMM

    # Synchronize after timing
    torch.cuda.synchronize()  # Ensure execution completes
    end_time = time.time()

    # Compute average kernel launch latency in microseconds (µs)
    total_time = end_time - start_time
    avg_latency = (total_time / num_iters) * 1e6  # Convert to µs
    print(f"Average execution time for size:{size} is {avg_latency} us")

    return avg_latency


def save_results_to_csv(sizes, latencies, filename="gemm_benchmark_results.csv"):
    """Saves GEMM benchmark results to CSV file."""
    header = ["Matrix Size", "Kernel execution Latency (µs)"]
    rows = zip(sizes, latencies, strict=False)

    # Write to CSV file
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Results saved to {filename}")


def sweep_gemm_sizes(start_size=4, end_size=1024, step=2, num_iters=10_000):
    """
    Sweeps GEMM sizes from `start_size` to `end_size`, measuring kernel launch latency.

    Args:
        start_size (int): Smallest matrix size.
        end_size (int): Largest matrix size.
        step (int): Factor to multiply for each step (e.g., 2 for powers of 2).
        num_iters (int): Number of iterations per GEMM size.
    """
    sizes = []
    latencies = []

    size = start_size
    while size <= end_size:
        print(f"Benchmarking GEMM size: {size}x{size} ...")
        avg_latency = benchmark_gemm(size, num_iters)
        sizes.append(size)
        latencies.append(avg_latency)
        size *= step  # Increase size multiplicatively

    save_results_to_csv(sizes, latencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark PyTorch kernel launch overhead for small GEMMs.")
    parser.add_argument("--start_size", type=int, default=4, help="Smallest square matrix size (default: 4)")
    parser.add_argument("--end_size", type=int, default=512, help="Largest square matrix size (default: 512)")
    parser.add_argument("--step", type=int, default=2, help="Factor to multiply size (default: 2)")
    parser.add_argument("--iters", type=int, default=100000, help="Number of GEMM iterations per size")
    args = parser.parse_args()

    sweep_gemm_sizes(args.start_size, args.end_size, args.step, args.iters)
