#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build and push Docker images for all 6 Nemotron task domains.
# Run this on a machine with Docker. Each domain shares one image.
#
# Usage:
#   bash scripts/build_nemotron_images.sh <registry_prefix> [task_dir]
#
# Example:
#   bash scripts/build_nemotron_images.sh <your-registry>/tasktrace-capture
#   bash scripts/build_nemotron_images.sh <your-registry>/nemotron
#   bash scripts/build_nemotron_images.sh <your-registry>/tasktrace-capture ./nemotron_tasks
set -euo pipefail

REGISTRY="${1:?Usage: $0 <registry_prefix> [task_dir]}"
TASK_DIR="${2:-./nemotron_tasks}"
DOMAINS=(data_processing data_science debugging file_operations scientific_computing security)

echo "Registry: $REGISTRY"
echo "Task dir: $TASK_DIR"
echo ""

for domain in "${DOMAINS[@]}"; do
    dockerfile_dir="$TASK_DIR/skill_based/mixed/$domain/${domain}_task_0001/environment"
    image="$REGISTRY/nemotron-${domain}:latest"

    if [[ ! -f "$dockerfile_dir/Dockerfile" ]]; then
        echo "SKIP $domain (no Dockerfile at $dockerfile_dir)"
        continue
    fi

    echo "Building $domain → $image"
    docker build -t "$image" "$dockerfile_dir"
    echo "Pushing $image"
    docker push "$image"
    echo "Done: $domain"
    echo ""
done

echo "All images pushed. Now update task.toml files:"
echo ""
echo "  uv run tasktrace-capture-setup-nemotron --docker-image $REGISTRY/nemotron-DOMAIN:latest"
echo ""
echo "Or patch manually:"
echo "  for domain in ${DOMAINS[*]}; do"
echo "    find $TASK_DIR/skill_based/mixed/\$domain -name task.toml -exec \\"
# shellcheck disable=SC2016
printf '      sed -i "s|^\\[environment\\]$|[environment]\\ndocker_image = \\"%s/nemotron-${domain}:latest\\"|" {} +\n' "$REGISTRY"
echo "  done"
