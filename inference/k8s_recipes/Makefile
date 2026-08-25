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

# Validation targets for the Kubernetes recipe collection.
# Every target is safe to run locally and is what CI invokes.
SHELL := /bin/bash
CELLS := $(shell find recipes -name recipe.yaml 2>/dev/null | xargs -n1 dirname 2>/dev/null)
HAS_INTERNAL_TEST_ASSETS := $(shell test -d scripts/fixtures -a -d scripts/testdata && echo 1)
HAS_ABSENCE_BASELINE := $(shell test -f scripts/testdata/absence_defaults_baseline.json && echo 1)
HAS_RESOURCE_POLICY := $(shell test -f scripts/testdata/resource_policy.json && echo 1)
HAS_MULTIARCH_MANIFEST := $(shell test -f serving/tasks/image-digest-manifest.json && echo 1)
HAS_MEASURED_SIGMA := $(shell test -f analysis/measured_sigma.json && echo 1)
SELFTEST_TMPDIR := $(CURDIR)/.selftest-runtime


.PHONY: reproduce reproduce-check
.PHONY: help all validate link-check contract-check check-invariants provenance-check render-check lint test matrix matrix-check results-check export export-check multiarch-check absence-check resource-check ci
help:
	@echo "targets: validate | render-check | lint | matrix | all"
	@echo "cells found: $(words $(CELLS))"

all: validate render-check lint matrix

## schema-validate each recipe.yaml against scenario-<metrics_family>.yaml
validate:
	@python3 scripts/validate.py

## re-render each cell and assert the committed rendered/ matches (no drift)
render-check:
	@set -e; for c in $(CELLS); do \
	  [ -d "$$c/rendered" ] || { echo "skip (declaration-only, no rendered/ yet): $$c"; continue; }; \
	  tmp=$$(mktemp -d); \
	  scripts/render.sh $$c --to $$tmp >/dev/null; \
	  if ! diff -rq "$$c/rendered" "$$tmp" >/dev/null 2>&1; then \
	    echo "DRIFT: $$c — committed rendered/ != re-render (run scripts/render.sh and commit)"; \
	    rm -rf "$$tmp"; exit 1; fi; \
	  rm -rf "$$tmp"; \
	done; echo "render-check OK"

## kubeconform the rendered manifests (skips cleanly if not installed)
lint:
	@python3 scripts/lint_manifests.py
	@command -v kubeconform >/dev/null 2>&1 && { \
	  for c in $(CELLS); do \
	    for f in $$c/rendered/*.yaml; do \
	      sed 's/\$${\([^}]*\)}/placeholder/g' "$$f" | \
	      kubeconform -strict -ignore-missing-schemas - || exit 1; \
	    done; \
	  done; \
	  echo "kubeconform OK"; } || echo "(kubeconform not installed — structural lint above; install for stricter schema checks)"

## rebuild catalog.json + inject the matrix into README.md from all envelopes
matrix:
	@python3 scripts/build_catalog.py

## CI drift guard: fail if the committed matrix/catalog is stale OR two different recipe definitions share a
## benchmark_id. Read-only — never mutates tracked files (safe in a shared worktree).
matrix-check:
	@python3 scripts/build_catalog.py --check
	@echo "matrix-check OK (README matrix + catalog.json up to date)"

## rebuild the upload-ready dataset export: per-run results/<run_id>/export.csv (GITIGNORED, ephemeral DB-upload
## unit) co-located with the run's artifacts. Schema per (scenario,goal); tidy grain (one row/run x rung).
export:
	@python3 scripts/export_dataset.py

## CI guard: every run renders a valid results/<run_id>/export.csv + no stale committed export.csv/upload.csv/export/ linger.
export-check:
	@python3 scripts/export_dataset.py --check

## regenerate every cell README's "How to reproduce" three-step block (+ the group block when the cell
## belongs to a family). scripts/reproduce.py is the SINGLE generator; hand-edits inside the markers are lost.
reproduce:
	@python3 scripts/reproduce.py --write

## CI guard: every recipe page leads with the CURRENT three steps and carries no hand-kubectl / render /
## deploy / stage / sweep / fetch / hand-aggregate step that `install` + `run` already do.
reproduce-check:
	@python3 scripts/reproduce.py

## every ```bash block in the docs must survive a literal copy-paste into zsh: a trailing `#` comment is
## NOT a comment there (interactive_comments is off), it becomes ARGUMENTS -- and `(no flags)` then parses
## as a glob qualifier, so the quickstart failed with "zsh: unknown sort specifier". Read-only.
pasteable-check:
	@python3 scripts/check_pasteable.py

## enforce the recipe contract: required files + sections + exemplar block
contract-check:
	@python3 scripts/contract_check.py

## determinism proof: regenerate RESULTS.md + record.json from COMMITTED curated data for every on-pipeline
## cell and byte-diff — fails on any drift (proves nothing was hand-edited). Read-only.
results-check:
	@python3 scripts/results_check.py

## enforce cross-cutting invariants that the schema cannot express:
## content-pinning at runs+, gpu.count==tp, valid exemplar metric, namespace parameterized
check-invariants:
	@python3 scripts/check_invariants.py

## every published result (runs+) must cite the CURRENT recipe_hash AND have a matching recipe_hash in its
## durable runs/index.jsonl ledger — proves the number is tied to this exact recipe (see scripts/provenance.py)
provenance-check:
	@python3 scripts/provenance.py --check

## a published headline must be a MEAN OF N WITH A CI, never a single run: refuse to publish a
## single-run headline for any cell whose MEASURED run-to-run sigma exceeds its own tolerance
## (see scripts/pooled_stats.py + analysis/measured_sigma.json). Read-only.
pooled-check:
	@if [ "$(HAS_MEASURED_SIGMA)" = 1 ]; then python3 scripts/check_pooled_publication.py; else echo "skip pooled-check (internal measured-sigma registry not shipped)"; fi

## A read that did not land must be representable as UNKNOWN, distinct from zero/empty. Fails on any
## NEW silent-absence default (error path returning []/0/""/False, a discarded return code, a broad
## `except: pass` around a read) in a module that RENDERS cluster state or GATES a published number.
## Allowlist-based (scripts/testdata/absence_defaults_baseline.json); known debt is ratcheted, never grows.
absence-check:
	@python3 scripts/check_absence_defaults.py

## Create without destroy: every resource the tooling creates (k8s kinds, detached local processes) must
## have a NAMED reclaim path that still exists, or an explicit "unreclaimed, because…" reason. A new
## resource type — or a new file creating an existing one — fails until someone decides.
## (112 leaked PVCs / ~6 TiB, 72 orphaned processes, a lease held 3h43m: all creates with no destroy.)
resource-check:
	@python3 scripts/check_create_destroy.py

## Regression tests for the deterministic tools (no cluster needed).
test: export TMPDIR := $(SELFTEST_TMPDIR)
test:
	@mkdir -p "$(TMPDIR)"
	@python3 scripts/selftest_sglang_cache_contract.py
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest.py; else echo "skip selftest.py (internal fixtures not shipped)"; fi
	@python3 scripts/selftest_provenance.py
	@python3 scripts/selftest_launch_attestation.py
	@python3 scripts/selftest_benchmark_identity.py
	@python3 scripts/selftest_onboarding.py
	@python3 scripts/selftest_context.py
	@python3 scripts/selftest_preflight_ux.py
	@python3 scripts/selftest_recovery_meta.py
	@python3 scripts/selftest_dispatcher_help.py
	@python3 scripts/selftest_cancel_run.py
	@python3 scripts/selftest_llmb_bench.py
	@python3 scripts/selftest_resilient_submit.py
	@if [ "$(HAS_RESOURCE_POLICY)" = 1 ]; then python3 scripts/selftest_governor.py; else echo "skip selftest_governor.py (internal governor policy not shipped)"; fi
	@python3 scripts/selftest_crashloop.py
	@python3 scripts/selftest_load_gate.py
	@python3 scripts/selftest_adopt_server.py
	@python3 scripts/selftest_run_owner.py
	@python3 scripts/selftest_env_override.py
	@python3 scripts/selftest_charts.py
	@python3 scripts/selftest_goal_handlers.py
	@python3 scripts/selftest_rung_coverage.py
	@python3 scripts/selftest_progress_signal.py
	@python3 scripts/selftest_cell_identity.py
	@python3 scripts/selftest_progress_liveness.py
	@python3 scripts/selftest_sweep_runid.py
	@python3 scripts/selftest_gpu_basis.py
	@python3 scripts/selftest_fleet.py
	@python3 scripts/selftest_fleet_viewport.py
	@python3 scripts/selftest_archive_run.py
	@python3 scripts/selftest_capacity_guards.py
	@python3 scripts/selftest_aggregate_cell.py
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_results_check.py; else echo "skip selftest_results_check.py (internal test assets not shipped)"; fi
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_export_dataset.py; else echo "skip selftest_export_dataset.py (internal test assets not shipped)"; fi
	@python3 scripts/selftest_runner_identity.py
	@python3 scripts/selftest_cluster_fingerprint.py
	@if [ "$(HAS_MULTIARCH_MANIFEST)" = 1 ]; then python3 scripts/selftest_multiarch_manifest.py; else echo "skip selftest_multiarch_manifest.py (internal image manifest not shipped)"; fi
	@python3 scripts/selftest_parallel_repro_spread.py
	@if [ "$(HAS_MEASURED_SIGMA)" = 1 ]; then python3 scripts/selftest_pooled_stats.py; else echo "skip selftest_pooled_stats.py (internal measured-sigma registry not shipped)"; fi
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_detach.py; else echo "skip selftest_detach.py (internal test assets not shipped)"; fi
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_capability_registry.py; else echo "skip selftest_capability_registry.py (internal test assets not shipped)"; fi
	@python3 scripts/selftest_preflight_gates.py
	@python3 scripts/selftest_cluster_readiness.py
	@python3 scripts/selftest_manifest_vars.py
	@python3 scripts/selftest_fetch_receipt.py
	@python3 scripts/selftest_resilient_results.py
	@python3 scripts/selftest_wizard_init.py
	@python3 scripts/selftest_run_id.py
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_install_recipes.py; else echo "skip selftest_install_recipes.py (internal test assets not shipped)"; fi
	@python3 scripts/selftest_install_interactive.py
	@if [ "$(HAS_INTERNAL_TEST_ASSETS)" = 1 ]; then python3 scripts/selftest_per_recipe_cache.py; else echo "skip selftest_per_recipe_cache.py (internal test assets not shipped)"; fi
	@python3 scripts/selftest_cache_claim_agreement.py
	@python3 scripts/selftest_run_result.py
	@python3 scripts/selftest_run_verdict.py
	@python3 scripts/selftest_reproduce.py
	@python3 scripts/selftest_reclaim_storage.py
	@python3 scripts/selftest_check_links.py
	@rm -rf "$(TMPDIR)"

## Check local Markdown links without requiring network access.
link-check:
	@python3 scripts/check_links.py

## Validate the task-image manifest structure without registry access. Legacy single-architecture
## manifests produce an advisory; partially migrated manifests fail.
multiarch-check:
	@if [ "$(HAS_MULTIARCH_MANIFEST)" = 1 ]; then \
	  python3 scripts/verify_multiarch_manifest.py --offline && echo "multiarch-check OK (task-image manifest shape)"; \
	else \
	  echo "skip multiarch-check (internal image manifest not shipped)"; \
	fi

## what CI runs (never mutates tracked files)
ci: validate link-check pasteable-check contract-check reproduce-check check-invariants provenance-check render-check lint test matrix-check results-check pooled-check export-check multiarch-check
	@if [ "$(HAS_ABSENCE_BASELINE)" = 1 ]; then $(MAKE) absence-check; else echo "skip absence-check (internal policy baseline not shipped)"; fi
	@if [ "$(HAS_RESOURCE_POLICY)" = 1 ]; then $(MAKE) resource-check; else echo "skip resource-check (internal resource policy not shipped)"; fi
