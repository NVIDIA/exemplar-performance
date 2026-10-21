# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HuggingFace download/verify workflow."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmb_install.downloads import huggingface as hf

REPO_ID = "meta-llama/Meta-Llama-3-70B"
NEMOTRON_REPO_ID = "nvidia/Nemotron-4-340B-Base"
WORKLOAD_KEY = "workload1"


def _workloads_with_downloads(downloads):
    return {WORKLOAD_KEY: {"downloads": downloads}}


def _asset_specs(repo_id, *assets):
    return (hf.HuggingFaceAssetSpec(repo_id=repo_id, assets=frozenset(assets)),)


def _assert_hf_env(test_case, cache_dir):
    test_case.assertEqual(os.environ.get("HF_HOME"), cache_dir)
    test_case.assertEqual(os.environ.get("HF_HUB_CACHE"), os.path.join(cache_dir, "hub"))


def _assert_offline_call(test_case, mocked_loader, repo_id):
    args, kwargs = mocked_loader.from_pretrained.call_args
    test_case.assertEqual(args[0], repo_id)
    test_case.assertTrue(kwargs["local_files_only"])
    test_case.assertTrue(kwargs["trust_remote_code"])


class TestNormalizeHfToken(unittest.TestCase):
    """Tests for HuggingFace token normalization."""

    def test_strips_whitespace(self):
        self.assertEqual(hf._normalize_hf_token("  hf_token  "), "hf_token")
        self.assertEqual(hf._normalize_hf_token("\thf_token\t"), "hf_token")

    def test_strips_newlines(self):
        self.assertEqual(hf._normalize_hf_token("hf_token\n"), "hf_token")
        self.assertEqual(hf._normalize_hf_token("hf_token\r\n"), "hf_token")

    def test_strips_quotes(self):
        self.assertEqual(hf._normalize_hf_token('"hf_token"'), "hf_token")
        self.assertEqual(hf._normalize_hf_token("'hf_token'"), "hf_token")

    def test_strips_quotes_with_whitespace(self):
        self.assertEqual(hf._normalize_hf_token('  "hf_token"  '), "hf_token")

    def test_returns_none_for_whitespace_only(self):
        self.assertIsNone(hf._normalize_hf_token("   "))
        self.assertIsNone(hf._normalize_hf_token("\n"))

    def test_preserves_valid_token(self):
        self.assertEqual(hf._normalize_hf_token("hf_abcd1234"), "hf_abcd1234")


class TestHuggingFaceDownloadPlan(unittest.TestCase):
    """Tests for HuggingFace download plan parsing."""

    def test_cache_assets_default_to_tokenizer_and_config(self):
        workloads = _workloads_with_downloads({"huggingface": [{"repo_id": REPO_ID}]})

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.cache), 1)
        self.assertEqual(plan.cache[0].repo_id, REPO_ID)
        self.assertEqual(plan.cache[0].assets, frozenset({"tokenizer", "config"}))

    def test_map_form_cache_entries_match_legacy_list_form(self):
        list_form = _workloads_with_downloads({"huggingface": [{"repo_id": REPO_ID, "assets": ["config"]}]})
        map_form = _workloads_with_downloads({"huggingface": {"cache": [{"repo_id": REPO_ID, "assets": ["config"]}]}})

        list_plan = hf.build_hf_download_plan(list_form, [WORKLOAD_KEY])
        map_plan = hf.build_hf_download_plan(map_form, [WORKLOAD_KEY])

        self.assertEqual(list_plan, map_plan)

    def test_repo_entry_defaults(self):
        workloads = _workloads_with_downloads(
            {"huggingface": {"repos": [{"repo_id": "org/model-name", "repo_type": "model"}]}}
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

        self.assertEqual(plan.cache, ())
        self.assertEqual(len(plan.repos), 1)
        repo = plan.repos[0]
        self.assertEqual(repo.repo_id, "org/model-name")
        self.assertEqual(repo.repo_type, "model")
        self.assertIsNone(repo.revision)
        self.assertEqual(repo.target, "workload")
        self.assertEqual(repo.name, "model-name")

    def test_repo_entry_requires_repo_type(self):
        workloads = _workloads_with_downloads({"huggingface": {"repos": [{"repo_id": "org/model-name"}]}})

        with self.assertRaisesRegex(ValueError, "repo_type"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_repo_gpu_filtering_happens_in_plan(self):
        workloads = _workloads_with_downloads(
            {"huggingface": {"repos": [{"repo_id": REPO_ID, "repo_type": "model", "when": {"gpu": ["gb200"]}}]}}
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

        self.assertEqual(plan.cache, ())
        self.assertEqual(plan.repos, ())

    def test_repo_entry_validated_even_when_gpu_filtered(self):
        # A malformed entry must fail on every GPU type, not only on matching ones.
        workloads = _workloads_with_downloads(
            {
                "huggingface": {
                    "repos": [
                        {
                            "repo_id": REPO_ID,
                            "repo_type": "model",
                            "target": "bogus",
                            "when": {"gpu": ["gb200"]},
                        }
                    ]
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "target"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY], gpu_type="h100")

    def test_cache_assets_merge_by_revision(self):
        workloads = _workloads_with_downloads(
            {
                "huggingface": [
                    {"repo_id": REPO_ID, "revision": "abc123", "assets": ["tokenizer"]},
                    {"repo_id": REPO_ID, "revision": "abc123", "assets": ["config"]},
                ]
            }
        )

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.cache), 1)
        self.assertEqual(plan.cache[0].assets, frozenset({"tokenizer", "config"}))
        self.assertEqual(plan.cache[0].revision, "abc123")

    def test_mixed_optional_fields_for_same_repo_do_not_break_sorting(self):
        # Two entries sharing a repo_id where revision/include are set on one but not the
        # other previously crashed the sort with a None-vs-value TypeError.
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "cache": [{"repo_id": REPO_ID, "assets": ["tokenizer"]}],
                        "repos": [{"repo_id": REPO_ID, "repo_type": "model", "name": "snap"}],
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "cache": [{"repo_id": REPO_ID, "revision": "v1", "assets": ["config"]}],
                        "repos": [{"repo_id": REPO_ID, "repo_type": "model", "name": "snap", "include": ["*.json"]}],
                    }
                }
            },
        }

        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        self.assertEqual({(s.revision or "") for s in plan.cache}, {"", "v1"})
        self.assertEqual({(s.include or ()) for s in plan.repos}, {(), ("*.json",)})

    def test_repo_exact_dedup(self):
        item = {"repo_id": REPO_ID, "repo_type": "model", "target": "shared", "name": "shared-copy"}
        workloads = _workloads_with_downloads({"huggingface": {"repos": [item, dict(item)]}})

        plan = hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

        self.assertEqual(len(plan.repos), 1)

    def test_cache_entry_rejects_repo_download_keys(self):
        workloads = _workloads_with_downloads(
            {"huggingface": [{"repo_id": REPO_ID, "snapshot": {"name": "model-files"}}]}
        )

        with self.assertRaisesRegex(ValueError, "downloads.huggingface.repos"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_huggingface_map_rejects_unknown_keys(self):
        workloads = _workloads_with_downloads({"huggingface": {"snapshots": [{"repo_id": REPO_ID}]}})

        with self.assertRaisesRegex(ValueError, "cache.*repos"):
            hf.build_hf_download_plan(workloads, [WORKLOAD_KEY])

    def test_same_repo_destination_with_different_source_fails(self):
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "repos": [
                            {"repo_id": "org/source-a", "repo_type": "model", "target": "shared", "name": "shared"}
                        ]
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "repos": [
                            {"repo_id": "org/source-b", "repo_type": "model", "target": "shared", "name": "shared"}
                        ]
                    }
                }
            },
        }
        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        with tempfile.TemporaryDirectory() as install_path:
            with self.assertRaises(ValueError):
                hf._dedup_and_validate_repo_destinations(plan.repos, install_path)

    def test_same_repo_destination_with_same_source_collapses(self):
        workloads = {
            "workload1": {
                "downloads": {
                    "huggingface": {
                        "repos": [{"repo_id": "org/source", "repo_type": "model", "target": "shared", "name": "shared"}]
                    }
                }
            },
            "workload2": {
                "downloads": {
                    "huggingface": {
                        "repos": [{"repo_id": "org/source", "repo_type": "model", "target": "shared", "name": "shared"}]
                    }
                }
            },
        }
        plan = hf.build_hf_download_plan(workloads, ["workload1", "workload2"])

        with tempfile.TemporaryDirectory() as install_path:
            deduped = hf._dedup_and_validate_repo_destinations(plan.repos, install_path)

        self.assertEqual(len(deduped), 1)

    def test_shared_target_resolves_to_datasets_dir(self):
        spec = hf.HuggingFaceRepoSpec(
            workload_key=WORKLOAD_KEY, repo_id="org/data", repo_type="dataset", target="shared", name="data-copy"
        )
        with tempfile.TemporaryDirectory() as install_path:
            local_dir = hf.resolve_huggingface_repo_local_dir(spec, install_path)

            self.assertEqual(local_dir, str((Path(install_path) / "datasets" / "data-copy").resolve()))

    def test_repo_destination_must_be_subdirectory(self):
        for name in (".", "..", "../escape"):
            spec = hf.HuggingFaceRepoSpec(
                workload_key=WORKLOAD_KEY, repo_id="org/data", repo_type="model", target="shared", name=name
            )
            with tempfile.TemporaryDirectory() as install_path:
                with self.assertRaises(ValueError):
                    hf.resolve_huggingface_repo_local_dir(spec, install_path)


class TestHuggingFaceDownloadPhase(unittest.TestCase):
    """Tests for HuggingFace download orchestration."""

    def test_cache_download_uses_expected_patterns(self):
        specs = _asset_specs(REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("huggingface_hub.snapshot_download") as mock_snapshot,
                patch("huggingface_hub.login") as mock_login,
                patch.dict(os.environ, {"LLMB_HF_MAX_WORKERS": "4"}),
            ):
                mock_snapshot.return_value = cache_dir
                # Avoid network access: the download phase may try to write tokenizer
                # metadata if it's not present in the snapshot output.
                Path(os.path.join(cache_dir, "tokenizer_config.json")).write_text("{}")

                hf.download_huggingface_cache_assets(specs, cache_dir)
                _assert_hf_env(self, cache_dir)

        mock_login.assert_not_called()
        mock_snapshot.assert_called_once()
        snapshot_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(snapshot_kwargs["repo_id"], REPO_ID)
        self.assertIn("allow_patterns", snapshot_kwargs)
        self.assertIn("ignore_patterns", snapshot_kwargs)
        self.assertEqual(snapshot_kwargs["cache_dir"], os.path.join(cache_dir, "hub"))
        self.assertEqual(snapshot_kwargs["max_workers"], 4)
        self.assertIsInstance(snapshot_kwargs["allow_patterns"], list)
        self.assertIsInstance(snapshot_kwargs["ignore_patterns"], list)
        self.assertTrue(snapshot_kwargs["allow_patterns"])
        self.assertTrue(snapshot_kwargs["ignore_patterns"])
        self.assertTrue(all(isinstance(item, str) for item in snapshot_kwargs["allow_patterns"]))
        self.assertTrue(all(isinstance(item, str) for item in snapshot_kwargs["ignore_patterns"]))
        self.assertIn("*.jinja", snapshot_kwargs["allow_patterns"])
        self.assertIn("chat_template*", snapshot_kwargs["allow_patterns"])
        self.assertIn("chat_templates/*", snapshot_kwargs["allow_patterns"])

    def test_cache_download_scopes_chat_template_patterns_to_tokenizers(self):
        specs = _asset_specs(REPO_ID, "config")
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("huggingface_hub.snapshot_download") as mock_snapshot:
                mock_snapshot.return_value = cache_dir

                hf.download_huggingface_cache_assets(specs, cache_dir)

        allow_patterns = mock_snapshot.call_args.kwargs["allow_patterns"]
        self.assertNotIn("*.jinja", allow_patterns)
        self.assertNotIn("chat_template*", allow_patterns)
        self.assertNotIn("chat_templates/*", allow_patterns)

    def test_cache_download_authenticates_with_token(self):
        """Token is passed explicitly to snapshot_download (no login() call)."""
        specs = _asset_specs(REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with patch("huggingface_hub.snapshot_download") as mock_snapshot:
                mock_snapshot.return_value = cache_dir
                # Avoid network access (see note in other download-phase test).
                Path(os.path.join(cache_dir, "tokenizer_config.json")).write_text("{}")

                hf.download_huggingface_cache_assets(specs, cache_dir, token="hf_test_token")

        # Token should be passed explicitly to snapshot_download
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(call_kwargs.get("token"), "hf_test_token")

    def test_repo_download_uses_configured_paths_and_cache(self):
        spec = hf.HuggingFaceRepoSpec(
            workload_key=WORKLOAD_KEY,
            repo_id="org/dataset-repo",
            repo_type="dataset",
            revision="main",
            target="shared",
            name="dataset-copy",
            include=("*.jsonl", "*.txt"),
            exclude=("scratch/*",),
        )
        with tempfile.TemporaryDirectory() as install_path:
            cache_dir = os.path.join(install_path, ".cache", "huggingface")
            with (
                patch("huggingface_hub.snapshot_download") as mock_snapshot,
                patch.dict(os.environ, {"LLMB_HF_MAX_WORKERS": "4"}),
            ):
                mock_snapshot.return_value = os.path.join(install_path, "datasets", "dataset-copy")

                hf.download_huggingface_repos((spec,), install_path, cache_dir, token="hf_test_token")

        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args.kwargs
        self.assertEqual(call_kwargs["repo_id"], "org/dataset-repo")
        self.assertEqual(call_kwargs["repo_type"], "dataset")
        self.assertEqual(call_kwargs["revision"], "main")
        self.assertEqual(call_kwargs["cache_dir"], os.path.join(cache_dir, "hub"))
        self.assertEqual(call_kwargs["token"], "hf_test_token")
        self.assertEqual(call_kwargs["max_workers"], 4)
        expected_local_dir = str((Path(install_path) / "datasets" / "dataset-copy").resolve())
        self.assertEqual(call_kwargs["local_dir"], expected_local_dir)
        self.assertEqual(call_kwargs["allow_patterns"], ["*.jsonl", "*.txt"])
        self.assertEqual(call_kwargs["ignore_patterns"], ["scratch/*"])


class TestFinalizeTokenizerSnapshot(unittest.TestCase):
    def test_skips_when_tokenizer_config_present(self):
        with tempfile.TemporaryDirectory() as snapshot_dir:
            Path(snapshot_dir, "tokenizer_config.json").write_text("{}")
            with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
                hf._maybe_finalize_tokenizer_snapshot(REPO_ID, {"tokenizer"}, snapshot_dir, None)

        mock_from_pretrained.assert_not_called()

    def test_generates_metadata_when_tokenizer_config_missing(self):
        with tempfile.TemporaryDirectory() as snapshot_dir:
            mock_tokenizer = MagicMock()
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
                hf._maybe_finalize_tokenizer_snapshot(REPO_ID, {"tokenizer"}, snapshot_dir, None)

        mock_tokenizer.save_pretrained.assert_called_once_with(snapshot_dir)


class TestHuggingFaceVerifyPhase(unittest.TestCase):
    """Tests for HuggingFace offline verification."""

    def test_verify_calls_expected_assets_with_offline_flags(self):
        specs = _asset_specs(REPO_ID, "tokenizer", "config")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("transformers.AutoTokenizer") as mock_auto_tokenizer,
                patch("transformers.AutoConfig") as mock_auto_config,
            ):
                hf.verify_huggingface_cache_assets(specs, cache_dir)

        _assert_offline_call(self, mock_auto_tokenizer, REPO_ID)
        _assert_offline_call(self, mock_auto_config, REPO_ID)
        _assert_hf_env(self, cache_dir)

    def test_nemotron_tokenizer_only_skips_config_verification(self):
        specs = _asset_specs(NEMOTRON_REPO_ID, "tokenizer")
        with tempfile.TemporaryDirectory() as cache_dir:
            with (
                patch("transformers.AutoTokenizer") as mock_auto_tokenizer,
                patch("transformers.AutoConfig") as mock_auto_config,
            ):
                hf.verify_huggingface_cache_assets(specs, cache_dir)

        mock_auto_tokenizer.from_pretrained.assert_called_once()
        mock_auto_config.from_pretrained.assert_not_called()


class TestXetConcurrencyDefault(unittest.TestCase):
    """set_hf_environment pins hf_xet per-file download concurrency by default."""

    _VAR = "HF_XET_FIXED_DOWNLOAD_CONCURRENCY"

    def setUp(self):
        self._saved = os.environ.pop(self._VAR, None)

    def tearDown(self):
        os.environ.pop(self._VAR, None)
        if self._saved is not None:
            os.environ[self._VAR] = self._saved

    def test_pins_concurrency_to_one_by_default(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            hf.set_hf_environment(cache_dir)
        self.assertEqual(os.environ.get(self._VAR), "1")

    def test_does_not_override_explicit_user_value(self):
        os.environ[self._VAR] = "4"
        with tempfile.TemporaryDirectory() as cache_dir:
            hf.set_hf_environment(cache_dir)
        self.assertEqual(os.environ.get(self._VAR), "4")


class TestHfMaxWorkers(unittest.TestCase):
    """_hf_max_workers default and LLMB_HF_MAX_WORKERS override handling."""

    _VAR = "LLMB_HF_MAX_WORKERS"

    def setUp(self):
        self._saved = os.environ.pop(self._VAR, None)

    def tearDown(self):
        os.environ.pop(self._VAR, None)
        if self._saved is not None:
            os.environ[self._VAR] = self._saved

    def test_default_is_four(self):
        self.assertEqual(hf._hf_max_workers(), hf._DEFAULT_HF_MAX_WORKERS)
        self.assertEqual(hf._hf_max_workers(), 4)

    def test_valid_override_is_used(self):
        os.environ[self._VAR] = "8"
        self.assertEqual(hf._hf_max_workers(), 8)

    def test_non_integer_override_falls_back_to_default(self):
        os.environ[self._VAR] = "not-an-int"
        self.assertEqual(hf._hf_max_workers(), 4)

    def test_non_positive_override_falls_back_to_default(self):
        os.environ[self._VAR] = "0"
        self.assertEqual(hf._hf_max_workers(), 4)


if __name__ == "__main__":
    unittest.main()
