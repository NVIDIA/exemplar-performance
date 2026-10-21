#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge-case tests for dependency canonicalization and venv grouping.

Covers:
- Mixed pip string and dict dependencies
- Git dependencies with different install methods
- Dependencies with missing url/commit fields
- Unifying identical sources across pip-git and git
- Editable installs force isolated venvs
"""

from llmb_install.core.dependency import group_workloads_by_dependencies


def _single_group_members(result):
    """Return the members of the sole non-scripted group, if exactly one exists."""
    non_scripted = {k: v for k, v in result.items() if k is not None}
    if len(non_scripted) != 1:
        return None
    return next(iter(non_scripted.values()))


def test_grouping_mixed_pip_string_and_dict_same_group():
    """Workloads with same pip strings and pip-git dict resolve to one shared group."""
    deps = {
        "pip": [
            "numpy==1.24.0",
            {"package": "nemo", "url": "https://github.com/NVIDIA/NeMo.git", "commit": "deadbeef"},
        ]
    }
    workloads = {
        "w1": {"setup": {"dependencies": deps}},
        "w2": {"setup": {"dependencies": deps}},
    }
    result = group_workloads_by_dependencies(workloads, ["w1", "w2"])
    members = _single_group_members(result)
    assert members is not None
    assert set(members) == {"w1", "w2"}


def test_grouping_git_script_install_method_differences_ignored():
    """Identical git sources with different script install methods should group together."""
    workloads = {
        "ga": {
            "setup": {
                "dependencies": {
                    "git": {
                        "nemo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "cafebabe",
                            "install_method": {"type": "script", "path": "install.sh"},
                        }
                    }
                }
            }
        },
        "gb": {
            "setup": {
                "dependencies": {
                    "git": {
                        "nemo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "cafebabe",
                            "install_method": {"type": "script", "path": "other.sh"},
                        }
                    }
                }
            }
        },
    }
    result = group_workloads_by_dependencies(workloads, ["ga", "gb"])
    members = _single_group_members(result)
    assert members is not None
    assert set(members) == {"ga", "gb"}


def test_clone_only_different_commits_share_venv():
    """Workloads with same pip deps but different clone-only commits should share a venv."""
    workloads = {
        "w1": {
            "setup": {
                "dependencies": {
                    "pip": ["numpy==1.24.0"],
                    "git": {
                        "repo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "aaaa1111",
                            "install_method": {"type": "clone"},
                        }
                    },
                }
            }
        },
        "w2": {
            "setup": {
                "dependencies": {
                    "pip": ["numpy==1.24.0"],
                    "git": {
                        "repo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "bbbb2222",
                            "install_method": {"type": "clone"},
                        }
                    },
                }
            }
        },
    }
    result = group_workloads_by_dependencies(workloads, ["w1", "w2"])
    members = _single_group_members(result)
    assert members is not None
    assert set(members) == {"w1", "w2"}


def test_clone_only_excluded_script_included():
    """Clone-only dep should not affect hash, but script dep should."""
    workloads = {
        "clone_only": {
            "setup": {
                "dependencies": {
                    "pip": ["numpy==1.24.0"],
                    "git": {
                        "repo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "aaaa1111",
                            "install_method": {"type": "clone"},
                        }
                    },
                }
            }
        },
        "with_script": {
            "setup": {
                "dependencies": {
                    "pip": ["numpy==1.24.0"],
                    "git": {
                        "repo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "aaaa1111",
                            "install_method": {"type": "script", "path": "install.sh"},
                        }
                    },
                }
            }
        },
    }
    result = group_workloads_by_dependencies(workloads, ["clone_only", "with_script"])
    # They should NOT share a group — script dep affects the venv, clone does not
    groups = {k: set(v) for k, v in result.items() if k is not None}
    assert len(groups) == 2
    assert any(g == {"clone_only"} for g in groups.values())
    assert any(g == {"with_script"} for g in groups.values())


def test_grouping_unifies_git_and_pip_git_sources():
    """Same source via git and pip-git should group together."""
    workloads = {
        "g": {
            "setup": {
                "dependencies": {
                    "git": {
                        "nemo": {
                            "url": "https://github.com/NVIDIA/NeMo.git",
                            "commit": "aaaa1111",
                            "install_method": {"type": "script", "path": "install.sh"},
                        }
                    }
                }
            }
        },
        "p": {
            "setup": {
                "dependencies": {
                    "pip": [{"package": "nemo", "url": "https://github.com/NVIDIA/NeMo.git", "commit": "aaaa1111"}]
                }
            }
        },
    }
    result = group_workloads_by_dependencies(workloads, ["g", "p"])
    members = _single_group_members(result)
    assert members is not None
    assert set(members) == {"g", "p"}


def test_grouping_handles_missing_url_commit_fields():
    """Missing url/commit in pip dicts should not crash; identical dicts group; differing do not."""
    w_same = {
        "setup": {
            "dependencies": {
                "pip": [
                    {"package": "locallib", "extra": "val"},  # missing url/commit
                ]
            }
        }
    }
    w_diff = {
        "setup": {
            "dependencies": {
                "pip": [
                    {"package": "locallib", "extra": "DIFFERENT"},
                ]
            }
        }
    }
    workloads = {"a": w_same, "b": w_same, "c": w_diff}
    result = group_workloads_by_dependencies(workloads, ["a", "b", "c"])

    # a and b should group together; c should be separate
    groups = {k: set(v) for k, v in result.items() if k is not None}
    assert len(groups) == 2
    assert any(g == {"a", "b"} for g in groups.values())
    assert any(g == {"c"} for g in groups.values())


def test_editable_pip_forces_isolation():
    """Editable pip deps force isolated venvs even if commits match."""
    deps = {
        "pip": [
            {"package": "nemo", "url": "https://github.com/NVIDIA/NeMo.git", "commit": "abcd1234", "editable": True}
        ]
    }
    workloads = {
        "e1": {"setup": {"dependencies": deps}},
        "e2": {"setup": {"dependencies": deps}},
    }
    result = group_workloads_by_dependencies(workloads, ["e1", "e2"])
    # Expect two distinct groups, each with a single workload (non-scripted)
    non_scripted = [set(v) for k, v in result.items() if k is not None]
    assert len(non_scripted) == 2
    assert {"e1"} in non_scripted and {"e2"} in non_scripted
