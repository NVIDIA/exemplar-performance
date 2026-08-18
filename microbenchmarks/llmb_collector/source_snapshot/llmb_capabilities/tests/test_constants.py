# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanity checks on the registry identifiers.

The constants module is the single source of truth for the entry-point
group name and the four currently-known capability names. These tests
guard against accidental rename / drift / type-erasure when refactoring
the module.
"""

from __future__ import annotations

from llmb_capabilities import (
    BEARER_TOKEN,
    CAPABILITY_GROUP,
    OIDC_ID_TOKEN,
    POST_PROCESSOR_PROCESS,
    STORAGE_UPLOAD,
    SUPPORTED,
    SYSTEM_COLLECT,
    SYSTEM_COMMANDS_CONFIG,
    TLS_SSL_CONTEXT,
)


class TestCapabilityGroup:
    def test_value_is_stable(self):
        assert CAPABILITY_GROUP == "llmb_capabilities"

    def test_is_a_string(self):
        assert isinstance(CAPABILITY_GROUP, str)


class TestCapabilityNames:
    def test_bearer_token_value(self):
        assert BEARER_TOKEN == "oauth.bearer-token"

    def test_oidc_id_token_value(self):
        assert OIDC_ID_TOKEN == "oidc.id-token"

    def test_tls_ssl_context_value(self):
        assert TLS_SSL_CONTEXT == "tls.ssl-context"

    def test_system_collect_value(self):
        assert SYSTEM_COLLECT == "system.collect"

    def test_system_commands_config_value(self):
        assert SYSTEM_COMMANDS_CONFIG == "system.commands-config"

    def test_storage_upload_value(self):
        assert STORAGE_UPLOAD == "storage.upload"

    def test_post_processor_process_value(self):
        assert POST_PROCESSOR_PROCESS == "post-processor.process"

    def test_all_names_unique(self):
        names = [
            BEARER_TOKEN,
            OIDC_ID_TOKEN,
            TLS_SSL_CONTEXT,
            SYSTEM_COLLECT,
            SYSTEM_COMMANDS_CONFIG,
            STORAGE_UPLOAD,
            POST_PROCESSOR_PROCESS,
        ]
        assert len(set(names)) == len(names)


class TestSupported:
    def test_covers_every_known_name(self):
        assert set(SUPPORTED) == {
            BEARER_TOKEN,
            OIDC_ID_TOKEN,
            TLS_SSL_CONTEXT,
            SYSTEM_COLLECT,
            SYSTEM_COMMANDS_CONFIG,
            STORAGE_UPLOAD,
            POST_PROCESSOR_PROCESS,
        }

    def test_versions_are_positive_ints(self):
        # A non-positive ceiling would silently filter out every
        # provider-advertised capability of that name.
        for name, ceiling in SUPPORTED.items():
            assert isinstance(ceiling, int), name
            assert ceiling >= 1, name
