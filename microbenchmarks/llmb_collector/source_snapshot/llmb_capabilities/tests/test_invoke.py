# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for :func:`llmb_capabilities.invoke_sync` / :func:`invoke_async`.

No ``pytest-asyncio`` dependency: async cases drive their coroutine
directly via :func:`asyncio.run` from an otherwise-plain sync test, which
keeps this stdlib-only package's dev dependency set unchanged.
"""

from __future__ import annotations

import asyncio
import threading

from llmb_capabilities import Capability, invoke_async, invoke_sync


def _sync_invoke(cli_args: str, /, *, extra: str = "") -> str:
    return f"sync:{cli_args}:{extra}:{threading.current_thread().name}"


async def _async_invoke(cli_args: str, /, *, extra: str = "") -> str:
    return f"async:{cli_args}:{extra}"


class _AsyncCallableInvoke:
    """A callable *instance* (not a function) with an async def __call__.

    inspect.iscoroutinefunction(instance) is False even though calling
    the instance returns a coroutine - this is the case invoke_sync /
    invoke_async must still detect correctly.
    """

    async def __call__(self, cli_args: str, /, *, extra: str = "") -> str:
        return f"async-callable:{cli_args}:{extra}"


def _capability(invoke) -> Capability:
    return Capability(name="x", version=1, invoke=invoke, metadata={"provider": "test"})


class TestInvokeSync:
    def test_calls_a_plain_sync_invoke_directly(self):
        cap = _capability(_sync_invoke)

        result = invoke_sync(cap, "results", extra="e")

        assert result.startswith("sync:results:e:")

    def test_drives_a_coroutine_invoke_to_completion(self):
        cap = _capability(_async_invoke)

        result = invoke_sync(cap, "results", extra="e")

        assert result == "async:results:e"

    def test_drives_an_async_callable_instance_invoke_to_completion(self):
        cap = _capability(_AsyncCallableInvoke())

        result = invoke_sync(cap, "results", extra="e")

        assert result == "async-callable:results:e"


class TestInvokeAsync:
    def test_awaits_a_coroutine_invoke_directly(self):
        cap = _capability(_async_invoke)

        result = asyncio.run(invoke_async(cap, "results", extra="e"))

        assert result == "async:results:e"

    def test_awaits_an_async_callable_instance_invoke_directly(self):
        cap = _capability(_AsyncCallableInvoke())

        result = asyncio.run(invoke_async(cap, "results", extra="e"))

        assert result == "async-callable:results:e"

    def test_offloads_a_plain_sync_invoke_to_a_worker_thread(self):
        cap = _capability(_sync_invoke)
        caller_thread = threading.current_thread().name

        result = asyncio.run(invoke_async(cap, "results", extra="e"))

        assert result.startswith("sync:results:e:")
        invoked_thread = result.rsplit(":", 1)[-1]
        assert invoked_thread != caller_thread

    def test_does_not_block_the_event_loop(self):
        cap = _capability(_sync_invoke)
        marker = []

        async def tick():
            await asyncio.sleep(0)
            marker.append("ticked")

        async def scenario():
            return await asyncio.gather(invoke_async(cap, "results"), tick())

        results = asyncio.run(scenario())

        assert marker == ["ticked"]
        assert results[0].startswith("sync:results::")
