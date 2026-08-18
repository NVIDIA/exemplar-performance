# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic sync/async bridges for calling :attr:`Capability.invoke`.

``Capability.invoke`` is typed as a plain ``Callable[..., Any]`` - a
producer may implement it as an ordinary function or as an ``async
def``. Calling either shape correctly depends on whether the caller
itself is inside a running event loop, which these two helpers handle
so consumers don't have to special-case every producer.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from llmb_capabilities._capability import CapabilityLike


def _invokes_asynchronously(invoke: Any) -> bool:
    """True if calling ``invoke`` returns an awaitable rather than blocking.

    ``inspect.iscoroutinefunction(invoke)`` alone misses a callable
    *instance* whose class defines ``async def __call__`` — ``invoke``
    there is an object, not a function, so the direct check is
    ``False`` even though calling it returns a coroutine. Falling back
    to the type's ``__call__`` catches that case too, without actually
    calling ``invoke`` (which could block) just to find out.
    """
    if inspect.iscoroutinefunction(invoke):
        return True
    # noqa justification: this isn't an is-callable check (that's already
    # true for any `invoke`) - the `__call__` object itself is needed to
    # test whether *it* is a coroutine function.
    dunder_call = getattr(type(invoke), "__call__", None)  # noqa: B004
    return dunder_call is not None and inspect.iscoroutinefunction(dunder_call)


def invoke_sync(capability: CapabilityLike, *args: Any, **kwargs: Any) -> Any:
    """Call ``capability.invoke(*args, **kwargs)`` from synchronous code.

    If ``invoke`` invokes asynchronously (an ``async def`` function/method,
    or a callable instance with an ``async def __call__``), drives it to
    completion with :func:`asyncio.run`. Otherwise, calls it directly.

    :raises RuntimeError: if called while an event loop is already
        running (from :func:`asyncio.run`) - use :func:`invoke_async`
        instead in that context.
    """
    if _invokes_asynchronously(capability.invoke):
        return asyncio.run(capability.invoke(*args, **kwargs))
    return capability.invoke(*args, **kwargs)


async def invoke_async(capability: CapabilityLike, *args: Any, **kwargs: Any) -> Any:
    """Call ``capability.invoke(*args, **kwargs)`` from a running event loop.

    If ``invoke`` invokes asynchronously (see :func:`invoke_sync`), awaits
    it directly. Otherwise, offloads the call to a worker thread via
    :func:`asyncio.to_thread` so a blocking producer doesn't stall the
    loop.
    """
    if _invokes_asynchronously(capability.invoke):
        return await capability.invoke(*args, **kwargs)
    return await asyncio.to_thread(capability.invoke, *args, **kwargs)


__all__ = ["invoke_async", "invoke_sync"]
