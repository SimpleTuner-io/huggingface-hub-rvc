"""Compatibility shim for RVC helpers.

SimpleTuner does not capture CUDA graphs for startup voice transforms yet.
"""

from __future__ import annotations

from typing import Any, Callable


def run_cuda_graph(model: Any, key: str, forward: Callable[..., Any], *args: Any) -> Any:
    _ = model, key
    return forward(*args)
