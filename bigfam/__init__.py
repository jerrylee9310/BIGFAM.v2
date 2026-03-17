"""Redirects ``import bigfam`` to the ``src.bigfam`` package in this workspace."""

from __future__ import annotations

import importlib
from typing import Any

_real = importlib.import_module("src.bigfam")


def __getattr__(name: str) -> Any:
    return getattr(_real, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_real)))


__all__ = getattr(_real, "__all__", [name for name in dir(_real) if not name.startswith("_")])

__path__ = _real.__path__  # type: ignore[assignment]
