"""Single import point for the HTTP client.

``httpx2`` is the maintained continuation of ``httpx`` (same API, now under the
Pydantic organisation) and is what current Starlette/FastAPI test clients use.
The fallback keeps the project working on environments that only have the
original ``httpx`` installed.
"""

from __future__ import annotations

try:
    import httpx2 as httpx
except ImportError:  # pragma: no cover - exercised only where httpx2 is absent
    import httpx  # type: ignore[import-not-found,no-redef]

__all__ = ["httpx"]
