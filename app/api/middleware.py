"""HTTP middleware: request correlation, rate limiting and security headers."""

from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.config import RateLimitSettings
from app.logging_config import request_id_var
from app.models.schemas import ErrorDetail, ErrorResponse

access_logger = logging.getLogger("llm_guard.access")

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")

#: The dashboard needs nothing but same-origin assets, so the policy can be strict.
_STRICT_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
#: Swagger UI (demo mode only) loads its bundle from jsDelivr and uses an inline bootstrap.
_DOCS_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "img-src 'self' data: https://fastapi.tiangolo.com; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)
_DOCS_PATHS = ("/docs", "/redoc")


def error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None,
    *,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the uniform JSON error envelope used by every non-2xx API response."""
    body = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(body.model_dump(), status_code=status_code, headers=headers)


def get_request_id(request: Request) -> str | None:
    request_id: str | None = getattr(request.state, "request_id", None)
    return request_id


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request ID, exposes it to logs and responses, and writes the access log.

    A client-supplied ``X-Request-ID`` is honoured only if it is a short token of
    safe characters: the value ends up in logs and response headers, so anything
    else is replaced rather than trusted.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _SAFE_REQUEST_ID.fullmatch(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            if not request.url.path.startswith("/static"):
                access_logger.info(
                    "http_request",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
                        "client": request.client.host if request.client else None,
                    },
                )
            request_id_var.reset(token)


@dataclass(frozen=True, slots=True)
class RateLimitResult:
    allowed: bool
    remaining: int
    retry_after: float


class SlidingWindowRateLimiter:
    """Exact sliding-window limiter keyed by client, with bounded memory."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10_000,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    def check(self, key: str) -> RateLimitResult:
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                self._evict_if_full(now)
                hits = self._hits[key] = deque()
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return RateLimitResult(False, 0, self._window - (now - hits[0]))
            hits.append(now)
            return RateLimitResult(True, self._limit - len(hits), 0.0)

    def _evict_if_full(self, now: float) -> None:
        """Keep the key table bounded so spoofed client addresses cannot exhaust memory."""
        if len(self._hits) < self._max_keys:
            return
        stale = [
            key for key, hits in self._hits.items() if not hits or now - hits[-1] >= self._window
        ]
        for key in stale:
            del self._hits[key]
        while len(self._hits) >= self._max_keys:
            del self._hits[next(iter(self._hits))]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client rate limit on ``/api/*`` (mitigates brute-force probing and resource abuse).

    The key is the socket peer address. ``X-Forwarded-For`` is deliberately not
    trusted: behind a reverse proxy, configure uvicorn's ``--proxy-headers`` with
    an explicit ``--forwarded-allow-ips`` so the peer address is the real client.
    """

    def __init__(self, app: ASGIApp, *, settings: RateLimitSettings) -> None:
        super().__init__(app)
        self._enabled = settings.enabled
        self._limiter = SlidingWindowRateLimiter(settings.requests, settings.window_seconds)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self._enabled or not request.url.path.startswith("/api/"):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        result = self._limiter.check(client)
        if not result.allowed:
            retry_after = max(1, math.ceil(result.retry_after))
            return error_response(
                429,
                "rate_limited",
                f"Too many requests. Retry in {retry_after} s.",
                get_request_id(request),
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self._limiter.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limiter.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds defensive headers to every response (strict CSP, no framing, no sniffing)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        path = request.url.path
        csp = _DOCS_CSP if path.startswith(_DOCS_PATHS) else _STRICT_CSP
        response.headers.setdefault("Content-Security-Policy", csp)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        if path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
