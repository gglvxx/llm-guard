"""Application entrypoint and FastAPI factory.

Run with ``python -m app.main`` or, for auto-reload during development::

    uvicorn app.main:create_app --factory --reload
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.api.dependencies import build_container
from app.api.middleware import (
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
    error_response,
    get_request_id,
)
from app.api.routes import router
from app.config import PROJECT_ROOT, Settings, load_settings
from app.core.ml_classifier import IntentClassifier
from app.logging_config import configure_logging
from app.providers.base import BaseLLMProvider

logger = logging.getLogger("llm_guard.app")

STATIC_DIR: Path = PROJECT_ROOT / "static"

_STATUS_CODES = {
    400: "bad_request",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    429: "rate_limited",
    503: "provider_unavailable",
}


def create_app(
    settings: Settings | None = None,
    *,
    provider: BaseLLMProvider | None = None,
    classifier: IntentClassifier | None = None,
) -> FastAPI:
    """Build the ASGI application. Every argument is optional and injectable for tests."""
    settings = settings or load_settings()
    configure_logging(settings.app.log_level)
    container = build_container(settings, provider=provider, classifier=classifier)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "llm-guard started",
            extra={
                "version": __version__,
                "provider": container.provider.name,
                "demo_mode": settings.app.demo_mode,
                "rules": container.firewall.rules_loaded,
            },
        )
        yield
        await container.provider.aclose()

    demo = settings.app.demo_mode
    app = FastAPI(
        title="LLM-Guard",
        version=__version__,
        description="Multi-layer AI security firewall and telemetry engine.",
        lifespan=lifespan,
        # Interactive API docs are a demo convenience; they are not exposed otherwise.
        docs_url="/docs" if demo else None,
        redoc_url=None,
        openapi_url="/openapi.json" if demo else None,
    )
    app.state.container = container

    # Starlette runs the middleware added last first: RequestContext is outermost, so
    # even a 429 produced by the rate limiter carries a request ID and security headers.
    app.add_middleware(RateLimitMiddleware, settings=settings.rate_limit)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

    _register_exception_handlers(app)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Map every failure onto the uniform error envelope; never leak internals."""

    # Registered on Starlette's base class so router-level errors (404, 405) are covered
    # as well as the HTTPException instances raised by our own routes.
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "error")
        return error_response(exc.status_code, code, str(exc.detail), get_request_id(request))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report where and why validation failed, without echoing the submitted values.
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return error_response(422, "validation_error", problems, get_request_id(request))

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled exception", exc_info=exc)
        return error_response(
            500, "internal_error", "Internal server error.", get_request_id(request)
        )


def main() -> None:
    """Console entrypoint: ``python -m app.main [--host HOST] [--port PORT]``."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the LLM-Guard server.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()
    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )


if __name__ == "__main__":
    main()
