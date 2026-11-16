"""Structured JSON logging with per-request correlation IDs.

One JSON object per line: trivially ingestible by a SIEM, and immune to log
injection because user-controlled text is always JSON-escaped (a newline in a
prompt can never forge a second log entry).
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

#: Set by the request-context middleware; read by the formatter for every record.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", logging.INFO, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON, including any ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id is not None:
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class _JsonStreamHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Marker subclass so repeated configuration replaces only our own handler."""


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler on the root logger (idempotent)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, _JsonStreamHandler):
            root.removeHandler(handler)

    handler = _JsonStreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)

    # Route uvicorn's own loggers through the same formatter; the request-context
    # middleware emits a richer access log, so uvicorn's is switched off.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.access").disabled = True
    # The HTTP client logs every request at INFO; only its warnings are interesting here.
    for name in ("httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(name).setLevel(logging.WARNING)
