"""Structured logging: valid single-line JSON, correlation IDs, no log injection."""

from __future__ import annotations

import json
import logging
import sys

from app.logging_config import JsonFormatter, configure_logging, request_id_var


def render(message: str, *, level: int = logging.INFO, **extra: object) -> dict[str, object]:
    record = logging.LogRecord("llm_guard.test", level, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    line = JsonFormatter().format(record)
    assert "\n" not in line
    parsed: dict[str, object] = json.loads(line)
    return parsed


def test_record_is_rendered_as_json_with_standard_fields() -> None:
    payload = render("security_event", level=logging.WARNING)
    assert payload["msg"] == "security_event"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "llm_guard.test"
    assert str(payload["ts"]).endswith("+00:00")
    assert "request_id" not in payload


def test_extra_fields_are_included() -> None:
    payload = render("security_event", decision="BLOCK", risk_score=97, rules=["PI-001"])
    assert payload["decision"] == "BLOCK"
    assert payload["risk_score"] == 97
    assert payload["rules"] == ["PI-001"]


def test_request_id_comes_from_the_context() -> None:
    token = request_id_var.set("req-123")
    try:
        assert render("hello")["request_id"] == "req-123"
    finally:
        request_id_var.reset(token)


def test_newlines_cannot_forge_a_second_log_entry() -> None:
    forged = 'first line\n{"level": "INFO", "msg": "forged entry"}'
    payload = render("security_event", prompt_preview=forged)
    # Still exactly one JSON object (asserted by render); the payload is data, not structure.
    assert payload["prompt_preview"] == forged


def test_exceptions_are_serialised() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "llm_guard.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )
    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exc"]


def test_non_serialisable_extras_do_not_break_logging() -> None:
    assert "object at" in str(render("hello", thing=object())["thing"])


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    before = list(root.handlers)
    configure_logging("WARNING")
    configure_logging("WARNING")
    added = [handler for handler in root.handlers if handler not in before]
    try:
        assert len(added) == 1
        assert isinstance(added[0].formatter, JsonFormatter)
        assert root.level == logging.WARNING
    finally:
        for handler in added:
            root.removeHandler(handler)
