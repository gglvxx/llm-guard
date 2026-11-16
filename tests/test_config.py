"""Configuration loading and validation: a bad config must fail loudly at start-up."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    ConfigError,
    CustomRule,
    DLPSettings,
    OllamaSettings,
    PolicySettings,
    Settings,
    Thresholds,
    load_settings,
)
from app.models.schemas import Severity


def test_shipped_config_matches_builtin_defaults() -> None:
    # The YAML file is documentation of the defaults; the two must never drift apart.
    assert load_settings(DEFAULT_CONFIG_PATH) == Settings()


def test_explicit_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "nope.yaml")


def test_env_var_selects_the_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "custom.yaml"
    config.write_text("policy:\n  thresholds:\n    flag: 30\n    block: 60\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))
    loaded = load_settings()
    assert (loaded.policy.thresholds.flag, loaded.policy.thresholds.block) == (30, 60)


@pytest.mark.parametrize(
    "content",
    [
        "app:\n  unknown_option: true\n",  # typo in a key must not be silently ignored
        "policy:\n  thresholds:\n    flag: 80\n    block: 70\n",  # inverted thresholds
        "policy:\n  weights:\n    heuristic: 1.5\n",  # weight outside 0-1
        "app:\n  fail_mode: maybe\n",
        "- just\n- a list\n",
        "app: [unclosed\n",
    ],
)
def test_invalid_config_is_rejected(tmp_path: Path, content: str) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config)


def test_empty_config_file_yields_defaults(tmp_path: Path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("# nothing here\n", encoding="utf-8")
    assert load_settings(config) == Settings()


def test_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Thresholds(flag=70, block=70)


def test_ollama_must_be_local_unless_explicitly_allowed() -> None:
    with pytest.raises(ValidationError, match="localhost"):
        OllamaSettings(base_url="http://llm.internal.example:11434")
    remote = OllamaSettings(base_url="http://llm.internal.example:11434", allow_remote=True)
    assert remote.allow_remote
    with pytest.raises(ValidationError):
        OllamaSettings(base_url="file:///etc/passwd")


def test_custom_rule_pattern_must_compile() -> None:
    with pytest.raises(ValidationError, match="invalid regular expression"):
        CustomRule(
            id="ORG-001", category="x", severity=Severity.HIGH, pattern="(unclosed", description="d"
        )


def test_block_messages_accept_only_the_request_id_placeholder() -> None:
    with pytest.raises(ValidationError):
        PolicySettings(block_message="blocked {user}")
    with pytest.raises(ValidationError):
        DLPSettings(output_block_message="withheld {0}")
    assert "abc" in PolicySettings().block_message.format(request_id="abc")


def test_settings_are_immutable() -> None:
    with pytest.raises(ValidationError):
        Settings().app.demo_mode = False  # type: ignore[misc]
