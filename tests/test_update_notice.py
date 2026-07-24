"""Owner-only release notice tests."""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

from hermes_lark_streaming import __version__
from hermes_lark_streaming.config import Config
from hermes_lark_streaming.update_notice import (
    UpdateNoticeManager,
    is_newer_version,
    parse_version,
)


def _config(**overrides: object) -> Config:
    cfg = Config()
    cfg._raw = {
        "streaming": {
            "enabled": True,
            "update_notice": {
                "enabled": True,
                "owner_user_id": "ou_owner",
                "repo": "Monoxide-Chen/hermes-lark-streaming",
                **overrides,
            },
        }
    }
    return cfg


def test_version_comparison_is_exact() -> None:
    assert is_newer_version("v0.12.0", "0.11.2") is True
    assert is_newer_version("0.11.2", "0.11.2") is False
    assert is_newer_version("0.11.1", "0.11.2") is False
    assert is_newer_version("0.12.0-rc.1", "0.12.0") is False
    assert is_newer_version("0.12.0-rc.1", "0.11.2") is False
    assert parse_version("not-a-version") is None


def test_only_owner_gets_transient_context(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )
    manager._latest_version = "99.0.0"
    manager._checked_at = 1e20

    assert manager.prepare_notice(message_id="other", user_id="ou_other") == ""
    prompt = manager.prepare_notice(message_id="owner", user_id="ou_owner")

    assert "99.0.0" in prompt
    assert __version__ in prompt
    assert "explicitly agrees" in prompt
    assert (
        manager.prepare_notice(
            message_id="internal",
            user_id="ou_owner",
            internal=True,
        )
        == ""
    )


def test_missing_owner_is_fail_closed(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(owner_user_id=""),
        state_path=tmp_path / "state.json",
    )
    manager._latest_version = "99.0.0"
    manager._checked_at = 1e20

    assert manager.prepare_notice(message_id="msg", user_id="ou_any") == ""


def test_failed_delivery_does_not_consume_notice(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    manager = UpdateNoticeManager(config=_config(), state_path=state_path)
    manager._latest_version = "99.0.0"
    manager._checked_at = 1e20

    assert manager.prepare_notice(message_id="first", user_id="ou_owner")
    manager.finish_notice(message_id="first", delivered=False)
    assert not state_path.exists()

    assert manager.prepare_notice(message_id="second", user_id="ou_owner")


def test_successful_delivery_persists_once(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    manager = UpdateNoticeManager(config=_config(), state_path=state_path)
    manager._latest_version = "99.0.0"
    manager._checked_at = 1e20

    assert manager.prepare_notice(message_id="first", user_id="ou_owner")
    manager.finish_notice(message_id="first", delivered=True)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["last_notified_version"] == "99.0.0"
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert manager.prepare_notice(message_id="second", user_id="ou_owner") == ""


def test_background_refresh_obeys_ttl(tmp_path: Path) -> None:
    clock = MagicMock(return_value=1_000.0)
    manager = UpdateNoticeManager(
        config=_config(check_ttl_sec=3600),
        state_path=tmp_path / "state.json",
        clock=clock,
    )
    manager._checked_at = 900.0

    with patch("threading.Thread") as thread_cls:
        manager.warm_release_cache()

    thread_cls.assert_not_called()


def test_fetches_latest_release_tag(tmp_path: Path) -> None:
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"tag_name":"v0.12.0"}'
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )

    with patch(
        "hermes_lark_streaming.update_notice.urllib.request.urlopen",
        return_value=response,
    ) as urlopen:
        assert manager.refresh_now() == "0.12.0"

    request = urlopen.call_args.args[0]
    assert request.full_url.endswith("/releases/latest")
    assert urlopen.call_args.kwargs["timeout"] == 5.0


def test_release_request_failure_keeps_notice_unavailable(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )

    with patch(
        "hermes_lark_streaming.update_notice.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        try:
            manager.refresh_now()
        except RuntimeError:
            pass
        else:
            raise AssertionError("request failure must fail closed")

    assert manager.prepare_notice(message_id="msg", user_id="ou_owner") == ""


def test_no_release_fails_closed(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b"{}"

    with patch(
        "hermes_lark_streaming.update_notice.urllib.request.urlopen",
        return_value=response,
    ):
        try:
            manager.refresh_now()
        except RuntimeError:
            pass
        else:
            raise AssertionError("missing release tag must fail closed")


def test_prerelease_fails_closed(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.read.return_value = b'{"tag_name":"v99.0.0-rc.1"}'

    with patch(
        "hermes_lark_streaming.update_notice.urllib.request.urlopen",
        return_value=response,
    ):
        try:
            manager.refresh_now()
        except RuntimeError:
            pass
        else:
            raise AssertionError("pre-release must fail closed")


def test_equal_version_does_not_prompt(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(),
        state_path=tmp_path / "state.json",
    )
    manager._latest_version = __version__
    manager._checked_at = 1e20

    assert manager.prepare_notice(message_id="msg", user_id="ou_owner") == ""


def test_persisted_release_cache_is_available_on_first_turn(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "latest_version": "99.0.0",
                "checked_at": 1e20,
                "last_notified_version": "",
            }
        ),
        encoding="utf-8",
    )
    manager = UpdateNoticeManager(config=_config(), state_path=state_path)

    assert manager.prepare_notice(message_id="msg", user_id="ou_owner")


def test_invalid_repo_never_builds_request(tmp_path: Path) -> None:
    manager = UpdateNoticeManager(
        config=_config(repo="https://example.invalid/repo"),
        state_path=tmp_path / "state.json",
    )

    with patch("hermes_lark_streaming.update_notice.urllib.request.urlopen") as urlopen:
        try:
            manager.refresh_now()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid repository must fail closed")

    urlopen.assert_not_called()
