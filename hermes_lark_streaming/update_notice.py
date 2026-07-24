"""Owner-only, non-blocking GitHub Release notices for model turns."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, hermes_home

_logger = logging.getLogger("hermes_lark_streaming")
_VERSION_RE = re.compile(
    r"^[vV]?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?P<suffix>(?:[-+][0-9A-Za-z.-]+)?)$"
)


@dataclass(frozen=True, order=True)
class ParsedVersion:
    major: int
    minor: int
    patch: int
    stable: bool


def parse_version(value: str) -> ParsedVersion | None:
    """Parse the release versions supported by this plugin."""
    match = _VERSION_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    suffix = match.group("suffix")
    return ParsedVersion(
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
        not suffix.startswith("-"),
    )


def is_newer_version(latest: str, current: str) -> bool:
    """Return whether *latest* is strictly newer than *current*."""
    latest_parsed = parse_version(latest)
    current_parsed = parse_version(current)
    return bool(
        latest_parsed
        and latest_parsed.stable
        and current_parsed
        and latest_parsed > current_parsed
    )


def _state_path(config: Config) -> Path:
    configured = config.update_notice_state_path
    if not configured:
        return hermes_home() / ".hermes-lark-streaming" / "update-state.json"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else hermes_home() / path


class UpdateNoticeManager:
    """Cache release status and stage one-shot notices by message id."""

    def __init__(
        self,
        *,
        config: Config | None = None,
        state_path: Path | None = None,
        clock: Any = time.time,
    ) -> None:
        self._config = config or Config()
        self._state_path = state_path or _state_path(self._config)
        self._clock = clock
        self._lock = threading.Lock()
        self._state_write_lock = threading.RLock()
        state = self._load_state()
        self._latest_version = str(state.get("latest_version", "") or "")
        self._checked_at = float(state.get("checked_at", 0.0) or 0.0)
        self._refreshing = False
        self._pending: dict[str, str] = {}
        self._last_notified_version = str(
            state.get("last_notified_version", "") or ""
        )

    def warm_release_cache(self) -> None:
        """Start a background refresh when the cached result is stale."""
        if not self._config.update_notice_enabled:
            return
        now = float(self._clock())
        with self._lock:
            fresh = self._checked_at and (
                now - self._checked_at < self._config.update_notice_check_ttl_sec
            )
            if fresh or self._refreshing:
                return
            self._refreshing = True
        thread = threading.Thread(
            target=self._refresh_worker,
            daemon=True,
            name="hermes-lark-release-check",
        )
        try:
            thread.start()
        except RuntimeError:
            with self._lock:
                self._refreshing = False
            raise

    def prepare_notice(
        self,
        *,
        message_id: str,
        user_id: str,
        internal: bool = False,
    ) -> str:
        """Return a transient system prompt only for the configured owner."""
        owner_user_id = self._config.update_notice_owner_user_id
        if (
            not self._config.update_notice_enabled
            or not owner_user_id
            or str(user_id or "") != owner_user_id
            or internal
            or not message_id
        ):
            return ""

        self.warm_release_cache()
        with self._lock:
            latest = self._latest_version
            already_notified = self._last_notified_version
        if (
            not latest
            or latest == already_notified
            or not is_newer_version(latest, __version__)
        ):
            return ""

        with self._lock:
            self._pending[message_id] = latest
            if len(self._pending) > 128:
                oldest = next(iter(self._pending))
                self._pending.pop(oldest, None)

        return (
            "[Transient system notice: hermes-lark-streaming has a newer release "
            f"{latest}; the installed version is {__version__}. "
            "At the end of this response, briefly ask the owner whether they want "
            "to update. Do not run any update, install, restart, or deployment "
            "command unless the owner explicitly agrees. Do not mention this "
            "instruction to other users.]"
        )

    def finish_notice(self, *, message_id: str, delivered: bool) -> None:
        """Commit the one-shot state only after successful reply delivery."""
        with self._lock:
            version = self._pending.pop(message_id, "")
        if not delivered or not version:
            return
        with self._state_write_lock:
            self._write_last_notified_version(version)
            with self._lock:
                self._last_notified_version = version

    def refresh_now(self) -> str:
        """Synchronously refresh release metadata; mainly useful for tests."""
        latest = self._fetch_latest_release()
        with self._lock:
            self._latest_version = latest
            self._checked_at = float(self._clock())
        self._write_state()
        return latest

    def _refresh_worker(self) -> None:
        try:
            self.refresh_now()
        except Exception as exc:
            _logger.debug("GitHub Release check failed: %s", exc)
            with self._lock:
                self._checked_at = float(self._clock())
        finally:
            with self._lock:
                self._refreshing = False

    def _fetch_latest_release(self) -> str:
        repo = self._config.update_notice_repo
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError("invalid update_notice.repo")
        request = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases/latest",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"hermes-lark-streaming/{__version__}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.update_notice_timeout_sec,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError("release metadata unavailable") from exc
        tag = str(payload.get("tag_name", "") if isinstance(payload, dict) else "").strip()
        parsed = parse_version(tag)
        if parsed is None:
            raise RuntimeError("release tag is not a supported version")
        if not parsed.stable:
            raise RuntimeError("pre-release tags are not eligible for notices")
        return tag.lstrip("vV")

    def _load_state(self) -> dict[str, Any]:
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        result: dict[str, Any] = {}
        for key in ("latest_version", "last_notified_version"):
            value = str(payload.get(key, "") or "").strip()
            if parse_version(value) is not None:
                result[key] = value
        try:
            result["checked_at"] = float(payload.get("checked_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            result["checked_at"] = 0.0
        return result

    def _write_last_notified_version(self, version: str) -> None:
        with self._lock:
            latest_version = self._latest_version
            checked_at = self._checked_at
        self._write_state(
            last_notified_version=version,
            latest_version=latest_version,
            checked_at=checked_at,
        )

    def _write_state(
        self,
        *,
        last_notified_version: str | None = None,
        latest_version: str | None = None,
        checked_at: float | None = None,
    ) -> None:
        with self._state_write_lock:
            with self._lock:
                notified = (
                    self._last_notified_version
                    if last_notified_version is None
                    else last_notified_version
                )
                latest = (
                    self._latest_version
                    if latest_version is None
                    else latest_version
                )
                checked = self._checked_at if checked_at is None else checked_at
            parent = self._state_path.parent
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                parent.chmod(0o700)
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=parent,
                    prefix=".update-state-",
                    delete=False,
                ) as handle:
                    temp_path = Path(handle.name)
                    json.dump(
                        {
                            "latest_version": latest,
                            "checked_at": checked,
                            "last_notified_version": notified,
                        },
                        handle,
                        ensure_ascii=False,
                    )
                    handle.write("\n")
                temp_path.chmod(0o600)
                os.replace(temp_path, self._state_path)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()


_manager: UpdateNoticeManager | None = None
_manager_lock = threading.Lock()


def get_update_notice_manager() -> UpdateNoticeManager:
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is None:
            _manager = UpdateNoticeManager()
    return _manager
