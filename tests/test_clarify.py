"""Inline Feishu clarify card rendering and lifecycle tests."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_lark_streaming.clarify import (
    ClarifyState,
    build_clarify_add_action,
    build_clarify_resolved_actions,
    parse_clarify_submission,
)
from hermes_lark_streaming.controller import StreamCardController
from hermes_lark_streaming.feishu import FeishuClient
from hermes_lark_streaming.streaming.session import CardSession, SessionState


def _setup_controller() -> StreamCardController:
    ctrl = StreamCardController()
    ctrl._cfg._raw = {
        "streaming": {"enabled": True},
        "feishu": {"app_id": "app", "app_secret": "secret"},
    }
    ctrl._initialized = True
    ctrl._client = AsyncMock(spec=FeishuClient)
    ctrl._client.cardkit_batch_update = AsyncMock()
    return ctrl


def _active_session() -> CardSession:
    session = CardSession("om_source", "oc_chat", asyncio.get_running_loop())
    session.state = SessionState.STREAMING
    session.card_id = "card_1"
    session.card_msg_id = "om_card"
    return session


def test_build_clarify_controls_support_choice_and_free_text() -> None:
    state = ClarifyState(
        clarify_id="abc123",
        session_key="feishu:session",
        question="选哪个?",
        choices=["A", "B"],
    )

    action = build_clarify_add_action(state)
    elements = action["params"]["elements"]

    assert action["params"]["type"] == "insert_before"
    assert action["params"]["target_element_id"] == "loading_icon"
    assert [element["tag"] for element in elements] == ["markdown", "select_static", "form"]
    assert elements[1]["behaviors"][0]["value"]["hermes_lark_clarify_id"] == "abc123"
    assert elements[2]["elements"][0]["required"] is True
    assert elements[2]["elements"][1]["form_action_type"] == "submit"
    assert all(len(element["element_id"]) <= 20 for element in elements)


def test_parse_choice_and_form_submissions() -> None:
    choice = parse_clarify_submission(
        SimpleNamespace(tag="select_static", option="1"),
        {"hermes_lark_clarify_id": "abc123"},
    )
    form = parse_clarify_submission(
        SimpleNamespace(
            tag="button",
            name="submit_abc123",
            form_value={"input_abc123": "  自定义答案  "},
        ),
        {},
    )

    assert choice is not None
    assert choice.clarify_id == "abc123"
    assert choice.choice_index == 1
    assert form is not None
    assert form.clarify_id == "abc123"
    assert form.text == "自定义答案"


def test_resolved_actions_remove_controls_and_record_actor() -> None:
    state = ClarifyState(
        clarify_id="abc123",
        session_key="s",
        question="原问题",
        choices=["A"],
    )

    actions = build_clarify_resolved_actions(
        state,
        response="A",
        actor_name="Alice",
    )

    assert actions[0] == {
        "action": "delete_elements",
        "params": {"element_ids": ["csabc123", "cfabc123"]},
    }
    assert actions[1]["action"] == "update_element"
    assert "Alice 已回答" in actions[1]["params"]["element"]["content"]
    assert "原问题" in actions[1]["params"]["element"]["content"]


@pytest.mark.asyncio
async def test_clarify_request_attaches_to_active_card() -> None:
    ctrl = _setup_controller()
    session = _active_session()
    ctrl._sessions[session.message_id] = session

    claimed = await ctrl.on_clarify_request(
        chat_id="oc_chat",
        question="继续吗?",
        choices=["继续", "停止"],
        clarify_id="abc123",
        session_key="feishu:session",
    )

    assert claimed is True
    assert session.clarify_state is not None
    assert ctrl._clarify_sessions["abc123"] is session
    ctrl._client.cardkit_batch_update.assert_awaited_once()
    action = ctrl._client.cardkit_batch_update.await_args.args[1][0]
    assert action["action"] == "add_elements"


@pytest.mark.asyncio
async def test_clarify_request_falls_back_without_active_card() -> None:
    ctrl = _setup_controller()

    assert await ctrl.on_clarify_request(
        chat_id="oc_missing",
        question="继续吗?",
        choices=None,
        clarify_id="abc123",
        session_key="feishu:session",
    ) is False
    ctrl._client.cardkit_batch_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_choice_records_operator_and_resolves_hermes() -> None:
    ctrl = _setup_controller()
    session = _active_session()
    ctrl._sessions[session.message_id] = session
    assert await ctrl.on_clarify_request(
        chat_id="oc_chat",
        question="选哪个?",
        choices=["A", "B"],
        clarify_id="abc123",
        session_key="feishu:session",
    )

    scheduled: list[asyncio.Task[None]] = []
    adapter = MagicMock()
    adapter._allow_group_message.return_value = True
    adapter._resolve_sender_profile = AsyncMock(return_value={"user_name": "Alice"})

    def submit(_loop: object, coro: object) -> bool:
        scheduled.append(asyncio.create_task(coro))  # type: ignore[arg-type]
        return True

    adapter._submit_on_loop.side_effect = submit
    event = SimpleNamespace(
        context=SimpleNamespace(open_chat_id="oc_chat"),
        operator=SimpleNamespace(open_id="ou_alice", user_id="u_alice"),
    )
    action = SimpleNamespace(tag="select_static", option="1")
    resolved: list[tuple[str, str]] = []
    tools_module = ModuleType("tools")
    clarify_module = ModuleType("tools.clarify_gateway")
    clarify_module.resolve_gateway_clarify = lambda cid, answer: resolved.append((cid, answer)) or True  # type: ignore[attr-defined]
    tools_module.clarify_gateway = clarify_module  # type: ignore[attr-defined]

    with patch.dict(
        sys.modules,
        {"tools": tools_module, "tools.clarify_gateway": clarify_module},
    ):
        claimed = ctrl.on_clarify_action(
            adapter=adapter,
            event=event,
            action=action,
            action_value={"hermes_lark_clarify_id": "abc123"},
            loop=asyncio.get_running_loop(),
        )
        assert claimed is True
        await scheduled[0]

    assert resolved == [("abc123", "B")]
    assert session.clarify_state is None
    assert len(session.clarify_results) == 1
    assert session.clarify_results[0].question == "选哪个?"
    assert session.clarify_results[0].response == "B"
    assert session.clarify_results[0].actor_name == "Alice"
    assert "abc123" not in ctrl._clarify_sessions
    final_actions = ctrl._client.cardkit_batch_update.await_args_list[-1].args[1]
    assert "Alice 已回答" in final_actions[1]["params"]["element"]["content"]
    adapter._allow_group_message.assert_called_once()


@pytest.mark.asyncio
async def test_unauthorized_group_operator_is_recorded_but_not_accepted() -> None:
    ctrl = _setup_controller()
    session = _active_session()
    ctrl._sessions[session.message_id] = session
    assert await ctrl.on_clarify_request(
        chat_id="oc_chat",
        question="选哪个?",
        choices=["A"],
        clarify_id="abc123",
        session_key="feishu:session",
    )

    adapter = MagicMock()
    adapter._allow_group_message.return_value = False
    event = SimpleNamespace(
        context=SimpleNamespace(open_chat_id="oc_chat"),
        operator=SimpleNamespace(open_id="ou_blocked", user_id="u_blocked"),
    )

    claimed = ctrl.on_clarify_action(
        adapter=adapter,
        event=event,
        action=SimpleNamespace(tag="select_static", option="0"),
        action_value={"hermes_lark_clarify_id": "abc123"},
        loop=asyncio.get_running_loop(),
    )

    assert claimed is True
    assert session.clarify_state is not None
    assert session.clarify_state.resolving is False
    adapter._submit_on_loop.assert_not_called()
