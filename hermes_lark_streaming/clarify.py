"""Feishu CardKit components and callback parsing for Hermes clarify prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .cardkit.builder import _LOADING_ELEMENT_ID

_CLARIFY_ACTION_KEY = "hermes_lark_clarify_id"
_ID_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_id(clarify_id: str) -> str:
    value = _ID_RE.sub("", str(clarify_id))[:12]
    return value or "unknown"


def _escape_markdown(value: str) -> str:
    return re.sub(r"([\\`*_{}\[\]<>])", r"\\\1", str(value))


@dataclass
class ClarifyState:
    """One clarify interaction attached to a streaming card."""

    clarify_id: str
    session_key: str
    question: str
    choices: list[str] = field(default_factory=list)
    resolving: bool = False

    @property
    def suffix(self) -> str:
        return _safe_id(self.clarify_id)

    @property
    def question_element_id(self) -> str:
        return f"cq{self.suffix}"

    @property
    def choice_element_id(self) -> str:
        return f"cs{self.suffix}"

    @property
    def form_element_id(self) -> str:
        return f"cf{self.suffix}"

    @property
    def input_name(self) -> str:
        return f"input_{self.suffix}"

    @property
    def submit_name(self) -> str:
        return f"submit_{self.suffix}"

    @property
    def interactive_element_ids(self) -> list[str]:
        element_ids = [self.form_element_id]
        if self.choices:
            element_ids.insert(0, self.choice_element_id)
        return element_ids


@dataclass(frozen=True)
class ClarifySubmission:
    clarify_id: str
    choice_index: int | None = None
    text: str = ""


def build_clarify_add_action(state: ClarifyState) -> dict[str, Any]:
    """Build a CardKit batch action that inserts choice/input controls."""
    elements: list[dict[str, Any]] = [
        {
            "tag": "markdown",
            "content": f"❓ **需要你的回答**\n\n{state.question}",
            "element_id": state.question_element_id,
        }
    ]

    if state.choices:
        elements.append(
            {
                "tag": "select_static",
                "name": f"choice_{state.suffix}",
                "placeholder": {"tag": "plain_text", "content": "选择一个选项"},
                "options": [
                    {
                        "text": {"tag": "plain_text", "content": str(choice)[:100]},
                        "value": str(index),
                    }
                    for index, choice in enumerate(state.choices)
                ],
                "width": "fill",
                "behaviors": [
                    {
                        "type": "callback",
                        "value": {_CLARIFY_ACTION_KEY: state.clarify_id},
                    }
                ],
                "element_id": state.choice_element_id,
            }
        )

    input_label = "或者输入其他回答" if state.choices else "输入回答"
    elements.append(
        {
            "tag": "form",
            "name": f"form_{state.suffix}",
            "element_id": state.form_element_id,
            "elements": [
                {
                    "tag": "input",
                    "name": state.input_name,
                    "required": True,
                    "label": {"tag": "plain_text", "content": input_label},
                    "placeholder": {"tag": "plain_text", "content": "请输入"},
                    "input_type": "multiline_text",
                    "rows": 2,
                    "auto_resize": True,
                    "max_rows": 6,
                    "max_length": 1000,
                    "width": "fill",
                },
                {
                    "tag": "button",
                    "name": state.submit_name,
                    "text": {"tag": "plain_text", "content": "提交回答"},
                    "type": "primary_filled",
                    "width": "fill",
                    "form_action_type": "submit",
                },
            ],
        }
    )

    return {
        "action": "add_elements",
        "params": {
            "type": "insert_before",
            "target_element_id": _LOADING_ELEMENT_ID,
            "elements": elements,
        },
    }


def build_clarify_resolved_actions(
    state: ClarifyState,
    *,
    response: str,
    actor_name: str,
) -> list[dict[str, Any]]:
    """Replace interactive controls with a compact, auditable answer."""
    actor = _escape_markdown(actor_name or "未知用户")
    answer = _escape_markdown(response)
    return [
        {
            "action": "delete_elements",
            "params": {"element_ids": state.interactive_element_ids},
        },
        {
            "action": "update_element",
            "params": {
                "element_id": state.question_element_id,
                "element": {
                    "tag": "markdown",
                    "content": (
                        f"✅ **{actor} 已回答**: {answer}\n\n"
                        f"<font color='grey'>原问题: {state.question}</font>"
                    ),
                    "element_id": state.question_element_id,
                },
            },
        },
    ]


def parse_clarify_submission(action: Any, action_value: Any) -> ClarifySubmission | None:
    """Parse either a select callback or a free-text form submission."""
    value = action_value if isinstance(action_value, dict) else {}
    clarify_id = str(value.get(_CLARIFY_ACTION_KEY) or "").strip()
    tag = str(getattr(action, "tag", "") or "")

    if clarify_id and tag == "select_static":
        option = str(getattr(action, "option", "") or "").strip()
        try:
            choice_index = int(option)
        except ValueError:
            return ClarifySubmission(clarify_id=clarify_id)
        return ClarifySubmission(clarify_id=clarify_id, choice_index=choice_index)

    name = str(getattr(action, "name", "") or "")
    if tag != "button" or not name.startswith("submit_"):
        return None

    suffix = name.removeprefix("submit_")
    form_value = getattr(action, "form_value", None)
    if not isinstance(form_value, dict):
        return None
    text = str(form_value.get(f"input_{suffix}") or "").strip()
    return ClarifySubmission(clarify_id=suffix, text=text)
