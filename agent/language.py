"""Model calls used by the graph. Tests pass a scripted stand-in.

`understand` reads the turn and may call read-only tools. `reply` phrases what
code decided and may also read. `decide_refund`, `issue_refund`, and
`create_ticket` are not on the model.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field, ValidationError

from agent.prompts import phrase_system, understand_system
from policy.config import TOOL_LOOP_CAP
from tools.read_tools import READ_ONLY_NAMES, READ_ONLY_TOOLS

ReadRunner = Callable[[str, dict], Any]


class TurnRead(BaseModel):
    """Your reading of the customer's latest message. Call this once you know."""

    goal: Literal["order_status", "refund", "faq", "handoff", "resolved", "rate", "chat"] = Field(
        description="What the customer wants from this message, given the open question."
    )
    candidate_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Every order that fits what they said, when they describe an order rather than pick "
            "one from cards on screen. Empty when they pick a card or name nothing."
        ),
    )
    order_id: str | None = Field(
        default=None,
        description=(
            "The one order this message is about, for any goal, including a policy question about "
            "a book they bought: the only fitting order, or the card they picked. Copied from the "
            "orders list. Null when none or more than one fits."
        ),
    )
    faq_query: str | None = Field(
        default=None,
        description=(
            "When goal is faq: the customer's question as one complete sentence that stands on "
            "its own, in their voice, such as 'Can I return an ebook for a refund?'. Not keywords. "
            "If their message already stands alone, copy it."
        ),
    )
    refund_reason: str | None = Field(
        default=None,
        description="What went wrong, in their words. Null if they only asked for a refund.",
    )
    confirmation: Literal["yes", "no", "none"] = Field(
        default="none",
        description="Their answer to a refund waiting on yes. none if nothing is waiting or they did not answer.",
    )
    stars: int | None = Field(default=None, description="Their score when a rating is open.")
    review: str | None = Field(default=None, description="Their written note when a review is open.")


TURN_READ_FALLBACK = {"goal": "chat", "candidate_ids": [], "confirmation": "none"}


class Language(Protocol):
    def understand(self, view: dict, conversation: list, run_read: ReadRunner) -> dict: ...

    def reply(self, state: dict) -> AIMessage: ...


class XAILanguage:
    def __init__(self, model: str | None = None):
        from langchain_xai import ChatXAI

        name = model or os.environ.get("XAI_MODEL", "grok-4.3")
        # grok-4.3 is the fast tool-calling model. Reasoning is off because
        # money and policy are decided by code, not by a long model think.
        effort = os.environ.get("XAI_REASONING_EFFORT", "none")
        self.llm = ChatXAI(
            model=name,
            temperature=0,
            extra_body={"reasoning_effort": effort} if effort else None,
        )

    def understand(self, view: dict, conversation: list, run_read: ReadRunner) -> dict:
        messages: list = [SystemMessage(content=understand_system())]
        messages.extend(_spoken(conversation))
        messages.append(SystemMessage(content="Context:\n" + json.dumps(view, default=str)))
        bound = self.llm.bind_tools([*READ_ONLY_TOOLS, TurnRead], tool_choice="required")
        try:
            for _ in range(TOOL_LOOP_CAP):
                answer = bound.invoke(messages)
                calls = getattr(answer, "tool_calls", None) or []
                for call in calls:
                    if call["name"] == "TurnRead":
                        return _turn_read(call.get("args") or {})
                reads = [call for call in calls if call["name"] in READ_ONLY_NAMES]
                if not reads:
                    break
                messages.append(answer)
                for call in reads:
                    result = run_read(call["name"], call.get("args") or {})
                    messages.append(
                        ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
                    )
            final = self.llm.with_structured_output(TurnRead).invoke(messages)
        except Exception:
            return dict(TURN_READ_FALLBACK)
        if isinstance(final, TurnRead):
            return final.model_dump()
        return _turn_read(final or {})

    def reply(self, state: dict) -> AIMessage:
        channel = state.get("channel") or "chat"
        directive = state.get("directive") or {"kind": "clarify"}
        messages = [SystemMessage(content=phrase_system(channel))]
        messages.extend(state.get("messages") or [])
        messages.append(
            SystemMessage(content="Directive:\n" + json.dumps(directive, default=str))
        )
        return self.llm.bind_tools(READ_ONLY_TOOLS).invoke(messages)


def _turn_read(args: dict) -> dict:
    try:
        return TurnRead.model_validate(args).model_dump()
    except ValidationError:
        return dict(TURN_READ_FALLBACK)


def _spoken(conversation: list) -> list:
    """Customer and assistant lines only. Tool traffic from earlier turns stays out."""

    kept = []
    for message in conversation:
        kind = getattr(message, "type", None)
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if kind == "human":
            kept.append(HumanMessage(content=content))
        elif kind == "ai" and not getattr(message, "tool_calls", None):
            kept.append(AIMessage(content=content))
    return kept
