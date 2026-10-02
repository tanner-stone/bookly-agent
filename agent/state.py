"""Graph state. Money decisions are stored here after code makes them.

The checkpointer persists this dict between turns. Values have to be plain
JSON-like objects: no Decimal, no datetime.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    channel: str
    intent: str | None
    customer_id: str | None
    email_attempts: int
    candidate_order_ids: list[str]
    selected_order_id: str | None
    refund_reason: str | None
    refund_decision: dict[str, Any] | None
    # The order a refund was offered on last turn. A yes only counts for this id.
    proposed_refund: str | None
    # What the model reported for this turn, after code dropped ids the customer does not own.
    read: dict[str, Any] | None
    awaiting: str | None
    directive: dict[str, Any] | None
    turn_tools: list[dict[str, Any]]
    tool_steps: int
    policy_fresh: bool
    trace: list[dict[str, Any]]
