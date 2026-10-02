"""Explicit state graph. No prebuilt agent.

Each turn: route, then the model reads the turn (understand), code checks
that read and picks the next node. Read tools loop between phrase and
execute_read_tools. Refund writes are separate nodes the model cannot route
to by naming a tool.
"""

from __future__ import annotations

from datetime import date

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.language import Language
from agent.runtime import BooklyRuntime, after_phrase, route_next
from agent.state import AgentState
from policy.refund_policy import load_refund_policy
from tools.store import Store


def build_graph(
    language: Language,
    store: Store,
    checkpointer: BaseCheckpointSaver,
    today: date | None = None,
    faq=None,
):
    # Fail here, before any turn, if the refund file is invalid.
    policy = load_refund_policy()
    runtime = BooklyRuntime(language, store, policy, today, faq)
    graph = StateGraph(AgentState)
    graph.add_node("route", runtime.route)
    graph.add_node("greet", runtime.greet)
    graph.add_node("understand", runtime.understand)
    graph.add_node("answer_faq", runtime.answer_faq)
    graph.add_node("fetch_order", runtime.fetch_order)
    graph.add_node("apply_policy", runtime.apply_policy)
    graph.add_node("issue_refund", runtime.issue_refund)
    graph.add_node("create_ticket", runtime.create_ticket)
    graph.add_node("save_rating", runtime.save_rating)
    graph.add_node("save_review", runtime.save_review)
    graph.add_node("phrase", runtime.phrase)
    graph.add_node("execute_read_tools", runtime.execute_read_tools)
    graph.add_node("finish", runtime.finish)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        route_next,
        {"greet": "greet", "understand": "understand"},
    )
    graph.add_conditional_edges(
        "phrase",
        after_phrase,
        {"execute_read_tools": "execute_read_tools", "finish": "finish"},
    )
    graph.add_edge("execute_read_tools", "phrase")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
