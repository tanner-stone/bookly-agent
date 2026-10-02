"""Opt-in scenarios against the real model.

Run: python3 -m evals.run

Pytest does not collect this. It checks the trace and the store, not the
wording. The store is in memory, so a run does not refund the demo database.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from agent.graph import build_graph
from agent.language import XAILanguage
from data.seed import build_seed
from tools.faq_search import NullFaq, VectorFaqIndex, VoyageEmbedder
from tools.store import MemoryStore


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("XAI_API_KEY"):
        print("XAI_API_KEY is not set.", file=sys.stderr)
        return 1
    today = date.today()
    data = build_seed(today)
    store = MemoryStore(data.customers, data.orders, data.refunds)
    faq, faq_note = _faq_index()
    graph = build_graph(XAILanguage(), store, InMemorySaver(), today=today, faq=faq)
    scenarios = list(_scenarios(store))
    if isinstance(faq, NullFaq):
        print(f"SKIP faq scenarios: {faq_note}")
    else:
        scenarios += _faq_scenarios()
    failures = []
    for name, check in scenarios:
        try:
            problem = check(graph, store)
        except Exception as exc:
            problem = f"{type(exc).__name__}: {exc}"
        if problem:
            failures.append(f"FAIL {name}: {problem}")
            print(failures[-1])
        else:
            print(f"PASS {name}")
    if failures:
        print(f"{len(failures)} failed", file=sys.stderr)
        return 1
    print("all scenarios passed")
    return 0


def _scenarios(store: MemoryStore):
    def customer(email: str) -> str:
        return store.get_customer_by_email(email)["_id"]

    return [
        ("maya standard refund", lambda graph, store: _refund(
            graph, store, customer("maya.chen@bookly.example"), "BK-10231", "standard_refund", "AUTO_REFUND", expect_write=True
        )),
        ("jordan goodwill", lambda graph, store: _refund(
            graph, store, customer("jordan.lee@bookly.example"), "BK-10232", "first_time_goodwill", "AUTO_REFUND", expect_write=True
        )),
        ("sam escalate", lambda graph, store: _refund(
            graph, store, customer("sam.patel@bookly.example"), "BK-10233", "escalate", "ESCALATE", expect_write=False, expect_ticket=True
        )),
        ("riley not delivered", lambda graph, store: _riley(graph, store, customer("riley.nguyen@bookly.example"))),
        ("alex second order", lambda graph, store: _alex(graph, store, customer("alex.rivera@bookly.example"))),
        ("casey already refunded", lambda graph, store: _refund(
            graph, store, customer("casey.brooks@bookly.example"), "BK-10238", "already_refunded", "ALREADY_REFUNDED", expect_write=False
        )),
        ("taylor escalate", lambda graph, store: _refund(
            graph, store, customer("taylor.kim@bookly.example"), "BK-10239", "escalate", "ESCALATE", expect_write=False, expect_ticket=True
        )),
        ("quinn cancelled", lambda graph, store: _refund(
            graph, store, customer("quinn.alvarez@bookly.example"), "BK-10240", "cancelled", "NOT_APPLICABLE", expect_write=False
        )),
        ("tanner the older one", lambda graph, store: _tanner_older(graph, store, TANNER)),
        ("tanner whichever first", lambda graph, store: _tanner_first(graph, store, TANNER)),
        ("tanner misspelled title", lambda graph, store: _tanner_typo(graph, store, TANNER)),
        ("tanner goldfinch escalates", lambda graph, store: _tanner_goldfinch(graph, store, TANNER)),
        ("tanner overstory mid-refund", lambda graph, store: _tanner_switch(graph, store, TANNER)),
        ("tanner casual yes refunds", lambda graph, store: _tanner_casual_yes(graph, store, TANNER)),
    ]


TANNER = "tanner-stone"


def _faq_index():
    """The seeded Atlas vector index, read only. Orders and refunds stay in memory."""

    if not os.environ.get("VOYAGE_API_KEY"):
        return NullFaq(), "VOYAGE_API_KEY is not set"
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        return NullFaq(), "MONGODB_URI is not set"
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        return NullFaq(), "MongoDB is not running"
    database = client[os.environ.get("MONGODB_DB", "bookly")]
    if database.faq.count_documents({}) == 0:
        return NullFaq(), "FAQ is not seeded (python3 -m data.seed)"
    return VectorFaqIndex(database, VoyageEmbedder()), ""


def _faq_scenarios():
    return [
        ("faq moving abroad", lambda graph, store: _faq(
            graph, "faq-canada", "I'm moving to Canada next month. Can you still ship my books there?",
            "International shipping",
        )),
        ("faq preorder delayed", lambda graph, store: _faq(
            graph, "faq-preorder", "I preordered a book and the release date got pushed back. Can I cancel?",
            "Preorders",
        )),
        ("faq expired gift card", lambda graph, store: _faq(
            graph, "faq-gift", "My friend gave me a gift card but it says it's expired. What can I do?",
            "Gift cards",
        )),
        ("faq follow-up keeps context", lambda graph, store: _faq_follow_up(graph)),
        ("faq about my order, then refund", lambda graph, store: _faq_then_refund(graph, store)),
    ]


def _cited(state) -> list[str]:
    exhibit = state["messages"][-1].additional_kwargs.get("exhibit") or {}
    if exhibit.get("kind") != "faq_sources":
        return []
    return [article["title"] for article in exhibit.get("articles") or []]


def _faq(graph, thread, line, title, customer_id=TANNER):
    state = _say(graph, thread, line, customer_id)
    if state.get("intent") != "faq":
        return f"goal was {state.get('intent')}"
    searched = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "search_faq"]
    if searched and not searched[0]["ok"]:
        return f"search failed: {searched[0]['result'].get('error')}"
    if title not in _cited(state):
        return f"cited {_cited(state)}, expected {title}"
    return None


def _faq_follow_up(graph):
    thread = "faq-follow"
    problem = _faq(graph, thread, "What's your return policy?", "Returns and refunds")
    if problem:
        return f"first turn: {problem}"
    return _faq(graph, thread, "What about ebooks?", "Ebooks")


def _faq_then_refund(graph, store):
    thread = "faq-damage"
    state = _say(
        graph, thread, "The Left Hand of Darkness came with water damage on the cover. What's your policy on that?",
        TANNER,
    )
    if state.get("intent") != "faq" or "Damaged or wrong books" not in _cited(state):
        return f"goal {state.get('intent')}, cited {_cited(state)}"
    if (state.get("directive") or {}).get("order", {}).get("order_id") != "BK-10243":
        return "the answer was not tied to BK-10243"
    if _refund_count(store, "BK-10243"):
        return "a policy question wrote a refund"
    state = _say(graph, thread, "Okay, please refund it.", TANNER)
    decision = state.get("refund_decision") or {}
    if decision.get("rule_id") != "standard_refund" or state.get("awaiting") != "refund_confirm":
        return f"refund step: {decision.get('rule_id')} awaiting {state.get('awaiting')}"
    if _refund_count(store, "BK-10243"):
        return "refunded before the yes"
    return None


def _tanner_older(graph, store, customer_id):
    thread = "tanner-older"
    state = _say(graph, thread, "I want a refund for the night book", customer_id)
    shown = set(state.get("candidate_order_ids") or [])
    if not {"BK-10241", "BK-10242"} <= shown:
        return f"cards were {sorted(shown)}"
    state = _say(graph, thread, "The older one.", customer_id)
    if state.get("selected_order_id") != "BK-10242":
        return f"selected {state.get('selected_order_id')}"
    if state.get("intent") != "refund":
        return f"goal became {state.get('intent')}"
    return None


def _tanner_first(graph, store, customer_id):
    thread = "tanner-first"
    state = _say(graph, thread, "Where is my order?", customer_id)
    shown = state.get("candidate_order_ids") or []
    if len(shown) < 2:
        return f"expected cards, got {shown}"
    by_id = {order["_id"]: order for order in store.list_orders(customer_id)}
    oldest = min(shown, key=lambda order_id: by_id[order_id]["ordered_at"])
    state = _say(graph, thread, "whichever I ordered first", customer_id)
    if state.get("selected_order_id") != oldest:
        return f"selected {state.get('selected_order_id')}, expected {oldest}"
    return None


def _tanner_typo(graph, store, customer_id):
    state = _say(graph, "tanner-typo", "where is the disposessed", customer_id)
    picked = state.get("selected_order_id")
    if picked != "BK-10244" and state.get("candidate_order_ids") != ["BK-10244"]:
        return f"selected {picked}, cards {state.get('candidate_order_ids')}"
    return None


def _tanner_goldfinch(graph, store, customer_id):
    tickets_before = len(store.tickets)
    state = _drive(
        graph,
        customer_id,
        "tanner-goldfinch",
        "Can you refund The Goldfinch too? I bought it months ago.",
        "The Goldfinch",
    )
    decision = state.get("refund_decision") or {}
    if decision.get("rule_id") != "escalate":
        return f"rule {decision.get('rule_id')}, intent {state.get('intent')}"
    if len(store.tickets) <= tickets_before or _refund_count(store, "BK-10246"):
        return "escalation did not open exactly a ticket"
    return None


def _tanner_switch(graph, store, customer_id):
    thread = "tanner-switch"
    state = _say(graph, thread, "Please refund The Left Hand of Darkness, the spine was cracked.", customer_id)
    if state.get("awaiting") != "refund_confirm":
        return f"no offer, awaiting {state.get('awaiting')}"
    state = _say(graph, thread, "Actually, where is The Overstory?", customer_id)
    if state.get("selected_order_id") != "BK-10245" or (state.get("directive") or {}).get("kind") != "order_status":
        return f"did not switch: {state.get('selected_order_id')} {(state.get('directive') or {}).get('kind')}"
    if _refund_count(store, "BK-10243"):
        return "switching subjects refunded the earlier order"
    return None


def _tanner_casual_yes(graph, store, customer_id):
    thread = "tanner-yes"
    state = _say(graph, thread, "I'd like a refund for The Night Circus, the cover was torn.", customer_id)
    if state.get("awaiting") != "refund_confirm":
        return f"no offer, awaiting {state.get('awaiting')}"
    _say(graph, thread, "yeah, go for it", customer_id)
    if _refund_count(store, "BK-10241") != 1:
        return "the yes did not refund once"
    return None


def _refund(graph, store, customer_id, order_id, rule_id, outcome, *, expect_write, expect_ticket=False):
    before = _refund_count(store, order_id)
    tickets_before = len(store.tickets)
    state = _drive(graph, customer_id, f"thread-{order_id}", f"I want a refund for {order_id}", order_id)
    decision = state.get("refund_decision") or {}
    if decision.get("rule_id") != rule_id or decision.get("outcome") != outcome:
        return f"decision {decision.get('outcome')} {decision.get('rule_id')}, expected {outcome} {rule_id}"
    wrote = _refund_count(store, order_id) > before
    if wrote != expect_write:
        return f"write={wrote}, expected {expect_write}"
    opened = len(store.tickets) > tickets_before
    if opened != expect_ticket:
        return f"ticket={opened}, expected {expect_ticket}"
    if expect_ticket and store.tickets[-1]["customer_id"] != customer_id:
        return "ticket customer does not match"
    return None


def _riley(graph, store, customer_id):
    tickets_before = len(store.tickets)
    state = _say(graph, "riley-status", "Where is order BK-10234?", customer_id)
    if state.get("intent") != "order_status":
        return f"status intent was {state.get('intent')}"
    if _refund_count(store, "BK-10234"):
        return "status turn wrote a refund"
    state = _drive(graph, customer_id, "riley-refund", "I want a refund for BK-10234", "BK-10234")
    decision = state.get("refund_decision") or {}
    if decision.get("rule_id") != "not_delivered" or decision.get("outcome") != "NOT_ELIGIBLE_YET":
        return f"decision {decision.get('outcome')} {decision.get('rule_id')}"
    if _refund_count(store, "BK-10234") or len(store.tickets) > tickets_before:
        return "not-delivered path wrote a refund or a ticket"
    return None


def _alex(graph, store, customer_id):
    state = _drive(graph, customer_id, "alex", "I want a refund", "the second one")
    if state.get("selected_order_id") != "BK-10236":
        return f"selected {state.get('selected_order_id')}"
    decision = state.get("refund_decision") or {}
    if decision.get("rule_id") != "standard_refund":
        return f"rule {decision.get('rule_id')}"
    if _refund_count(store, "BK-10236") != 1:
        return "second order was not refunded once"
    return None


def _drive(graph, customer_id, thread, opening, choice) -> dict:
    state = _say(graph, thread, opening, customer_id)
    for _ in range(5):
        awaiting = state.get("awaiting")
        if awaiting == "order_choice":
            kind = (state.get("directive") or {}).get("kind")
            state = _say(graph, thread, "yes" if kind == "confirm_order" else choice, customer_id)
        elif awaiting == "refund_reason":
            state = _say(graph, thread, "It arrived damaged", customer_id)
        elif awaiting == "refund_confirm":
            state = _say(graph, thread, "yes", customer_id)
        else:
            return state
    return state


def _say(graph, thread, text, customer_id) -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "channel": "chat",
            "customer_id": customer_id,
        },
        {"configurable": {"thread_id": thread}},
    )


def _refund_count(store, order_id: str) -> int:
    return sum(1 for row in store.refunds if row["order_id"] == order_id)


if __name__ == "__main__":
    raise SystemExit(main())
