"""Multi-turn flows against a scripted model read and an in-memory checkpointer.

Each test scripts what the model reads for each line (goal, order, reason,
yes or no, stars). The assertions are about what code does with that read:
tool calls, rule ids, and whether money moved. Not the wording. One test uses
the Mongo checkpointer when Atlas Local is up. It still does not call the
model.
"""

import os
import uuid
from datetime import date
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from agent.graph import build_graph
from agent.matching import owned_order_ids
from agent.prompts import phrase_system
from data.seed import build_seed
from tests.fakes import ScriptedLanguage, StaticFaq
from tools.read_tools import READ_ONLY_NAMES
from tools.store import MemoryStore, RefundWriteError

TODAY = date(2026, 6, 15)
MAYA = "maya-chen"
TANNER = "tanner-stone"


def _graph(script: dict | None = None, behavior: str = "text", faq=None):
    data = build_seed(TODAY)
    store = MemoryStore(data.customers, data.orders, data.refunds)
    language = ScriptedLanguage(script, behavior)
    graph = build_graph(language, store, InMemorySaver(), today=TODAY, faq=faq)
    return graph, store, language


def _say(graph, thread: str, text: str, *, customer_id: str | None = None) -> dict:
    payload = {"messages": [HumanMessage(content=text)], "channel": "chat"}
    if customer_id is not None:
        payload["customer_id"] = customer_id
    return graph.invoke(payload, {"configurable": {"thread_id": thread}})


def _refunds(store: MemoryStore, order_id: str) -> list:
    return [row for row in store.refunds if row["order_id"] == order_id]


def _refund(order_id: str, reason: str | None = None, **extra) -> dict:
    return {"goal": "refund", "order_id": order_id, "refund_reason": reason, **extra}


YES = {"goal": "refund", "confirmation": "yes"}
NO = {"goal": "refund", "confirmation": "no"}
DONE = {"goal": "resolved"}


def _maya_offered(extra: dict | None = None):
    """Maya has one delivered order under the standard limits, offered and waiting on yes."""

    script = {"I want a refund, it arrived damaged": _refund("BK-10231", "It arrived damaged")}
    script.update(extra or {})
    graph, store, language = _graph(script)
    state = _say(graph, "maya", "I want a refund, it arrived damaged", customer_id=MAYA)
    assert state["awaiting"] == "refund_confirm"
    assert state["proposed_refund"] == "BK-10231"
    assert _refunds(store, "BK-10231") == []
    return graph, store, language


def test_read_tools_do_not_include_writes():
    assert READ_ONLY_NAMES == {"get_order", "search_orders", "search_faq"}


def test_voice_prompt_is_spoken_and_chat_prompt_is_not():
    voice = phrase_system("voice")
    chat = phrase_system("chat")
    assert "markdown" in voice.lower()
    assert "markdown" not in chat.lower()


def test_owned_ids_drop_what_the_customer_does_not_own():
    orders = [{"_id": "BK-10241"}, {"_id": "BK-10242"}]
    assert owned_order_ids(["bk-10242", "BK-10239", "BK-10242", 7], orders) == ["BK-10242"]


def test_greeting_does_not_call_the_model():
    graph, _store, language = _graph()
    state = graph.invoke(
        {"customer_id": MAYA, "channel": "chat"},
        {"configurable": {"thread_id": "hello"}},
    )
    assert state["messages"][-1].content == "Maya, how can I assist you today?"
    assert language.understand_calls == 0


def test_the_model_sees_the_orders_and_what_is_open():
    graph, _store, language = _graph(
        {"I want a refund for the night book": {"goal": "refund", "candidate_ids": ["BK-10241", "BK-10242"]}}
    )
    _say(graph, "view", "I want a refund for the night book", customer_id=TANNER)
    _say(graph, "view", "the older one", customer_id=TANNER)
    first, second = language.views
    assert {order["order_id"] for order in first["orders"]} == {
        "BK-10241", "BK-10242", "BK-10243", "BK-10244", "BK-10245", "BK-10246",
    }
    assert "points" not in str(first["orders"])
    assert first["today"] == TODAY.isoformat()
    assert second["open"]["goal"] == "refund"
    assert second["open"]["waiting_on"] == "order_choice"
    cards = second["open"]["cards_on_screen"]
    assert [(card["position"], card["order_id"]) for card in cards] == [(1, "BK-10241"), (2, "BK-10242")]
    assert all(card["ordered_at"] for card in cards)


def test_candidates_show_cards_and_the_models_pick_lands():
    graph, _store, _language = _graph(
        {
            "I want a refund for the night book": {"goal": "refund", "candidate_ids": ["BK-10241", "BK-10242"]},
            "the older one.": _refund("BK-10242"),
        }
    )
    state = _say(graph, "night", "I want a refund for the night book", customer_id=TANNER)
    assert state["directive"]["kind"] == "choose_order"
    assert [order["order_id"] for order in state["directive"]["orders"]] == ["BK-10241", "BK-10242"]
    exhibit = state["messages"][-1].additional_kwargs["exhibit"]
    assert [order["order_id"] for order in exhibit["orders"]] == ["BK-10241", "BK-10242"]
    state = _say(graph, "night", "the older one.", customer_id=TANNER)
    assert state["selected_order_id"] == "BK-10242"
    assert state["awaiting"] == "refund_reason"
    assert state["directive"]["kind"] == "ask_reason"
    assert state["trace"][-1]["read"]["order_id"] == "BK-10242"


def test_one_pick_with_other_fits_lets_the_customer_choose():
    graph, store, _language = _graph(
        {
            "refund the night book, it was torn": _refund(
                "BK-10241", "torn", candidate_ids=["BK-10242"]
            )
        }
    )
    state = _say(graph, "both", "refund the night book, it was torn", customer_id=TANNER)
    assert state["directive"]["kind"] == "choose_order"
    assert state["candidate_order_ids"] == ["BK-10241", "BK-10242"]
    assert state["refund_reason"] == "torn"
    assert state.get("refund_decision") is None


def test_one_candidate_asks_to_confirm_it():
    graph, _store, _language = _graph({"the disposessed": {"goal": "order_status", "candidate_ids": ["BK-10244"]}})
    state = _say(graph, "one", "the disposessed", customer_id=TANNER)
    assert state["directive"]["kind"] == "confirm_order"
    assert state["candidate_order_ids"] == ["BK-10244"]


def test_an_id_the_customer_does_not_own_shows_cards():
    graph, store, _language = _graph({"refund BK-10239": _refund("BK-10239", "damaged")})
    state = _say(graph, "foreign", "refund BK-10239", customer_id=TANNER)
    assert state["directive"]["kind"] == "choose_order"
    assert "BK-10239" not in state["candidate_order_ids"]
    assert state["trace"][-1]["read"]["dropped_ids"] == ["BK-10239"]
    assert state.get("refund_decision") is None
    assert _refunds(store, "BK-10239") == []


def test_standard_refund_waits_for_the_models_yes():
    graph, store, _language = _maya_offered({"yeah, go for it": YES})
    state = graph.get_state({"configurable": {"thread_id": "maya"}}).values
    assert state["refund_decision"]["rule_id"] == "standard_refund"
    assert state["directive"]["ask_confirm"] is True
    assert state["trace"][-1]["policy"]["rule_id"] == "standard_refund"

    state = _say(graph, "maya", "yeah, go for it", customer_id=MAYA)
    assert state["directive"]["kind"] == "refund_issued"
    assert state["awaiting"] is None
    assert state["proposed_refund"] is None
    assert _refunds(store, "BK-10231")[0]["tier"] == "standard"
    assert store.orders["BK-10231"]["status"] == "refunded"
    assert store.customers[MAYA]["prior_refund_count"] == 1
    issued = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "issue_refund"]
    assert issued and issued[0]["caller"] == "code" and issued[0]["ok"] is True
    assert state["trace"][-1]["read"]["confirmation"] == "yes"


def test_a_yes_with_nothing_offered_does_not_write():
    graph, store, _language = _graph({"yes": _refund("BK-10231", "damaged", confirmation="yes")})
    state = _say(graph, "cold-yes", "yes", customer_id=MAYA)
    assert state["awaiting"] == "refund_confirm"
    assert state["directive"]["kind"] == "refund_decision"
    assert _refunds(store, "BK-10231") == []


def test_a_yes_for_a_different_order_does_not_write():
    script = {
        "refund the night circus, it was damaged": _refund("BK-10241", "damaged"),
        "yes, the left hand of darkness": _refund("BK-10243", confirmation="yes"),
    }
    graph, store, _language = _graph(script)
    state = _say(graph, "other", "refund the night circus, it was damaged", customer_id=TANNER)
    assert state["proposed_refund"] == "BK-10241"
    state = _say(graph, "other", "yes, the left hand of darkness", customer_id=TANNER)
    assert state["proposed_refund"] is None
    assert state["selected_order_id"] == "BK-10243"
    assert state["directive"]["kind"] == "ask_reason"
    assert _refunds(store, "BK-10241") == []
    assert _refunds(store, "BK-10243") == []


def test_a_yes_when_the_fresh_decision_changed_does_not_write():
    graph, store, _language = _maya_offered({"yes": YES})
    store.orders["BK-10231"]["status"] = "cancelled"
    state = _say(graph, "maya", "yes", customer_id=MAYA)
    assert state["directive"]["kind"] == "refund_decision"
    assert state["refund_decision"]["rule_id"] == "cancelled"
    assert state["trace"][-1]["policy"]["rule_id"] == "cancelled"
    assert _refunds(store, "BK-10231") == []


def test_a_no_declines_and_clears_the_offer():
    graph, store, _language = _maya_offered({"actually no": NO, "yes": YES})
    state = _say(graph, "maya", "actually no", customer_id=MAYA)
    assert state["directive"]["kind"] == "refund_declined"
    assert state["proposed_refund"] is None
    _say(graph, "maya", "yes", customer_id=MAYA)
    assert _refunds(store, "BK-10231") == []


def test_a_hedge_keeps_the_offer_open_without_paying():
    graph, store, _language = _maya_offered(
        {"I guess": {"goal": "refund", "confirmation": "none"}, "yes": YES}
    )
    state = _say(graph, "maya", "I guess", customer_id=MAYA)
    assert state["directive"]["kind"] == "confirm_unclear"
    assert state["awaiting"] == "refund_confirm"
    assert _refunds(store, "BK-10231") == []
    state = _say(graph, "maya", "yes", customer_id=MAYA)
    assert state["directive"]["kind"] == "refund_issued"
    assert len(_refunds(store, "BK-10231")) == 1


def test_a_subject_change_mid_refund_clears_the_pending_yes():
    script = {
        "refund the night circus, it was damaged": _refund("BK-10241", "damaged"),
        "wait, where is the overstory?": {"goal": "order_status", "order_id": "BK-10245"},
        "ok yes": YES,
    }
    graph, store, _language = _graph(script)
    _say(graph, "switch", "refund the night circus, it was damaged", customer_id=TANNER)
    state = _say(graph, "switch", "wait, where is the overstory?", customer_id=TANNER)
    assert state["directive"]["kind"] == "order_status"
    assert state["directive"]["order"]["tracking"]["place"] == "Chicago"
    assert state["proposed_refund"] is None
    _say(graph, "switch", "ok yes", customer_id=TANNER)
    assert _refunds(store, "BK-10241") == []


def test_the_reason_is_asked_then_carried_to_policy():
    graph, store, _language = _graph(
        {
            "Refund BK-10232 please": _refund("BK-10232"),
            "I finished it and did not love it": _refund("BK-10232", "did not love it"),
            "yes": YES,
        }
    )
    state = _say(graph, "jordan", "Refund BK-10232 please", customer_id="jordan-lee")
    assert state["awaiting"] == "refund_reason"
    state = _say(graph, "jordan", "I finished it and did not love it", customer_id="jordan-lee")
    assert state["refund_decision"]["rule_id"] == "first_time_goodwill"
    assert "one-time" in state["refund_decision"]["customer_note"]
    assert _refunds(store, "BK-10232") == []
    _say(graph, "jordan", "yes", customer_id="jordan-lee")
    assert _refunds(store, "BK-10232")[0]["tier"] == "first_time_goodwill"


def test_an_answer_to_the_reason_question_counts_without_the_model_restating_it():
    graph, _store, _language = _graph(
        {"Refund BK-10232 please": _refund("BK-10232"), "it was boring": _refund("BK-10232")}
    )
    _say(graph, "jordan", "Refund BK-10232 please", customer_id="jordan-lee")
    state = _say(graph, "jordan", "it was boring", customer_id="jordan-lee")
    assert state["refund_reason"] == "it was boring"
    assert state["refund_decision"]["rule_id"] == "first_time_goodwill"


def test_a_reason_given_before_the_pick_is_kept():
    graph, _store, _language = _graph(
        {
            "my night book came damaged, refund it": {
                "goal": "refund",
                "candidate_ids": ["BK-10241", "BK-10242"],
                "refund_reason": "came damaged",
            },
            "the circus one": _refund("BK-10241"),
        }
    )
    _say(graph, "carry", "my night book came damaged, refund it", customer_id=TANNER)
    state = _say(graph, "carry", "the circus one", customer_id=TANNER)
    assert state["refund_decision"]["rule_id"] == "standard_refund"
    assert state["awaiting"] == "refund_confirm"


@pytest.mark.parametrize(
    ("customer", "order_id", "rule_id", "outcome", "tickets"),
    [
        ("maya-chen", "BK-10231", "standard_refund", "AUTO_REFUND", 0),
        ("jordan-lee", "BK-10232", "first_time_goodwill", "AUTO_REFUND", 0),
        ("sam-patel", "BK-10233", "escalate", "ESCALATE", 1),
        ("riley-nguyen", "BK-10234", "not_delivered", "NOT_ELIGIBLE_YET", 0),
        ("casey-brooks", "BK-10238", "already_refunded", "ALREADY_REFUNDED", 0),
        ("quinn-alvarez", "BK-10240", "cancelled", "NOT_APPLICABLE", 0),
        ("tanner-stone", "BK-10246", "escalate", "ESCALATE", 1),
    ],
)
def test_the_rule_comes_from_policy_not_the_model(customer, order_id, rule_id, outcome, tickets):
    line = "Can you refund this one too? I bought it months ago."
    graph, store, _language = _graph({line: _refund(order_id, "bought it months ago")})
    prior = len(_refunds(store, order_id))
    state = _say(graph, f"rule-{order_id}", line, customer_id=customer)
    assert state["refund_decision"]["rule_id"] == rule_id
    assert state["refund_decision"]["outcome"] == outcome
    assert state["trace"][-1]["policy"]["rule_id"] == rule_id
    assert len(store.tickets) == tickets
    assert len(_refunds(store, order_id)) == prior
    if outcome == "ESCALATE":
        assert state["directive"]["kind"] == "handoff"
        assert state["directive"]["decision"]["rule_id"] == "escalate"
        assert store.tickets[0]["reason"] == "escalate"
        assert state["awaiting"] is None
    if rule_id == "not_delivered":
        assert state["directive"]["order"]["tracking_number"] == "1Z999AA10123456784"


def test_a_second_refund_on_the_same_order_stops():
    graph, store, _language = _maya_offered({"yes": YES, "refund it again": _refund("BK-10231", "still damaged")})
    _say(graph, "maya", "yes", customer_id=MAYA)
    state = _say(graph, "maya", "refund it again", customer_id=MAYA)
    assert state["refund_decision"]["rule_id"] == "already_refunded"
    assert state["directive"]["ask_confirm"] is False
    assert len(_refunds(store, "BK-10231")) == 1


def test_status_reads_the_order_and_keeps_the_map():
    graph, _store, _language = _graph({"where is the disposessed?": {"goal": "order_status", "order_id": "BK-10244"}})
    state = _say(graph, "status", "where is the disposessed?", customer_id=TANNER)
    assert state["directive"]["kind"] == "order_status"
    tracking = state["directive"]["order"]["tracking"]
    assert tracking["place"] == "Denver"
    assert tracking["points"][-1]["lat"] == 39.739
    assert any(call["name"] == "get_order" and call["caller"] == "code" for call in state["trace"][-1]["tool_calls"])
    exhibit = state["messages"][-1].additional_kwargs["exhibit"]
    assert exhibit["order"]["tracking"]["place"] == "Denver"


def test_no_order_named_lists_recent_orders():
    graph, _store, _language = _graph({"My order.": {"goal": "order_status"}})
    state = _say(graph, "vague", "My order.", customer_id=TANNER)
    assert state["directive"]["kind"] == "choose_order"
    assert state["awaiting"] == "order_choice"
    assert len(state["directive"]["orders"]) == 4


def test_chat_while_cards_are_open_keeps_them():
    graph, _store, _language = _graph(
        {"I want a refund for the night book": {"goal": "refund", "candidate_ids": ["BK-10241", "BK-10242"]}}
    )
    _say(graph, "hmm", "I want a refund for the night book", customer_id=TANNER)
    state = _say(graph, "hmm", "hmm", customer_id=TANNER)
    assert state["directive"]["kind"] == "clarify"
    assert state["awaiting"] == "order_choice"
    assert state["candidate_order_ids"] == ["BK-10241", "BK-10242"]


def test_the_model_can_filter_orders_and_the_call_is_logged():
    def find(view, run_read):
        found = run_read("search_orders", {"title": "dispossessed", "customer_id": "maya-chen"})
        return {"goal": "order_status", "order_id": found[0]["order_id"]}

    graph, _store, _language = _graph({"where's the dispossessed": find})
    state = _say(graph, "search", "where's the dispossessed", customer_id=TANNER)
    assert state["selected_order_id"] == "BK-10244"
    searched = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "search_orders"]
    assert searched and searched[0]["caller"] == "model" and searched[0]["ok"]
    assert [row["order_id"] for row in searched[0]["result"]] == ["BK-10244"]


def test_the_model_cannot_write_from_its_read_step():
    def sneak(view, run_read):
        result = run_read("issue_refund", {"order_id": "BK-10231"})
        assert result["error"] == "refused"
        return {"goal": "chat"}

    graph, store, _language = _graph({"refund now": sneak})
    state = _say(graph, "sneak", "refund now", customer_id=MAYA)
    refused = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "issue_refund"]
    assert refused and refused[0]["ok"] is False and refused[0]["caller"] == "model"
    assert _refunds(store, "BK-10231") == []


def test_model_cannot_call_issue_refund_while_phrasing():
    graph, store, _language = _graph(
        {"Where is my package?": {"goal": "order_status", "order_id": "BK-10231"}}, "call_issue_refund"
    )
    state = _say(graph, "refused", "Where is my package?", customer_id=MAYA)
    refused = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "issue_refund"]
    assert refused and refused[0]["ok"] is False and refused[0]["caller"] == "model"
    assert _refunds(store, "BK-10231") == []


def test_model_cannot_read_another_customers_order():
    graph, _store, _language = _graph(
        {"Where is my package?": {"goal": "order_status", "order_id": "BK-10231"}}, "call_other_order"
    )
    state = _say(graph, "boundary", "Where is my package?", customer_id=MAYA)
    other = [
        call
        for call in state["trace"][-1]["tool_calls"]
        if call["caller"] == "model" and call["args"].get("order_id") == "BK-10239"
    ]
    assert other and other[0]["ok"] is False
    assert other[0]["result"]["error"] == "not found"


def test_model_read_is_executed_and_logged():
    graph, _store, _language = _graph(
        {"Where is my package?": {"goal": "order_status", "order_id": "BK-10231"}}, "call_get_order"
    )
    state = _say(graph, "model-read", "Where is my package?", customer_id=MAYA)
    reads = [
        call
        for call in state["trace"][-1]["tool_calls"]
        if call["name"] == "get_order" and call["caller"] == "model" and call["ok"]
    ]
    assert reads and reads[0]["result"]["order_id"] == "BK-10231"


def test_faq_does_not_ask_for_an_email():
    faq = StaticFaq(grounded=True)
    graph, store, _language = _graph({"What is your shipping policy?": {"goal": "faq"}}, faq=faq)
    state = _say(graph, "faq", "What is your shipping policy?")
    assert state["intent"] == "faq"
    assert state.get("customer_id") is None
    assert state["directive"]["kind"] == "faq_answer"
    assert "2 to 5" in state["directive"]["articles"][0]["body"]
    assert any(
        call["name"] == "search_faq" and call["caller"] == "code" and call["ok"]
        for call in state["trace"][-1]["tool_calls"]
    )
    assert store.tickets == []


def test_low_faq_score_does_not_invent_an_answer():
    graph, store, _language = _graph(
        {"What is your shipping policy?": {"goal": "faq"}}, faq=StaticFaq(grounded=False)
    )
    state = _say(graph, "faq-miss", "What is your shipping policy?")
    assert state["directive"]["kind"] == "faq_unsure"
    assert "articles" not in state["directive"]
    assert state["directive"]["email"] == "support@bookly.example"
    assert store.tickets == []


def test_a_grounded_answer_cites_its_articles():
    graph, _store, _language = _graph({"What is your shipping policy?": {"goal": "faq"}}, faq=StaticFaq(grounded=True))
    state = _say(graph, "cite", "What is your shipping policy?")
    exhibit = state["messages"][-1].additional_kwargs["exhibit"]
    assert exhibit["kind"] == "faq_sources"
    assert [(article["id"], article["title"]) for article in exhibit["articles"]] == [
        ("shipping-times", "Shipping times and cost"),
    ]
    assert "2 to 5" in exhibit["articles"][0]["body"]


def test_a_missed_search_cites_nothing():
    graph, _store, _language = _graph({"What is your shipping policy?": {"goal": "faq"}}, faq=StaticFaq())
    state = _say(graph, "no-cite", "What is your shipping policy?")
    assert "exhibit" not in state["messages"][-1].additional_kwargs


def test_the_models_standalone_query_is_what_gets_searched():
    faq = StaticFaq(grounded=True)
    graph, _store, _language = _graph(
        {"what about ebooks?": {"goal": "faq", "faq_query": "return policy for ebooks"}}, faq=faq
    )
    state = _say(graph, "follow", "what about ebooks?")
    assert faq.queries == ["return policy for ebooks"]
    assert state["trace"][-1]["read"]["faq_query"] == "return policy for ebooks"


def test_a_failed_search_is_not_reported_as_no_article():
    graph, _store, _language = _graph(
        {"What is your shipping policy?": {"goal": "faq"}}, faq=StaticFaq(error="voyage 429")
    )
    state = _say(graph, "fail", "What is your shipping policy?")
    assert state["directive"]["kind"] == "faq_unsure"
    assert state["directive"]["search_failed"] is True
    searched = [call for call in state["trace"][-1]["tool_calls"] if call["name"] == "search_faq"]
    assert searched[0]["ok"] is False
    assert searched[0]["result"]["error"] == "voyage 429"


def test_a_policy_question_about_an_order_answers_then_the_refund_still_goes_through_policy():
    line = "The Night Circus came with water damage, what's your policy on that?"
    graph, store, _language = _graph(
        {
            line: {"goal": "faq", "order_id": "BK-10241", "refund_reason": "water damage"},
            "ok, please refund it": _refund("BK-10241"),
            "yes": YES,
        },
        faq=StaticFaq(grounded=True),
    )
    state = _say(graph, "mixed", line, customer_id=TANNER)
    assert state["directive"]["kind"] == "faq_answer"
    assert state["directive"]["order"]["order_id"] == "BK-10241"
    assert state["selected_order_id"] == "BK-10241"
    assert state.get("refund_decision") is None
    assert _refunds(store, "BK-10241") == []

    state = _say(graph, "mixed", "ok, please refund it", customer_id=TANNER)
    assert state["refund_reason"] == "water damage"
    assert state["refund_decision"]["rule_id"] == "standard_refund"
    assert state["awaiting"] == "refund_confirm"
    assert _refunds(store, "BK-10241") == []

    _say(graph, "mixed", "yes", customer_id=TANNER)
    assert len(_refunds(store, "BK-10241")) == 1


def test_a_policy_question_naming_someone_elses_order_attaches_no_order():
    graph, _store, _language = _graph(
        {"what's your policy on BK-10239?": {"goal": "faq", "order_id": "BK-10239"}},
        faq=StaticFaq(grounded=True),
    )
    state = _say(graph, "faq-foreign", "what's your policy on BK-10239?", customer_id=TANNER)
    assert state["directive"]["kind"] == "faq_answer"
    assert "order" not in state["directive"]


def test_asking_for_a_person_opens_a_ticket():
    graph, store, _language = _graph({"can I talk to a human": {"goal": "handoff"}})
    state = _say(graph, "human", "can I talk to a human", customer_id=MAYA)
    assert state["directive"]["kind"] == "handoff"
    assert state["directive"]["decision"] is None
    assert store.tickets[0]["reason"] == "handoff"


def test_an_order_question_without_sign_in_does_not_ask_in_the_chat():
    graph, store, _language = _graph({"I want a refund": {"goal": "refund"}})
    state = _say(graph, "anon", "I want a refund")
    assert state["directive"]["kind"] == "sign_in_required"
    assert state["awaiting"] is None
    text = state["messages"][-1].content.lower()
    assert "sign in" in text
    assert "?" not in text
    assert store.tickets == []


def test_threads_do_not_share_a_customer():
    graph, _store, _language = _graph({"I want a refund": {"goal": "refund"}})
    _say(graph, "one", "I want a refund", customer_id=MAYA)
    other = _say(graph, "two", "I want a refund")
    assert other.get("customer_id") is None
    assert other["directive"]["kind"] == "sign_in_required"


def _rated(stars):
    """Maya's status question, a thank-you, then a score."""

    script = {
        "where is it?": {"goal": "order_status", "order_id": "BK-10231"},
        "thanks": DONE,
        "score": {"goal": "rate", "stars": stars},
        "The wait was too long.": {"goal": "rate", "review": "The wait was too long."},
    }
    graph, store, _language = _graph(script)
    _say(graph, "rate", "where is it?", customer_id=MAYA)
    state = _say(graph, "rate", "thanks", customer_id=MAYA)
    assert state["intent"] == "resolved"
    assert state["awaiting"] == "rating"
    assert state["directive"]["kind"] == "ask_rating"
    assert state["messages"][-1].additional_kwargs["exhibit"] == {"kind": "ask_rating"}
    assert store.ratings == []
    return graph, store, _say(graph, "rate", "score", customer_id=MAYA)


@pytest.mark.parametrize(
    ("stars", "saved", "awaiting", "closed"),
    [
        (0, False, "rating", False),
        (1, True, "review", False),
        (2, True, "review", False),
        (3, True, None, True),
        (5, True, None, True),
        (6, False, "rating", False),
        (None, False, "rating", False),
    ],
)
def test_rating_boundaries(stars, saved, awaiting, closed):
    _graph_, store, state = _rated(stars)
    assert state["awaiting"] == awaiting
    assert bool(state["directive"].get("closed")) is closed
    if saved:
        assert store.ratings[-1]["stars"] == stars
        assert store.ratings[-1]["thread_id"] == "rate"
        assert store.ratings[-1]["customer_id"] == MAYA
        assert state["messages"][-1].additional_kwargs["exhibit"] == {"kind": "rated", "stars": stars}
        assert bool(state["directive"].get("needs_review")) is (stars <= 2)
    else:
        assert store.ratings == []
        assert state["directive"]["kind"] == "rating_unclear"


def test_a_low_score_saves_the_review():
    graph, store, _state = _rated(1)
    state = _say(graph, "rate", "The wait was too long.", customer_id=MAYA)
    assert state["directive"]["kind"] == "review_saved"
    assert state["directive"]["closed"] is True
    assert store.ratings[-1]["comment"] == "The wait was too long."


def test_a_new_request_during_the_rating_leaves_it():
    graph, store, _state = _rated(6)
    state = _say(graph, "rate", "where is it?", customer_id=MAYA)
    assert state["directive"]["kind"] == "order_status"
    assert state["awaiting"] is None
    assert store.ratings == []


def test_mongo_checkpointer_resumes_the_customer():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        pytest.skip("MONGODB_URI is not set")
    client = MongoClient(uri, serverSelectionTimeoutMS=1000)
    try:
        client.admin.command("ping")
    except PyMongoError:
        client.close()
        pytest.skip("MongoDB is not running")
    thread = "phase2-" + uuid.uuid4().hex
    saver = MongoDBSaver(client, db_name="bookly_test")
    data = build_seed(TODAY)
    store = MemoryStore(data.customers, data.orders, data.refunds)
    script = {"I want a refund": _refund("BK-10231", "damaged"), "yes": YES}
    try:
        graph = build_graph(ScriptedLanguage(script), store, saver, today=TODAY)
        _say(graph, thread, "I want a refund", customer_id=MAYA)
        resumed = build_graph(ScriptedLanguage(script), store, saver, today=TODAY)
        state = _say(resumed, thread, "yes")
        assert state["customer_id"] == MAYA
        assert state["directive"]["kind"] == "refund_issued"
        assert len(_refunds(store, "BK-10231")) == 1
    finally:
        saver.delete_thread(thread)
        client.close()


def test_store_refuses_a_second_write_directly():
    data = build_seed(TODAY)
    store = MemoryStore(data.customers, data.orders, data.refunds)
    store.issue_refund("BK-10231", "standard")
    with pytest.raises(RefundWriteError):
        store.issue_refund("BK-10231", "standard")
    assert len(_refunds(store, "BK-10231")) == 1


def test_memory_order_search_is_scoped_and_filtered():
    data = build_seed(TODAY)
    store = MemoryStore(data.customers, data.orders, data.refunds)
    ids = lambda rows: [row["_id"] for row in rows]  # noqa: E731
    assert ids(store.search_orders(TANNER, title="night")) == ["BK-10241", "BK-10242"]
    assert ids(store.search_orders(TANNER, title="night", sort="oldest", limit=1)) == ["BK-10242"]
    assert ids(store.search_orders(TANNER, author="le guin")) == ["BK-10244", "BK-10243"]
    assert set(ids(store.search_orders(TANNER, status="in_transit"))) == {"BK-10244", "BK-10245"}
    after = (TODAY.replace(day=TODAY.day - 7)).isoformat()
    assert ids(store.search_orders(TANNER, ordered_after=after)) == ["BK-10244", "BK-10245"]
    assert store.search_orders(MAYA, title="dispossessed") == []
