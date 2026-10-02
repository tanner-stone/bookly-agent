"""One turn of the support graph.

The model reads each turn (`understand`) against this customer's orders and
whatever is open. Code checks that read, then runs status, policy, the refund
write, a ticket, or a rating. Refund eligibility, the refund write, tickets,
and ratings are methods on this object, not tools.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_config
from langgraph.types import Command

from agent.matching import owned_order_id, owned_order_ids
from agent.replies import fixed_reply
from agent.session import first_name, greeting
from policy.config import ORDER_CONTEXT_LIMIT, RECENT_ORDERS_LIMIT, TOOL_LOOP_CAP
from policy.refund_policy import RefundPolicy, decide_refund
from tools.faq_search import FaqResult, NullFaq
from tools.read_tools import READ_ONLY_NAMES
from tools.records import directive_for_model, money_str, public_order
from tools.store import RefundWriteError, Store

GOALS = {"order_status", "refund", "faq", "handoff", "resolved", "rate", "chat"}
CONFIRMATIONS = {"yes", "no", "none"}


def route_next(state: dict) -> str:
    if not _texts(state):
        return "greet"
    return "understand"


def after_phrase(state: dict) -> str:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None)
    if calls and state.get("tool_steps", 0) < TOOL_LOOP_CAP:
        return "execute_read_tools"
    return "finish"


class BooklyRuntime:
    def __init__(self, language, store: Store, policy: RefundPolicy, today: date | None, faq=None):
        self.language = language
        self.store = store
        self.policy = policy
        self.fixed_today = today
        self.faq = faq or NullFaq()

    def current_day(self) -> date:
        return self.fixed_today or date.today()

    def route(self, state: dict) -> dict:
        return {
            "channel": state.get("channel") or "chat",
            "turn_tools": [],
            "tool_steps": 0,
            "policy_fresh": False,
            "read": None,
        }

    def greet(self, state: dict) -> Command:
        name = self._first_name(state)
        return Command(
            update={
                "intent": "greet",
                "awaiting": None,
                "directive": {"kind": "greet", "first_name": name},
                "messages": [AIMessage(content=greeting(name))],
            },
            goto="finish",
        )

    def understand(self, state: dict) -> Command:
        customer_id = state.get("customer_id")
        orders = self.store.list_orders(customer_id) if customer_id else []
        tools: list[dict] = []

        def run_read(name: str, args: dict) -> Any:
            if name not in READ_ONLY_NAMES:
                result: Any = {"error": "refused", "reason": "writes and refunds are not model tools"}
                tools.append(_tool(name, "model", False, args, result))
                return result
            result, ok = self._run_read(name, args, customer_id)
            tools.append(_tool(name, "model", ok, args, result))
            return result

        raw = self.language.understand(
            self._view(state, orders),
            state.get("messages") or [],
            run_read,
        )
        read = _checked_read(raw, orders)
        update: dict[str, Any] = {
            "read": read,
            "turn_tools": list(state.get("turn_tools") or []) + tools,
            "proposed_refund": None,
        }
        return self._act(state, read, orders, update)

    def _act(self, state: dict, read: dict, orders: list[dict], update: dict) -> Command:
        goal = read["goal"]
        awaiting = state.get("awaiting")
        proposed = state.get("proposed_refund")
        talking_back = goal in {"rate", "chat", "resolved"}

        if awaiting == "rating" and talking_back:
            stars = read.get("stars")
            if _valid_stars(stars):
                return Command(update={**update, "intent": "resolved"}, goto="save_rating")
            update.update({"awaiting": "rating", "directive": {"kind": "rating_unclear"}})
            return Command(update=update, goto="phrase")

        if awaiting == "review" and talking_back:
            return Command(update=update, goto="save_review")

        if awaiting == "refund_confirm" and proposed:
            same_order = read.get("order_id") in (None, proposed)
            if same_order and read["confirmation"] == "yes" and goal in {"refund", "chat"}:
                update.update({"selected_order_id": proposed, "proposed_refund": proposed})
                return Command(update=update, goto="issue_refund")
            if same_order and read["confirmation"] == "no":
                update.update(
                    {
                        "awaiting": None,
                        "directive": {"kind": "refund_declined", "order_id": proposed},
                    }
                )
                return Command(update=update, goto="phrase")
            if same_order and goal in {"refund", "chat"}:
                update.update(
                    {
                        "proposed_refund": proposed,
                        "awaiting": "refund_confirm",
                        "directive": {
                            "kind": "confirm_unclear",
                            "order": self._public_by_id(orders, proposed),
                        },
                    }
                )
                return Command(update=update, goto="phrase")

        if goal == "resolved":
            update.update(
                {"intent": "resolved", "awaiting": "rating", "directive": {"kind": "ask_rating"}}
            )
            return Command(update=update, goto="phrase")
        if goal == "faq":
            update.update({"intent": "faq", "awaiting": None})
            if read.get("order_id"):
                update.update(
                    {
                        "selected_order_id": read["order_id"],
                        "candidate_order_ids": [read["order_id"]],
                        "refund_reason": read.get("refund_reason"),
                    }
                )
            return Command(update=update, goto="answer_faq")
        if goal == "handoff":
            update.update({"intent": "handoff", "awaiting": None, "refund_decision": None})
            return Command(update=update, goto="create_ticket")
        if goal in {"order_status", "refund"}:
            if not state.get("customer_id"):
                update.update(
                    {"intent": goal, "awaiting": None, "directive": {"kind": "sign_in_required"}}
                )
                return Command(update=update, goto="phrase")
            return self._order_goal(state, read, orders, goal, update)

        keep = awaiting if awaiting in {"order_choice", "refund_reason"} else None
        update.update({"awaiting": keep, "directive": {"kind": "clarify"}})
        return Command(update=update, goto="phrase")

    def _order_goal(self, state: dict, read: dict, orders: list[dict], goal: str, update: dict) -> Command:
        update["intent"] = goal
        carried = None
        if state.get("intent") == "refund" and state.get("awaiting") in {"order_choice", "refund_reason"}:
            carried = state.get("refund_reason")
        order_id = read.get("order_id")
        candidates = read["candidate_ids"]
        picking = state.get("awaiting") == "order_choice"
        if order_id and candidates and not picking:
            # The model named one order but said others fit too. The customer chooses.
            candidates = [order_id, *candidates]
            order_id = None
        if order_id is None and len(candidates) == 1 and picking:
            order_id = candidates[0]
        if order_id is None:
            if candidates:
                shown = [order for order in orders if order["_id"] in candidates]
            else:
                shown = orders[:RECENT_ORDERS_LIMIT]
            if not shown:
                update.update(
                    {"awaiting": None, "directive": {"kind": "no_orders", **self._contacts()}}
                )
                return Command(update=update, goto="phrase")
            update.update(
                {
                    "awaiting": "order_choice",
                    "candidate_order_ids": [order["_id"] for order in shown],
                    "refund_reason": (read.get("refund_reason") or carried) if goal == "refund" else None,
                    "directive": {
                        "kind": "confirm_order" if len(shown) == 1 else "choose_order",
                        "orders": [public_order(order, self.current_day()) for order in shown],
                    },
                }
            )
            return Command(update=update, goto="phrase")

        if state.get("intent") == "faq" and state.get("selected_order_id") == order_id:
            carried = carried or state.get("refund_reason")
        update.update(
            {"selected_order_id": order_id, "candidate_order_ids": [order_id], "awaiting": None}
        )
        if goal == "order_status":
            return Command(update=update, goto="fetch_order")
        reason = read.get("refund_reason")
        if not reason and state.get("awaiting") == "refund_reason" and state.get("selected_order_id") == order_id:
            reason = _latest(state)
        reason = reason or carried
        if reason:
            update["refund_reason"] = reason
            return Command(update=update, goto="apply_policy")
        update.update(
            {
                "awaiting": "refund_reason",
                "refund_reason": None,
                "refund_decision": None,
                "directive": {"kind": "ask_reason", "order": self._public_by_id(orders, order_id)},
            }
        )
        return Command(update=update, goto="phrase")

    def fetch_order(self, state: dict) -> Command:
        order_id = state.get("selected_order_id")
        customer_id = state.get("customer_id") or ""
        order = self.store.get_order(order_id, customer_id) if order_id else None
        if order is None:
            tool = _tool("get_order", "code", False, {"order_id": order_id}, {"error": "not found"})
            directive = {"kind": "order_missing", **self._contacts()}
        else:
            viewed = public_order(order, self.current_day())
            tool = _tool("get_order", "code", True, {"order_id": order_id}, viewed)
            directive = {"kind": "order_status", "order": viewed}
        return Command(
            update={
                "turn_tools": list(state.get("turn_tools") or []) + [tool],
                "directive": directive,
                "awaiting": None,
            },
            goto="phrase",
        )

    def apply_policy(self, state: dict) -> Command:
        order_id = state.get("selected_order_id")
        customer_id = state.get("customer_id") or ""
        order = self.store.get_order(order_id, customer_id) if order_id else None
        customer = self.store.get_customer(customer_id) if customer_id else None
        if order is None or customer is None:
            return Command(
                update={"directive": {"kind": "order_missing", **self._contacts()}, "awaiting": None},
                goto="phrase",
            )
        decision = decide_refund(order, customer, self.current_day(), self.policy)
        viewed = public_order(order, self.current_day())
        packet = _decision_dict(decision)
        tool = _tool("get_order", "code", True, {"order_id": order["_id"]}, viewed)
        offered = decision.outcome == "AUTO_REFUND"
        update: dict[str, Any] = {
            "refund_decision": packet,
            "policy_fresh": True,
            "turn_tools": list(state.get("turn_tools") or []) + [tool],
            "proposed_refund": order["_id"] if offered else None,
            "directive": {
                "kind": "refund_decision",
                "decision": packet,
                "order": viewed,
                "reason": state.get("refund_reason"),
                "ask_confirm": offered,
                **self._contacts(),
            },
        }
        if decision.outcome == "ESCALATE":
            return Command(update=update, goto="create_ticket")
        update["awaiting"] = "refund_confirm" if offered else None
        return Command(update=update, goto="phrase")

    def issue_refund(self, state: dict) -> Command:
        order_id = state.get("selected_order_id")
        read = state.get("read") or {}
        # The model's yes is only an answer to the refund offered last turn, on that order.
        if not order_id or state.get("proposed_refund") != order_id or read.get("confirmation") != "yes":
            return Command(
                update={
                    "awaiting": None,
                    "proposed_refund": None,
                    "directive": {"kind": "clarify"},
                },
                goto="phrase",
            )
        customer_id = state.get("customer_id") or ""
        order = self.store.get_order(order_id, customer_id)
        customer = self.store.get_customer(customer_id) if customer_id else None
        if order is None or customer is None:
            return Command(
                update={
                    "awaiting": None,
                    "proposed_refund": None,
                    "directive": {"kind": "order_missing", **self._contacts()},
                },
                goto="phrase",
            )
        # Decide again. The offer last turn is not permission to pay.
        decision = decide_refund(order, customer, self.current_day(), self.policy)
        packet = _decision_dict(decision)
        viewed = public_order(order, self.current_day())
        tools = list(state.get("turn_tools") or [])
        tools.append(_tool("get_order", "code", True, {"order_id": order["_id"]}, viewed))
        if decision.outcome != "AUTO_REFUND" or not decision.tier:
            return Command(
                update={
                    "refund_decision": packet,
                    "policy_fresh": True,
                    "awaiting": None,
                    "proposed_refund": None,
                    "turn_tools": tools,
                    "directive": {
                        "kind": "refund_decision",
                        "decision": packet,
                        "order": viewed,
                        "ask_confirm": False,
                        **self._contacts(),
                    },
                },
                goto="phrase",
            )
        try:
            refund = self.store.issue_refund(order["_id"], decision.tier)
        except RefundWriteError as exc:
            tools.append(
                _tool("issue_refund", "code", False, {"order_id": order["_id"]}, {"error": str(exc)})
            )
            return Command(
                update={
                    "awaiting": None,
                    "proposed_refund": None,
                    "policy_fresh": True,
                    "refund_decision": packet,
                    "turn_tools": tools,
                    "directive": {"kind": "refund_blocked", "error": str(exc), **self._contacts()},
                },
                goto="phrase",
            )
        tools.append(
            _tool(
                "issue_refund",
                "code",
                True,
                {"order_id": order["_id"], "tier": decision.tier},
                {"amount": money_str(refund["amount"])},
            )
        )
        return Command(
            update={
                "awaiting": None,
                "proposed_refund": None,
                "refund_decision": packet,
                "policy_fresh": True,
                "turn_tools": tools,
                "directive": {
                    "kind": "refund_issued",
                    "decision": packet,
                    "amount": money_str(refund["amount"]),
                    "order_id": order["_id"],
                    **self._contacts(),
                },
            },
            goto="phrase",
        )

    def create_ticket(self, state: dict) -> Command:
        decision = state.get("refund_decision") or {}
        reason = decision.get("rule_id") or "handoff"
        summary = (
            f"intent={state.get('intent')}; order={state.get('selected_order_id')}; "
            f"rule={reason}; customer_said={state.get('refund_reason') or _latest(state)}"
        )
        tools = list(state.get("turn_tools") or [])
        ticket_id = None
        customer_id = state.get("customer_id")
        if customer_id:
            ticket = self.store.create_ticket(
                customer_id, state.get("selected_order_id"), reason, summary
            )
            ticket_id = str(ticket.get("_id") or ticket.get("id"))
            tools.append(_tool("create_ticket", "code", True, {"reason": reason}, {"ticket_id": ticket_id}))
        return Command(
            update={
                "awaiting": None,
                "turn_tools": tools,
                "policy_fresh": bool(decision),
                "directive": self._handoff_directive(reason, ticket_id, decision or None),
            },
            goto="phrase",
        )

    def save_rating(self, state: dict) -> Command:
        stars = (state.get("read") or {}).get("stars")
        if not _valid_stars(stars):
            return Command(
                update={"awaiting": "rating", "directive": {"kind": "rating_unclear"}},
                goto="phrase",
            )
        self.store.save_rating(_thread_id(), state.get("customer_id"), stars)
        directive: dict[str, Any] = {"kind": "rating_saved", "stars": stars}
        if stars <= 2:
            directive["needs_review"] = True
            awaiting = "review"
        else:
            directive["closed"] = True
            awaiting = None
        return Command(update={"awaiting": awaiting, "directive": directive}, goto="phrase")

    def save_review(self, state: dict) -> Command:
        comment = ((state.get("read") or {}).get("review") or _latest(state)).strip()
        self.store.save_review(_thread_id(), comment)
        stars = (state.get("directive") or {}).get("stars")
        return Command(
            update={
                "awaiting": None,
                "directive": {
                    "kind": "review_saved",
                    "stars": stars,
                    "comment": comment,
                    "closed": True,
                },
            },
            goto="phrase",
        )

    def answer_faq(self, state: dict) -> Command:
        read = state.get("read") or {}
        query = read.get("faq_query") or _latest(state)
        result = self.faq.search(query)
        tool = _tool("search_faq", "code", result.error is None, {"query": query}, _faq_trace(result))
        if result.grounded:
            directive: dict[str, Any] = {
                "kind": "faq_answer",
                "articles": [_article(hit) for hit in result.hits],
                **self._contacts(),
            }
            order_id = read.get("order_id")
            customer_id = state.get("customer_id")
            order = self.store.get_order(order_id, customer_id) if order_id and customer_id else None
            if order is not None:
                directive["order"] = public_order(order, self.current_day())
        else:
            directive = {
                "kind": "faq_unsure",
                "search_failed": result.error is not None,
                **self._contacts(),
            }
        return Command(
            update={
                "awaiting": None,
                "directive": directive,
                "turn_tools": list(state.get("turn_tools") or []) + [tool],
            },
            goto="phrase",
        )

    def phrase(self, state: dict) -> Command | dict:
        directive = dict(state.get("directive") or {"kind": "clarify"})
        name = self._first_name(state)
        if name:
            directive["first_name"] = name
        # An order-status sentence is blocked until this turn has a real read.
        if directive.get("kind") == "order_status" and not _has_order_read(state):
            return Command(goto="fetch_order")
        text = fixed_reply(directive, state.get("channel") or "chat")
        if text is not None:
            return {"messages": [_stamp(AIMessage(content=text), directive)]}
        reply_state = dict(state)
        reply_state["directive"] = directive_for_model(directive)
        return {"messages": [_stamp(self.language.reply(reply_state), directive)]}

    def execute_read_tools(self, state: dict) -> dict:
        last = state["messages"][-1]
        tools = list(state.get("turn_tools") or [])
        messages = []
        for call in last.tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            if name not in READ_ONLY_NAMES:
                result: Any = {"error": "refused", "reason": "writes and refunds are not model tools"}
                ok = False
            else:
                result, ok = self._run_read(name, args, state.get("customer_id"))
            tools.append(_tool(name, "model", ok, args, result))
            messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
            )
        return {
            "messages": messages,
            "turn_tools": tools,
            "tool_steps": state.get("tool_steps", 0) + 1,
        }

    def finish(self, state: dict) -> dict:
        policy = state.get("refund_decision") if state.get("policy_fresh") else None
        read = state.get("read")
        record = {
            "intent": state.get("intent"),
            "read": _trace_read(read) if read else None,
            "awaiting": state.get("awaiting"),
            "tool_calls": state.get("turn_tools") or [],
            "policy": policy,
        }
        updates: dict[str, Any] = {"trace": list(state.get("trace") or []) + [record]}
        last = state["messages"][-1]
        content = getattr(last, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if not content.strip():
            updates["messages"] = [
                AIMessage(
                    content=(
                        "I could not finish that lookup. "
                        f"You can reach us at {self.policy.support_email} "
                        f"or {self.policy.support_phone}."
                    )
                )
            ]
        return updates

    def _view(self, state: dict, orders: list[dict]) -> dict:
        today = self.current_day()
        awaiting = state.get("awaiting")
        listed = [_order_for_view(public_order(order, today)) for order in orders[:ORDER_CONTEXT_LIMIT]]
        by_id = {order["order_id"]: order for order in listed}
        cards = state.get("candidate_order_ids") if awaiting == "order_choice" else []
        return {
            "today": today.isoformat(),
            "channel": state.get("channel") or "chat",
            "signed_in": bool(state.get("customer_id")),
            "orders": listed,
            "more_orders_than_listed": len(orders) > ORDER_CONTEXT_LIMIT,
            "open": {
                "goal": state.get("intent"),
                "waiting_on": awaiting,
                "cards_on_screen": [
                    {"position": index, **by_id[order_id]}
                    for index, order_id in enumerate(cards or [], start=1)
                    if order_id in by_id
                ],
                "order_in_discussion": state.get("selected_order_id"),
                "refund_waiting_on_yes": state.get("proposed_refund") if awaiting == "refund_confirm" else None,
                "refund_reason_so_far": state.get("refund_reason"),
            },
        }

    def _public_by_id(self, orders: list[dict], order_id: str) -> dict | None:
        for order in orders:
            if order["_id"] == order_id:
                return public_order(order, self.current_day())
        return None

    def _first_name(self, state: dict) -> str | None:
        customer_id = state.get("customer_id")
        if not customer_id:
            return None
        customer = self.store.get_customer(customer_id)
        if not customer:
            return None
        return first_name(customer.get("name"))

    def _contacts(self) -> dict[str, str]:
        return {"email": self.policy.support_email, "phone": self.policy.support_phone}

    def _handoff_directive(self, reason: str, ticket_id: str | None, decision: dict | None) -> dict:
        return {
            "kind": "handoff",
            "reason": reason,
            "ticket_id": ticket_id,
            "decision": decision,
            **self._contacts(),
        }

    def _run_read(self, name: str, args: dict, customer_id: str | None) -> tuple[Any, bool]:
        if name == "search_faq":
            result = self.faq.search(str(args.get("query") or ""))
            payload = _faq_trace(result)
            if result.grounded:
                payload["articles"] = [_article(hit) for hit in result.hits]
            return payload, result.error is None
        if not customer_id:
            return {"error": "no customer yet"}, False
        if name == "get_order":
            order = self.store.get_order(str(args.get("order_id") or ""), customer_id)
            if order is None:
                return {"error": "not found"}, False
            return public_order(order, self.current_day()), True
        if name == "search_orders":
            found = self.store.search_orders(customer_id, **_search_filters(args))
            return [_order_for_view(public_order(order, self.current_day())) for order in found], True
        return {"error": "unknown tool"}, False


def _checked_read(raw: Any, orders: list[dict]) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    goal = raw.get("goal") if raw.get("goal") in GOALS else "chat"
    confirmation = raw.get("confirmation") if raw.get("confirmation") in CONFIRMATIONS else "none"
    order_id = owned_order_id(raw.get("order_id"), orders)
    candidates = owned_order_ids(raw.get("candidate_ids"), orders)
    named = [raw.get("order_id"), *(raw.get("candidate_ids") or [])]
    dropped = [value for value in named if isinstance(value, str) and value.strip() and owned_order_id(value, orders) is None]
    stars = raw.get("stars")
    if isinstance(stars, bool) or not isinstance(stars, int):
        stars = None
    return {
        "goal": goal,
        "order_id": order_id,
        "candidate_ids": [value for value in candidates if value != order_id],
        "refund_reason": _text_or_none(raw.get("refund_reason")),
        "faq_query": _text_or_none(raw.get("faq_query")),
        "confirmation": confirmation,
        "stars": stars,
        "review": _text_or_none(raw.get("review")),
        "dropped_ids": dropped,
    }


def _valid_stars(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _trace_read(read: dict) -> dict:
    return {
        key: read.get(key)
        for key in (
            "goal",
            "order_id",
            "candidate_ids",
            "refund_reason",
            "faq_query",
            "confirmation",
            "stars",
            "dropped_ids",
        )
        if read.get(key) not in (None, [], "none")
    }


def _order_for_view(viewed: dict) -> dict:
    keys = (
        "order_id",
        "title",
        "author",
        "status",
        "total",
        "ordered_at",
        "ordered_label",
        "delivered_at",
        "delivered_label",
        "eta",
    )
    return {key: viewed.get(key) for key in keys if viewed.get(key) is not None}


def _search_filters(args: dict) -> dict:
    filters: dict[str, Any] = {}
    for key in ("status", "title", "author", "ordered_after", "ordered_before"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            filters[key] = value.strip()
    sort = args.get("sort")
    filters["sort"] = sort if sort in {"newest", "oldest"} else "newest"
    limit = args.get("limit")
    filters["limit"] = limit if isinstance(limit, int) and 1 <= limit <= ORDER_CONTEXT_LIMIT else 10
    return filters


def _thread_id() -> str:
    return str((get_config().get("configurable") or {}).get("thread_id") or "")


def _orders_by_id(orders: list[dict], order_ids: list[str]) -> list[dict]:
    by_id = {order["_id"]: order for order in orders}
    return [by_id[order_id] for order_id in order_ids if order_id in by_id]


def _faq_trace(result: FaqResult) -> dict:
    payload: dict[str, Any] = {"grounded": result.grounded, "best_score": result.best_score}
    if result.grounded:
        payload["titles"] = [hit.title for hit in result.hits]
    if result.error:
        payload["error"] = result.error
    return payload


def _article(hit) -> dict:
    return {"id": hit.article_id, "title": hit.title, "body": hit.body}


def _tool(name: str, caller: str, ok: bool, args: dict, result: Any) -> dict:
    return {"name": name, "caller": caller, "ok": ok, "args": args, "result": result}


def _decision_dict(decision) -> dict:
    return {
        "outcome": decision.outcome,
        "tier": decision.tier,
        "reason_code": decision.reason_code,
        "rule_id": decision.rule_id,
        "customer_note": decision.customer_note,
    }


def _texts(state: dict) -> list[str]:
    found = []
    for message in state.get("messages") or []:
        if getattr(message, "type", None) == "human":
            content = message.content
            found.append(content if isinstance(content, str) else str(content))
    return found


def _exhibit(directive: dict) -> dict | None:
    kind = directive.get("kind")
    if kind in {"choose_order", "confirm_order"} and directive.get("orders"):
        return {"kind": kind, "orders": directive["orders"]}
    if kind == "order_status":
        order = directive.get("order") or {}
        tracking = order.get("tracking") or {}
        if tracking.get("points"):
            return {"kind": kind, "order": order}
    if kind == "faq_answer" and directive.get("articles"):
        return {"kind": "faq_sources", "articles": directive["articles"]}
    if kind in {"ask_rating", "rating_unclear"}:
        return {"kind": "ask_rating"}
    if kind == "rating_saved":
        return {"kind": "rated", "stars": directive.get("stars")}
    return None


def _stamp(message: AIMessage, directive: dict) -> AIMessage:
    if getattr(message, "tool_calls", None):
        return message
    content = message.content if isinstance(message.content, str) else ""
    if not content.strip():
        return message
    exhibit = _exhibit(directive)
    if exhibit is None:
        return message
    return AIMessage(
        content=content,
        additional_kwargs={**(message.additional_kwargs or {}), "exhibit": exhibit},
    )


def _latest(state: dict) -> str:
    texts = _texts(state)
    return texts[-1] if texts else ""


def _has_order_read(state: dict) -> bool:
    return any(
        call.get("name") == "get_order" and call.get("ok") for call in state.get("turn_tools") or []
    )
