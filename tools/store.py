"""Order, refund, and ticket access.

`issue_refund` is not a model tool. Graph nodes call it after a separate
code check that the policy still says AUTO_REFUND. The three writes move
together so a refund row cannot exist without the status and the prior-refund
count changing too.
"""

from __future__ import annotations

import copy
import re
from datetime import date, datetime
from typing import Any, Literal, Protocol

from pymongo.database import Database
from pymongo.errors import PyMongoError

from tools.records import as_date

# Atlas Search index on order titles and authors, created by data/seed.py.
ORDER_SEARCH_INDEX = "order_text"

Sort = Literal["newest", "oldest"]


class RefundWriteError(Exception):
    """A refund write was refused. No document was changed."""


class Store(Protocol):
    def get_customer_by_email(self, email: str) -> dict[str, Any] | None: ...

    def get_customer(self, customer_id: str) -> dict[str, Any] | None: ...

    def list_orders(self, customer_id: str) -> list[dict[str, Any]]: ...

    def get_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None: ...

    def search_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        author: str | None = None,
        ordered_after: str | None = None,
        ordered_before: str | None = None,
        sort: Sort = "newest",
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...

    def issue_refund(self, order_id: str, tier: str) -> dict[str, Any]: ...

    def create_ticket(
        self,
        customer_id: str,
        order_id: str | None,
        reason: str,
        summary: str,
    ) -> dict[str, Any]: ...

    def save_rating(
        self,
        thread_id: str,
        customer_id: str | None,
        stars: int,
    ) -> dict[str, Any]: ...

    def save_review(self, thread_id: str, comment: str) -> None: ...


def _ordered_on(order: dict[str, Any]):
    return as_date(order.get("ordered_at")) or datetime.min.date()


def _iso_day(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _within_dates(order: dict[str, Any], after: date | None, before: date | None) -> bool:
    ordered = _ordered_on(order)
    if after and ordered < after:
        return False
    if before and ordered > before:
        return False
    return True


def _sorted(rows: list[dict[str, Any]], sort: Sort, limit: int) -> list[dict[str, Any]]:
    rows.sort(key=_ordered_on, reverse=sort != "oldest")
    return rows[: max(1, limit)]


def _has_item(order: dict[str, Any], field: str, needle: str | None) -> bool:
    if not needle:
        return True
    needle = needle.strip().lower()
    return any(needle in str(item.get(field) or "").lower() for item in order.get("items") or [])


class MemoryStore:
    """In-memory stand-in used by tests. Same write rules as MongoStore."""

    def __init__(self, customers, orders, refunds):
        self.customers = {row["_id"]: copy.deepcopy(row) for row in customers}
        self.orders = {row["_id"]: copy.deepcopy(row) for row in orders}
        self.refunds = [copy.deepcopy(row) for row in refunds]
        self.tickets: list[dict[str, Any]] = []
        self.ratings: list[dict[str, Any]] = []

    def get_customer_by_email(self, email: str) -> dict[str, Any] | None:
        needle = email.strip().lower()
        for customer in self.customers.values():
            if customer["email"].lower() == needle:
                return copy.deepcopy(customer)
        return None

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        customer = self.customers.get(customer_id)
        return copy.deepcopy(customer) if customer else None

    def list_orders(self, customer_id: str) -> list[dict[str, Any]]:
        rows = [
            copy.deepcopy(order)
            for order in self.orders.values()
            if order["customer_id"] == customer_id
        ]
        rows.sort(key=_ordered_on, reverse=True)
        return rows

    def get_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None:
        order = self.orders.get(order_id)
        if order is None or order["customer_id"] != customer_id:
            return None
        return copy.deepcopy(order)

    def search_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        author: str | None = None,
        ordered_after: str | None = None,
        ordered_before: str | None = None,
        sort: Sort = "newest",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        after, before = _iso_day(ordered_after), _iso_day(ordered_before)
        rows = [
            order
            for order in self.list_orders(customer_id)
            if (not status or order["status"] == status)
            and _has_item(order, "title", title)
            and _has_item(order, "author", author)
            and _within_dates(order, after, before)
        ]
        return _sorted(rows, sort, limit)

    def issue_refund(self, order_id: str, tier: str) -> dict[str, Any]:
        order = self.orders.get(order_id)
        if order is None:
            raise RefundWriteError(f"No order {order_id}")
        if order["status"] == "refunded":
            raise RefundWriteError(f"{order_id} is already refunded")
        order["status"] = "refunded"
        self.customers[order["customer_id"]]["prior_refund_count"] += 1
        refund = {
            "order_id": order_id,
            "customer_id": order["customer_id"],
            "amount": order["total"],
            "tier": tier,
            "created_at": datetime.now(),
        }
        self.refunds.append(refund)
        return copy.deepcopy(refund)

    def create_ticket(
        self,
        customer_id: str,
        order_id: str | None,
        reason: str,
        summary: str,
    ) -> dict[str, Any]:
        ticket = {
            "customer_id": customer_id,
            "order_id": order_id,
            "reason": reason,
            "summary": summary,
            "created_at": datetime.now(),
        }
        self.tickets.append(ticket)
        return copy.deepcopy(ticket)

    def save_rating(
        self,
        thread_id: str,
        customer_id: str | None,
        stars: int,
    ) -> dict[str, Any]:
        rating = {
            "thread_id": thread_id,
            "customer_id": customer_id,
            "stars": stars,
            "created_at": datetime.now(),
        }
        self.ratings.append(rating)
        return copy.deepcopy(rating)

    def save_review(self, thread_id: str, comment: str) -> None:
        for rating in reversed(self.ratings):
            if rating["thread_id"] == thread_id:
                rating["comment"] = comment
                return


class MongoStore:
    def __init__(self, database: Database):
        self.database = database

    def get_customer_by_email(self, email: str) -> dict[str, Any] | None:
        return self.database.customers.find_one({"email": email.strip().lower()})

    def get_customer(self, customer_id: str) -> dict[str, Any] | None:
        return self.database.customers.find_one({"_id": customer_id})

    def list_orders(self, customer_id: str) -> list[dict[str, Any]]:
        rows = list(self.database.orders.find({"customer_id": customer_id}))
        rows.sort(key=_ordered_on, reverse=True)
        return rows

    def get_order(self, order_id: str, customer_id: str) -> dict[str, Any] | None:
        return self.database.orders.find_one({"_id": order_id, "customer_id": customer_id})

    def search_orders(
        self,
        customer_id: str,
        *,
        status: str | None = None,
        title: str | None = None,
        author: str | None = None,
        ordered_after: str | None = None,
        ordered_before: str | None = None,
        sort: Sort = "newest",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        # The customer id comes from the session, never from the model's arguments.
        match: dict[str, Any] = {"customer_id": customer_id}
        if status:
            match["status"] = status
        after, before = _iso_day(ordered_after), _iso_day(ordered_before)
        if after or before:
            match["ordered_at"] = {}
            if after:
                match["ordered_at"]["$gte"] = datetime(after.year, after.month, after.day)
            if before:
                match["ordered_at"]["$lte"] = datetime(before.year, before.month, before.day, 23, 59, 59)
        text = [(path, value) for path, value in (("items.title", title), ("items.author", author)) if value]
        if not text:
            rows = list(self.database.orders.find(match))
            return _sorted(rows, sort, limit)
        try:
            rows = list(
                self.database.orders.aggregate(
                    [
                        {
                            "$search": {
                                "index": ORDER_SEARCH_INDEX,
                                "compound": {
                                    "must": [
                                        {"text": {"query": value, "path": path, "fuzzy": {"maxEdits": 1}}}
                                        for path, value in text
                                    ]
                                },
                            }
                        },
                        {"$match": match},
                    ]
                )
            )
        except PyMongoError:
            # No search index (an unseeded database): plain substring match.
            for path, value in text:
                match[path] = {"$regex": re.escape(value), "$options": "i"}
            rows = list(self.database.orders.find(match))
        return _sorted(rows, sort, limit)

    def issue_refund(self, order_id: str, tier: str) -> dict[str, Any]:
        # One transaction: a crash between these writes would either refund
        # twice later or forget that the customer had already been refunded.
        with self.database.client.start_session() as session:
            with session.start_transaction():
                order = self.database.orders.find_one({"_id": order_id}, session=session)
                if order is None:
                    raise RefundWriteError(f"No order {order_id}")
                if order["status"] == "refunded":
                    raise RefundWriteError(f"{order_id} is already refunded")
                self.database.orders.update_one(
                    {"_id": order_id},
                    {"$set": {"status": "refunded"}},
                    session=session,
                )
                self.database.customers.update_one(
                    {"_id": order["customer_id"]},
                    {"$inc": {"prior_refund_count": 1}},
                    session=session,
                )
                refund = {
                    "order_id": order_id,
                    "customer_id": order["customer_id"],
                    "amount": order["total"],
                    "tier": tier,
                    "created_at": datetime.now(),
                }
                self.database.refunds.insert_one(refund, session=session)
                return refund

    def create_ticket(
        self,
        customer_id: str,
        order_id: str | None,
        reason: str,
        summary: str,
    ) -> dict[str, Any]:
        ticket = {
            "customer_id": customer_id,
            "order_id": order_id,
            "reason": reason,
            "summary": summary,
            "created_at": datetime.now(),
        }
        self.database.tickets.insert_one(ticket)
        return ticket

    def save_rating(
        self,
        thread_id: str,
        customer_id: str | None,
        stars: int,
    ) -> dict[str, Any]:
        rating = {
            "thread_id": thread_id,
            "customer_id": customer_id,
            "stars": stars,
            "created_at": datetime.now(),
        }
        self.database.ratings.insert_one(rating)
        return rating

    def save_review(self, thread_id: str, comment: str) -> None:
        found = self.database.ratings.find_one(
            {"thread_id": thread_id},
            sort=[("created_at", -1)],
        )
        if found is None:
            return
        self.database.ratings.update_one({"_id": found["_id"]}, {"$set": {"comment": comment}})
