"""Seed the bookly database with one customer per refund path.

Dates are computed from the day the script runs, so a demo next month still
lands on the same side of each policy limit. The engine reads
prior_refund_count on the customer. Sam's earlier refund is not an order in
this dataset; the count is what the rule sees. Casey's order status is
refunded, which is what the already_refunded rule matches. The refunds row
is the record of that write.

Run: python3 -m data.seed
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from bson.decimal128 import Decimal128
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from pymongo.operations import SearchIndexModel

from policy.refund_policy import RefundDecision, decide_refund, load_refund_policy
from tools.store import ORDER_SEARCH_INDEX

from data.faq_seed import seed_faq


@dataclass(frozen=True)
class SeedData:
    customers: tuple[dict[str, Any], ...]
    orders: tuple[dict[str, Any], ...]
    refunds: tuple[dict[str, Any], ...]
    expected_rule_id: dict[str, str]


def build_seed(today: date) -> SeedData:
    customers = (
        _customer("maya-chen", "Maya Chen", "maya.chen@bookly.example", 0),
        _customer("jordan-lee", "Jordan Lee", "jordan.lee@bookly.example", 0),
        _customer("sam-patel", "Sam Patel", "sam.patel@bookly.example", 1),
        _customer("riley-nguyen", "Riley Nguyen", "riley.nguyen@bookly.example", 0),
        _customer("alex-rivera", "Alex Rivera", "alex.rivera@bookly.example", 0),
        _customer("casey-brooks", "Casey Brooks", "casey.brooks@bookly.example", 1),
        _customer("taylor-kim", "Taylor Kim", "taylor.kim@bookly.example", 0),
        _customer("quinn-alvarez", "Quinn Alvarez", "quinn.alvarez@bookly.example", 0),
        _customer("tanner-stone", "Tanner Stone", "tanner.stone@bookly.example", 0),
    )
    orders = (
        _order(
            "BK-10231",
            "maya-chen",
            [_item("The Midnight Library", "Matt Haig", "32.00")],
            "delivered",
            today,
            ordered_days=14,
            shipped_days=12,
            delivered_days=10,
            carrier="USPS",
            tracking_number="9400111899223344556677",
        ),
        _order(
            "BK-10232",
            "jordan-lee",
            [_item("Circe", "Madeline Miller", "120.00")],
            "delivered",
            today,
            ordered_days=48,
            shipped_days=44,
            delivered_days=40,
            carrier="UPS",
            tracking_number="1Z999AA10123456701",
        ),
        _order(
            "BK-10233",
            "sam-patel",
            [_item("Project Hail Mary", "Andy Weir", "60.00")],
            "delivered",
            today,
            ordered_days=26,
            shipped_days=23,
            delivered_days=20,
            carrier="FedEx",
            tracking_number="748927489274892",
        ),
        _order(
            "BK-10234",
            "riley-nguyen",
            [_item("Klara and the Sun", "Kazuo Ishiguro", "24.00")],
            "in_transit",
            today,
            ordered_days=5,
            shipped_days=2,
            eta_in_days=3,
            carrier="UPS",
            tracking_number="1Z999AA10123456784",
        ),
        _order(
            "BK-10235",
            "alex-rivera",
            [_item("The Left Hand of Darkness", "Ursula K. Le Guin", "18.00")],
            "delivered",
            today,
            ordered_days=8,
            shipped_days=6,
            delivered_days=4,
            carrier="USPS",
            tracking_number="9400111899223344556601",
        ),
        _order(
            "BK-10236",
            "alex-rivera",
            [_item("Piranesi", "Susanna Clarke", "27.00")],
            "delivered",
            today,
            ordered_days=16,
            shipped_days=14,
            delivered_days=12,
            carrier="USPS",
            tracking_number="9400111899223344556602",
        ),
        _order(
            "BK-10237",
            "alex-rivera",
            [_item("A Wizard of Earthsea", "Ursula K. Le Guin", "16.00")],
            "delivered",
            today,
            ordered_days=25,
            shipped_days=23,
            delivered_days=21,
            carrier="UPS",
            tracking_number="1Z999AA10123456737",
        ),
        _order(
            "BK-10238",
            "casey-brooks",
            [_item("Beloved", "Toni Morrison", "22.00")],
            "refunded",
            today,
            ordered_days=20,
            shipped_days=17,
            delivered_days=15,
            carrier="USPS",
            tracking_number="9400111899223344556638",
        ),
        _order(
            "BK-10239",
            "taylor-kim",
            [_item("The Complete Calvin and Hobbes", "Bill Watterson", "200.00")],
            "delivered",
            today,
            ordered_days=9,
            shipped_days=7,
            delivered_days=5,
            carrier="FedEx",
            tracking_number="748900000000039",
        ),
        _order(
            "BK-10240",
            "quinn-alvarez",
            [_item("Tomorrow, and Tomorrow, and Tomorrow", "Gabrielle Zevin", "28.00")],
            "cancelled",
            today,
            ordered_days=7,
        ),
        _order(
            "BK-10241",
            "tanner-stone",
            [_item("The Night Circus", "Erin Morgenstern", "26.00")],
            "delivered",
            today,
            ordered_days=8,
            shipped_days=6,
            delivered_days=4,
            carrier="USPS",
            tracking_number="9400111899223344556641",
        ),
        _order(
            "BK-10242",
            "tanner-stone",
            [_item("The Night Watchman", "Louise Erdrich", "20.00")],
            "delivered",
            today,
            ordered_days=22,
            shipped_days=20,
            delivered_days=18,
            carrier="UPS",
            tracking_number="1Z999AA10123456742",
        ),
        _order(
            "BK-10243",
            "tanner-stone",
            [_item("The Left Hand of Darkness", "Ursula K. Le Guin", "18.00")],
            "delivered",
            today,
            ordered_days=12,
            shipped_days=10,
            delivered_days=9,
            carrier="USPS",
            tracking_number="9400111899223344556643",
        ),
        _order(
            "BK-10244",
            "tanner-stone",
            [_item("The Dispossessed", "Ursula K. Le Guin", "19.00")],
            "in_transit",
            today,
            ordered_days=3,
            shipped_days=1,
            eta_in_days=4,
            carrier="UPS",
            tracking_number="1Z999AA10123456744",
            checkpoints=(
                {"place": "Bookly, Portland", "lat": 45.515, "lng": -122.678},
                {"place": "Denver", "lat": 39.739, "lng": -104.990},
            ),
        ),
        _order(
            "BK-10245",
            "tanner-stone",
            [_item("The Overstory", "Richard Powers", "18.00")],
            "in_transit",
            today,
            ordered_days=6,
            shipped_days=2,
            eta_in_days=5,
            carrier="UPS",
            tracking_number="1Z999AA10123456745",
            checkpoints=(
                {"place": "Bookly, Portland", "lat": 45.515, "lng": -122.678},
                {"place": "Chicago", "lat": 41.878, "lng": -87.630},
            ),
        ),
        _order(
            "BK-10246",
            "tanner-stone",
            [_item("The Goldfinch", "Donna Tartt", "36.00")],
            "delivered",
            today,
            ordered_days=70,
            shipped_days=65,
            delivered_days=60,
            carrier="USPS",
            tracking_number="9400111899223344556646",
        ),
    )
    refunds = (
        {
            "order_id": "BK-10238",
            "customer_id": "casey-brooks",
            "amount": Decimal("22.00"),
            "tier": "standard",
            "created_at": _days_ago(today, 2),
        },
    )
    expected = {
        "BK-10231": "standard_refund",
        "BK-10232": "first_time_goodwill",
        "BK-10233": "escalate",
        "BK-10234": "not_delivered",
        "BK-10235": "standard_refund",
        "BK-10236": "standard_refund",
        "BK-10237": "standard_refund",
        "BK-10238": "already_refunded",
        "BK-10239": "escalate",
        "BK-10240": "cancelled",
        "BK-10241": "standard_refund",
        "BK-10242": "standard_refund",
        "BK-10243": "standard_refund",
        "BK-10244": "not_delivered",
        "BK-10245": "not_delivered",
        "BK-10246": "escalate",
    }
    return SeedData(customers, orders, refunds, expected)


def decide_seed(
    data: SeedData,
    today: date,
    policy: Any = None,
) -> dict[str, RefundDecision]:
    customers = {customer["_id"]: customer for customer in data.customers}
    return {
        order["_id"]: decide_refund(order, customers[order["customer_id"]], today, policy)
        for order in data.orders
    }


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("MONGODB_URI is not set. Copy .env.example to .env.", file=sys.stderr)
        return 1

    # A bad policy file stops the seed. The demo data and the rules must agree.
    try:
        policy = load_refund_policy()
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    today = date.today()
    data = build_seed(today)
    mismatches = _mismatches(data, today, policy)
    if mismatches:
        print("Seed data does not match the policy:", file=sys.stderr)
        print("\n".join(mismatches), file=sys.stderr)
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        db_name = os.environ.get("MONGODB_DB", "bookly")
        database = client[db_name]
        for name in ("customers", "orders", "refunds", "tickets"):
            database[name].delete_many({})
        database.customers.insert_many(_for_mongo(customer) for customer in data.customers)
        database.orders.insert_many(_for_mongo(order) for order in data.orders)
        database.refunds.insert_many(_for_mongo(refund) for refund in data.refunds)
        database.customers.create_index("email", unique=True)
        database.orders.create_index("customer_id")
        database.refunds.create_index("order_id")
        try:
            ensure_order_search_index(database.orders)
        except Exception as exc:
            print(f"Order search index failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        stored = SeedData(
            customers=tuple(database.customers.find()),
            orders=tuple(database.orders.find()),
            refunds=tuple(database.refunds.find()),
            expected_rule_id=data.expected_rule_id,
        )
        stored_mismatches = _mismatches(stored, today, policy)
        if stored_mismatches:
            print("Stored documents drifted from the policy:", file=sys.stderr)
            print("\n".join(stored_mismatches), file=sys.stderr)
            return 1
        try:
            faq_note = seed_faq(database)
        except Exception as exc:
            print(f"FAQ seed failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    except PyMongoError as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()

    print(f"Seeded {os.environ.get('MONGODB_DB', 'bookly')} for {today.isoformat()}")
    print(faq_note)
    for order in data.orders:
        print(f"  {order['_id']}  {data.expected_rule_id[order['_id']]}")
    return 0


def ensure_order_search_index(collection, timeout_seconds: int = 120) -> None:
    """Fuzzy text search on item titles and authors, for the search_orders tool."""

    if not any(index.get("name") == ORDER_SEARCH_INDEX for index in collection.list_search_indexes()):
        collection.create_search_index(
            SearchIndexModel(
                definition={
                    "mappings": {
                        "dynamic": False,
                        "fields": {
                            "items": {
                                "type": "document",
                                "fields": {
                                    "title": {"type": "string"},
                                    "author": {"type": "string"},
                                },
                            }
                        },
                    }
                },
                name=ORDER_SEARCH_INDEX,
                type="search",
            )
        )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for index in collection.list_search_indexes():
            if index.get("name") == ORDER_SEARCH_INDEX and (
                index.get("queryable") or index.get("status") == "READY"
            ):
                return
        time.sleep(2)
    raise TimeoutError(f"{ORDER_SEARCH_INDEX} was not queryable within {timeout_seconds}s")


def _mismatches(data: SeedData, today: date, policy: Any) -> list[str]:
    found = decide_seed(data, today, policy)
    problems = []
    for order_id, expected in data.expected_rule_id.items():
        actual = found[order_id].rule_id
        if actual != expected:
            problems.append(f"{order_id}: expected {expected}, got {actual}")
    return problems


def _customer(customer_id: str, name: str, email: str, prior_refund_count: int) -> dict[str, Any]:
    return {
        "_id": customer_id,
        "name": name,
        "email": email,
        "prior_refund_count": prior_refund_count,
    }


def _item(title: str, author: str, price: str, qty: int = 1) -> dict[str, Any]:
    return {"title": title, "author": author, "qty": qty, "price": Decimal(price)}


def _order(
    order_id: str,
    customer_id: str,
    items: list[dict[str, Any]],
    status: str,
    today: date,
    ordered_days: int,
    shipped_days: int | None = None,
    delivered_days: int | None = None,
    eta_in_days: int | None = None,
    carrier: str | None = None,
    tracking_number: str | None = None,
    checkpoints: tuple[dict[str, Any], ...] | None = None,
) -> dict[str, Any]:
    total = sum((item["price"] * item["qty"] for item in items), Decimal("0"))
    document = {
        "_id": order_id,
        "customer_id": customer_id,
        "items": items,
        "total": total,
        "status": status,
        "carrier": carrier,
        "tracking_number": tracking_number,
        "ordered_at": _days_ago(today, ordered_days),
        "shipped_at": _days_ago(today, shipped_days) if shipped_days is not None else None,
        "delivered_at": _days_ago(today, delivered_days) if delivered_days is not None else None,
        "eta": today + timedelta(days=eta_in_days) if eta_in_days is not None else None,
    }
    if checkpoints:
        document["checkpoints"] = [dict(point) for point in checkpoints]
    return document


def _days_ago(today: date, days: int) -> date:
    return today - timedelta(days=days)


def _for_mongo(document: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in document.items():
        converted[key] = _mongo_value(value)
    return converted


def _mongo_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return Decimal128(value)
    # Noon, with no timezone, keeps the calendar day. A UTC conversion could
    # move a delivery across midnight and change which policy rule matches.
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day, 12, 0, 0)
    if isinstance(value, list):
        return [_mongo_value(item) for item in value]
    if isinstance(value, dict):
        return _for_mongo(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
