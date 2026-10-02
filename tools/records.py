"""Plain dictionaries the model and the checkpointer can store.

Order documents in Mongo use Decimal128 and datetimes. Graph state cannot.
Conversion lives here so a node cannot hand the model a live database object.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def money_str(value: Any) -> str:
    if hasattr(value, "to_decimal"):
        value = value.to_decimal()
    return f"{Decimal(str(value)):.2f}"


def as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def date_iso(value: Any) -> str | None:
    day = as_date(value)
    if day is None:
        return None
    return day.isoformat()


def age_label(value: Any, today: date) -> str | None:
    """A label the model can read. It does not get to do the day math."""

    day = as_date(value)
    if day is None:
        return None
    days = (today - day).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def order_title(order: dict[str, Any]) -> str:
    titles = [item["title"] for item in order.get("items") or []]
    return "; ".join(titles)


def _authors(order: dict[str, Any]) -> str:
    names = [item.get("author") or "" for item in order.get("items") or []]
    return "; ".join(name for name in names if name)


def public_order(order: dict[str, Any], today: date) -> dict[str, Any]:
    """Facts a reply may quote. Eligibility inputs stay out of this view."""

    viewed = {
        "order_id": order["_id"],
        "title": order_title(order),
        "author": _authors(order),
        "status": order.get("status"),
        "total": money_str(order["total"]),
        "ordered_at": date_iso(order.get("ordered_at")),
        "ordered_label": age_label(order.get("ordered_at"), today),
        "delivered_at": date_iso(order.get("delivered_at")),
        "delivered_label": age_label(order.get("delivered_at"), today),
        "carrier": order.get("carrier"),
        "tracking_number": order.get("tracking_number"),
        "eta": date_iso(order.get("eta")),
    }
    tracking = _tracking(order)
    if tracking is not None:
        viewed["tracking"] = tracking
    return viewed


def directive_for_model(directive: dict[str, Any]) -> dict[str, Any]:
    """Drop map coordinates. The model may say the city. The UI draws the points."""

    packed = dict(directive)
    order = packed.get("order")
    if isinstance(order, dict):
        packed["order"] = _order_for_model(order)
    orders = packed.get("orders")
    if isinstance(orders, list):
        packed["orders"] = [
            _order_for_model(item) if isinstance(item, dict) else item for item in orders
        ]
    return packed


def _tracking(order: dict[str, Any]) -> dict[str, Any] | None:
    points = order.get("checkpoints") or []
    if not points:
        return None
    current = points[-1]
    return {
        "carrier": order.get("carrier"),
        "place": current.get("place"),
        "eta": date_iso(order.get("eta")),
        "points": [
            {"place": point.get("place"), "lat": point.get("lat"), "lng": point.get("lng")}
            for point in points
        ],
    }


def _order_for_model(order: dict[str, Any]) -> dict[str, Any]:
    tracking = order.get("tracking")
    if not isinstance(tracking, dict) or "points" not in tracking:
        return order
    spoken = dict(order)
    spoken["tracking"] = {
        key: tracking.get(key) for key in ("carrier", "place", "eta") if tracking.get(key)
    }
    return spoken
