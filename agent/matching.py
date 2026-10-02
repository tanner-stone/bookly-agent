"""Check that an order the model named belongs to this customer.

The model reads the customer's words against their orders. Code only keeps
ids that are in that customer's list, so a guessed or foreign id goes nowhere.
"""

from __future__ import annotations


def owned_order_id(value: object, orders: list[dict]) -> str | None:
    if not isinstance(value, str):
        return None
    by_upper = {order["_id"].upper(): order["_id"] for order in orders}
    return by_upper.get(value.strip().upper())


def owned_order_ids(values: object, orders: list[dict]) -> list[str]:
    if not isinstance(values, list):
        return []
    kept: list[str] = []
    for value in values:
        order_id = owned_order_id(value, orders)
        if order_id and order_id not in kept:
            kept.append(order_id)
    return kept
