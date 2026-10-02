"""Schemas for the only tools the model may call.

The functions raise on purpose. The graph executes reads itself so a write
can never be smuggled in as a tool body.
"""

from typing import Literal

from langchain_core.tools import tool


@tool
def get_order(order_id: str) -> str:
    """Look up one order for the customer already identified in this conversation."""

    raise RuntimeError("get_order is executed by the graph")


@tool
def search_orders(
    status: Literal["processing", "in_transit", "delivered", "cancelled", "refunded"] | None = None,
    title: str | None = None,
    author: str | None = None,
    ordered_after: str | None = None,
    ordered_before: str | None = None,
    sort: Literal["newest", "oldest"] = "newest",
    limit: int = 10,
) -> str:
    """Filter the signed-in customer's orders. Every argument is optional.

    title and author tolerate small misspellings. Dates are YYYY-MM-DD and
    inclusive. Use this when the order you need is not in the listed orders.
    """

    raise RuntimeError("search_orders is executed by the graph")


@tool
def search_faq(query: str) -> str:
    """Search Bookly help articles. Pass the customer's question as the query."""

    raise RuntimeError("search_faq is executed by the graph")


READ_ONLY_TOOLS = [get_order, search_orders, search_faq]
READ_ONLY_NAMES = {item.name for item in READ_ONLY_TOOLS}
