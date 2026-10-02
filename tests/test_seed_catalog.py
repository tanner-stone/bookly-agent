"""The demo customers stay on the rule the README promises."""

from datetime import date, timedelta
from decimal import Decimal

from data.seed import build_seed, decide_seed
from tools.records import directive_for_model, public_order


def test_each_seed_order_hits_its_documented_rule():
    today = date(2026, 9, 29)
    data = build_seed(today)
    decisions = decide_seed(data, today)
    assert decisions.keys() == data.expected_rule_id.keys()
    for order_id, expected in data.expected_rule_id.items():
        assert decisions[order_id].rule_id == expected
        assert decisions[order_id].reason_code == expected


def test_seed_dates_are_relative_to_the_given_day():
    today = date(2026, 1, 20)
    data = build_seed(today)
    orders = {order["_id"]: order for order in data.orders}
    assert orders["BK-10231"]["delivered_at"] == today - timedelta(days=10)
    assert orders["BK-10232"]["delivered_at"] == today - timedelta(days=40)
    assert orders["BK-10233"]["delivered_at"] == today - timedelta(days=20)
    assert orders["BK-10239"]["delivered_at"] == today - timedelta(days=5)
    assert orders["BK-10234"]["delivered_at"] is None
    assert orders["BK-10234"]["eta"] == today + timedelta(days=3)
    assert orders["BK-10234"]["tracking_number"] == "1Z999AA10123456784"
    assert orders["BK-10240"]["status"] == "cancelled"
    assert orders["BK-10240"]["delivered_at"] is None


def test_order_totals_match_line_items_and_casey_has_a_refund_row():
    data = build_seed(date(2026, 3, 1))
    for order in data.orders:
        line_total = sum(item["price"] * item["qty"] for item in order["items"])
        assert order["total"] == line_total
    refund = data.refunds[0]
    assert refund["order_id"] == "BK-10238"
    assert refund["amount"] == Decimal("22.00")
    assert refund["tier"] == "standard"
    customers = {customer["_id"]: customer for customer in data.customers}
    assert customers["sam-patel"]["prior_refund_count"] == 1
    assert customers["casey-brooks"]["prior_refund_count"] == 1
    alex_orders = [order for order in data.orders if order["customer_id"] == "alex-rivera"]
    assert len(alex_orders) == 3
    assert len({order["items"][0]["title"] for order in alex_orders}) == 3
    tanner = [order for order in data.orders if order["customer_id"] == "tanner-stone"]
    assert [order["_id"] for order in tanner] == [
        "BK-10241",
        "BK-10242",
        "BK-10243",
        "BK-10244",
        "BK-10245",
        "BK-10246",
    ]
    moving = [order for order in tanner if order["status"] == "in_transit"]
    assert [order["_id"] for order in moving] == ["BK-10244", "BK-10245"]
    assert moving[0]["checkpoints"][-1]["place"] == "Denver"
    assert moving[1]["checkpoints"][-1]["place"] == "Chicago"
    assert tanner[-1]["status"] == "delivered"
    assert tanner[-1]["delivered_at"] == date(2026, 3, 1) - timedelta(days=60)
    assert customers["tanner-stone"]["email"] == "tanner.stone@bookly.example"


def test_the_model_hears_the_city_and_not_the_coordinates():
    data = build_seed(date(2026, 6, 1))
    order = next(item for item in data.orders if item["_id"] == "BK-10244")
    viewed = public_order(order, date(2026, 6, 1))
    spoken = directive_for_model({"kind": "order_status", "order": viewed})
    assert spoken["order"]["tracking"]["place"] == "Denver"
    assert "points" not in spoken["order"]["tracking"]
    assert viewed["tracking"]["points"][-1]["lat"] == 39.739
