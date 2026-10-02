"""Table-driven coverage for the shipped refund policy.

Outcomes come from the YAML file. A second file with different limits proves
the engine is not hiding 50 or 150 in code. A broken file must raise.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from policy.refund_policy import PolicyError, decide_refund, load_refund_policy

TODAY = date(2026, 6, 15)


def _order(status: str, total: str, days: int | None, delivered_at=None):
    if delivered_at is None and days is not None:
        delivered_at = TODAY - timedelta(days=days)
    return {"status": status, "total": Decimal(total), "delivered_at": delivered_at}


def _customer(prior: int):
    return {"prior_refund_count": prior}


# name, status, total, days since delivery, prior refunds, outcome, rule id, tier
CASES = [
    ("standard on 50.00 day 30", "delivered", "50.00", 30, 0, "AUTO_REFUND", "standard_refund", "standard"),
    ("standard ignores a prior refund", "delivered", "50.00", 30, 1, "AUTO_REFUND", "standard_refund", "standard"),
    ("50.01 day 30 first refund is goodwill", "delivered", "50.01", 30, 0, "AUTO_REFUND", "first_time_goodwill", "first_time_goodwill"),
    ("50.01 day 30 repeat escalates", "delivered", "50.01", 30, 1, "ESCALATE", "escalate", None),
    ("day 31 at 50.00 is goodwill", "delivered", "50.00", 31, 0, "AUTO_REFUND", "first_time_goodwill", "first_time_goodwill"),
    ("day 31 at 50.00 repeat escalates", "delivered", "50.00", 31, 1, "ESCALATE", "escalate", None),
    ("goodwill on 150.00 day 45", "delivered", "150.00", 45, 0, "AUTO_REFUND", "first_time_goodwill", "first_time_goodwill"),
    ("150.01 day 45 escalates", "delivered", "150.01", 45, 0, "ESCALATE", "escalate", None),
    ("day 46 at 150.00 escalates", "delivered", "150.00", 46, 0, "ESCALATE", "escalate", None),
    ("day 46 at 50.00 escalates", "delivered", "50.00", 46, 0, "ESCALATE", "escalate", None),
    ("goodwill on 150.00 day 45 repeat escalates", "delivered", "150.00", 45, 1, "ESCALATE", "escalate", None),
    ("60 day 20 repeat escalates", "delivered", "60.00", 20, 1, "ESCALATE", "escalate", None),
    ("60 day 20 first refund is goodwill", "delivered", "60.00", 20, 0, "AUTO_REFUND", "first_time_goodwill", "first_time_goodwill"),
    ("32 day 10 is standard", "delivered", "32.00", 10, 0, "AUTO_REFUND", "standard_refund", "standard"),
    ("120 day 40 is goodwill", "delivered", "120.00", 40, 0, "AUTO_REFUND", "first_time_goodwill", "first_time_goodwill"),
    ("200 day 5 escalates", "delivered", "200.00", 5, 0, "ESCALATE", "escalate", None),
    ("delivered today is standard", "delivered", "50.00", 0, 3, "AUTO_REFUND", "standard_refund", "standard"),
    ("refunded cheap order stops", "refunded", "10.00", 1, 0, "ALREADY_REFUNDED", "already_refunded", None),
    ("cancelled is not applicable", "cancelled", "10.00", 1, 0, "NOT_APPLICABLE", "cancelled", None),
    ("cancelled wins even without a delivery date", "cancelled", "28.00", None, 0, "NOT_APPLICABLE", "cancelled", None),
    ("in transit is too early", "in_transit", "24.00", None, 0, "NOT_ELIGIBLE_YET", "not_delivered", None),
    ("shipped is too early", "shipped", "24.00", None, 0, "NOT_ELIGIBLE_YET", "not_delivered", None),
    ("processing is too early", "processing", "24.00", None, 0, "NOT_ELIGIBLE_YET", "not_delivered", None),
    ("delivered with no delivery date escalates", "delivered", "20.00", None, 0, "ESCALATE", "escalate", None),
]


@pytest.mark.parametrize(
    ("name", "status", "total", "days", "prior", "outcome", "rule_id", "tier"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_shipped_policy_branches(name, status, total, days, prior, outcome, rule_id, tier):
    del name
    decision = decide_refund(_order(status, total, days), _customer(prior), TODAY)
    assert decision.outcome == outcome
    assert decision.rule_id == rule_id
    assert decision.reason_code == rule_id
    assert decision.tier == tier


def test_goodwill_copies_the_one_time_note():
    policy = load_refund_policy()
    decision = decide_refund(_order("delivered", "120.00", 40), _customer(0), TODAY, policy)
    note = policy.get_rule("first_time_goodwill").customer_note
    assert note and "one-time" in note
    assert decision.customer_note == note
    standard = decide_refund(_order("delivered", "32.00", 10), _customer(0), TODAY, policy)
    assert standard.customer_note is None


def test_float_boundary_is_not_rounded_into_standard():
    order = {"status": "delivered", "total": 50.01, "delivered_at": TODAY - timedelta(days=30)}
    decision = decide_refund(order, _customer(0), TODAY)
    assert decision.rule_id == "first_time_goodwill"


def test_datetime_delivery_uses_the_calendar_day():
    delivered = datetime(2026, 5, 16, 12, 0, 0)
    order = {"status": "delivered", "total": Decimal("50.00"), "delivered_at": delivered}
    decision = decide_refund(order, _customer(0), TODAY)
    assert decision.rule_id == "standard_refund"


def test_aware_datetime_is_read_in_utc():
    # 03:30 UTC on June 15 is still June 14 in US Eastern, but the stored instant
    # is the 15th in UTC. Day count is 0, so the standard rule matches.
    delivered = datetime(2026, 6, 15, 3, 30, tzinfo=timezone.utc)
    order = {"status": "delivered", "total": Decimal("40.00"), "delivered_at": delivered}
    decision = decide_refund(order, _customer(1), TODAY)
    assert decision.rule_id == "standard_refund"


def test_status_beats_a_delivery_date_on_an_in_transit_order():
    order = _order("in_transit", "20.00", 2)
    decision = decide_refund(order, _customer(0), TODAY)
    assert decision.outcome == "NOT_ELIGIBLE_YET"
    assert decision.rule_id == "not_delivered"


def test_today_must_be_a_date_not_a_datetime():
    with pytest.raises(TypeError):
        decide_refund(
            _order("delivered", "10.00", 1),
            _customer(0),
            datetime(2026, 6, 15, 18, 0, 0),
        )


def test_shipped_contacts():
    policy = load_refund_policy()
    assert policy.support_email == "support@bookly.example"
    assert policy.support_phone == "1-800-555-0148"


def test_alternate_yaml_changes_the_outcome(tmp_path):
    path = tmp_path / "tight.yaml"
    path.write_text(
        """
support:
  email: support@bookly.example
  phone: "1-800-555-0199"
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: standard_refund
    when: {max_total_usd: "10.00", max_days_since_delivery: 30}
    outcome: AUTO_REFUND
    tier: standard
  - id: first_time_goodwill
    when:
      max_prior_refunds: 0
      max_total_usd: "10.00"
      max_days_since_delivery: 45
    outcome: AUTO_REFUND
    tier: first_time_goodwill
  - id: escalate
    outcome: ESCALATE
""",
        encoding="utf-8",
    )
    policy = load_refund_policy(path)
    thirty_two = decide_refund(_order("delivered", "32.00", 10), _customer(0), TODAY, policy)
    nine = decide_refund(_order("delivered", "9.00", 10), _customer(0), TODAY, policy)
    assert thirty_two.rule_id == "escalate"
    assert nine.rule_id == "standard_refund"
    shipped = decide_refund(_order("delivered", "32.00", 10), _customer(0), TODAY)
    assert shipped.rule_id == "standard_refund"


@pytest.mark.parametrize(
    ("text", "snippet"),
    [
        (
            "support: {email: nope, phone: '1'}\nrules: []\n",
            "support.email",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: standard_refund
    when: {max_total: "50.00", max_days_since_delivery: 30}
    outcome: AUTO_REFUND
    tier: standard
  - id: escalate
    outcome: ESCALATE
""",
            "max_total",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: standard_refund
    when: {max_total_usd: "-1.00", max_days_since_delivery: 30}
    outcome: AUTO_REFUND
    tier: standard
  - id: escalate
    outcome: ESCALATE
""",
            "non-negative",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: already_refunded
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: escalate
    outcome: ESCALATE
""",
            "Duplicate rule id",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: escalate
    outcome: ESCALATE
""",
            "before the not-delivered rule",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: standard_refund
    when: {max_total_usd: "50.00", max_days_since_delivery: 30}
    outcome: AUTO_REFUND
  - id: escalate
    outcome: ESCALATE
""",
            "tier is required",
        ),
        (
            """
support: {email: a@b.example, phone: "1"}
rules:
  - id: already_refunded
    when: {order_status: refunded}
    outcome: ALREADY_REFUNDED
  - id: cancelled
    when: {order_status: cancelled}
    outcome: NOT_APPLICABLE
  - id: not_delivered
    when: {order_status_not: delivered}
    outcome: NOT_ELIGIBLE_YET
  - id: standard_refund
    when: {max_days_since_delivery: 30.5, max_total_usd: "50.00"}
    outcome: AUTO_REFUND
    tier: standard
  - id: escalate
    outcome: ESCALATE
""",
            "whole number",
        ),
    ],
)
def test_invalid_yaml_fails_loudly(tmp_path, text, snippet):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(PolicyError, match=snippet):
        load_refund_policy(path)


def test_missing_file_fails_loudly(tmp_path):
    with pytest.raises(PolicyError, match="not found"):
        load_refund_policy(tmp_path / "missing.yaml")
