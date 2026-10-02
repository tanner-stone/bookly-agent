"""Load refund rules from YAML and apply them.

The model must not decide eligibility. This module does. Limits are whatever
the YAML file says, compared with <=, so a support lead can change a number
without a code change. The first matching rule wins, which is why rule order
in the file is the policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_POLICY_PATH = Path(__file__).with_name("refund_policy.yaml")

OUTCOMES = (
    "ALREADY_REFUNDED",
    "NOT_APPLICABLE",
    "NOT_ELIGIBLE_YET",
    "AUTO_REFUND",
    "ESCALATE",
)
ORDER_STATUSES = (
    "processing",
    "shipped",
    "in_transit",
    "delivered",
    "refunded",
    "cancelled",
)
WHEN_KEYS = (
    "order_status",
    "order_status_not",
    "max_total_usd",
    "max_days_since_delivery",
    "max_prior_refunds",
)
RULE_KEYS = ("id", "description", "when", "outcome", "tier", "customer_note")


class PolicyError(Exception):
    """The YAML file cannot be enforced. No refund decision was made."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"Invalid refund policy:\n{detail}")


@dataclass(frozen=True)
class RuleWhen:
    order_status: str | None = None
    order_status_not: str | None = None
    max_total_usd: Decimal | None = None
    max_days_since_delivery: int | None = None
    max_prior_refunds: int | None = None


@dataclass(frozen=True)
class RefundRule:
    id: str
    outcome: str
    when: RuleWhen | None
    tier: str | None = None
    description: str | None = None
    customer_note: str | None = None


@dataclass(frozen=True)
class RefundPolicy:
    support_email: str
    support_phone: str
    rules: tuple[RefundRule, ...]
    source: Path | None = None

    def get_rule(self, rule_id: str) -> RefundRule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)


@dataclass(frozen=True)
class RefundDecision:
    outcome: str
    tier: str | None
    reason_code: str
    rule_id: str
    customer_note: str | None = None


def load_refund_policy(path: Path | str | None = None) -> RefundPolicy:
    """Read and validate the policy. Raise PolicyError instead of guessing."""

    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    if not policy_path.is_file():
        raise PolicyError([f"Policy file not found: {policy_path}"])
    try:
        loaded = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError([f"Could not parse YAML: {exc}"]) from exc
    return _validate(loaded, policy_path)


def decide_refund(
    order: Mapping[str, Any],
    customer: Mapping[str, Any],
    today: date,
    policy: RefundPolicy | None = None,
) -> RefundDecision:
    """Apply the loaded rules. `today` is passed in so tests control the clock.

    `today` must be a calendar date. A datetime is rejected so a time-of-day
    cannot change which day a delivery falls on.
    """

    if isinstance(today, datetime) or not isinstance(today, date):
        raise TypeError("today must be a datetime.date")

    active = policy if policy is not None else load_refund_policy()
    status = _require(order, "status")
    if status not in ORDER_STATUSES:
        raise ValueError(f"Unknown order status: {status!r}")
    prior = _require(customer, "prior_refund_count")
    if isinstance(prior, bool) or not isinstance(prior, int) or prior < 0:
        raise ValueError("prior_refund_count must be a non-negative integer")

    for rule in active.rules:
        if _matches(rule, order, prior, today):
            # reason_code repeats the YAML id so a trace can be read on its own.
            return RefundDecision(
                outcome=rule.outcome,
                tier=rule.tier,
                reason_code=rule.id,
                rule_id=rule.id,
                customer_note=rule.customer_note,
            )
    raise PolicyError(["No rule matched, including the catch-all"])


def _matches(rule: RefundRule, order: Mapping[str, Any], prior: int, today: date) -> bool:
    if rule.when is None:
        return True
    condition = rule.when
    status = order["status"]
    if condition.order_status is not None and status != condition.order_status:
        return False
    if condition.order_status_not is not None and status == condition.order_status_not:
        return False
    if condition.max_total_usd is not None:
        if _money(order.get("total"), "order.total") > condition.max_total_usd:
            return False
    if condition.max_days_since_delivery is not None:
        delivered = order.get("delivered_at")
        if delivered is None:
            return False
        # A future delivery date is bad data, not an eligible refund.
        days = (today - _as_date(delivered)).days
        if days < 0 or days > condition.max_days_since_delivery:
            return False
    if condition.max_prior_refunds is not None and prior > condition.max_prior_refunds:
        return False
    return True


def _validate(loaded: Any, path: Path) -> RefundPolicy:
    errors: list[str] = []
    if not isinstance(loaded, dict):
        raise PolicyError(["The policy file must be a mapping with support and rules"])

    extra_root = set(loaded) - {"support", "rules"}
    for key in sorted(extra_root):
        errors.append(f"Unknown top-level key {key!r}")

    support = loaded.get("support")
    email = ""
    phone = ""
    if not isinstance(support, dict):
        errors.append("support must be a mapping with email and phone")
    else:
        extra_support = set(support) - {"email", "phone"}
        for key in sorted(extra_support):
            errors.append(f"support.{key} is not a support field")
        email = _text(support.get("email"), "support.email", errors)
        phone = _text(support.get("phone"), "support.phone", errors)
        if email and "@" not in email:
            errors.append("support.email must contain @")

    raw_rules = loaded.get("rules")
    rules: list[RefundRule] = []
    if not isinstance(raw_rules, list) or not raw_rules:
        errors.append("rules must be a non-empty list")
    else:
        seen: set[str] = set()
        for index, raw in enumerate(raw_rules, start=1):
            rule, rule_errors = _parse_rule(raw, index)
            errors.extend(rule_errors)
            if rule is None:
                continue
            if rule.id in seen:
                errors.append(f"Duplicate rule id {rule.id!r}")
            seen.add(rule.id)
            rules.append(rule)
        _check_shape(rules, errors)

    if errors:
        raise PolicyError(errors)
    return RefundPolicy(
        support_email=email,
        support_phone=phone,
        rules=tuple(rules),
        source=path,
    )


def _parse_rule(raw: Any, index: int) -> tuple[RefundRule | None, list[str]]:
    where = f"rules[{index}]"
    if not isinstance(raw, dict):
        return None, [f"{where} must be a mapping"]
    errors: list[str] = []
    for key in sorted(set(raw) - set(RULE_KEYS)):
        errors.append(f"{where}.{key} is not a rule field")

    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        errors.append(f"{where}.id must be a non-empty string")
        rule_id = ""
    outcome = raw.get("outcome")
    if outcome not in OUTCOMES:
        errors.append(f"{where}.outcome must be one of {', '.join(OUTCOMES)}")
        outcome = ""

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        errors.append(f"{where}.description must be a string")
        description = None
    customer_note = raw.get("customer_note")
    if customer_note is not None and not isinstance(customer_note, str):
        errors.append(f"{where}.customer_note must be a string")
        customer_note = None

    tier = raw.get("tier")
    if outcome == "AUTO_REFUND":
        if not isinstance(tier, str) or not tier.strip():
            errors.append(f"{where}.tier is required when outcome is AUTO_REFUND")
            tier = None
    elif tier is not None:
        errors.append(f"{where}.tier is only allowed when outcome is AUTO_REFUND")
        tier = None

    when, when_errors = _parse_when(raw.get("when", None), where)
    errors.extend(when_errors)
    if "when" not in raw:
        when = None
    if errors or not rule_id or not outcome:
        return None, errors
    return (
        RefundRule(
            id=rule_id,
            outcome=outcome,
            when=when,
            tier=tier if isinstance(tier, str) else None,
            description=description,
            customer_note=customer_note if isinstance(customer_note, str) else None,
        ),
        [],
    )


def _parse_when(raw: Any, where: str) -> tuple[RuleWhen | None, list[str]]:
    if raw is None:
        return None, []
    if not isinstance(raw, dict):
        return None, [f"{where}.when must be a mapping"]
    if not raw:
        return None, [f"{where}.when must not be empty"]
    errors: list[str] = []
    for key in sorted(set(raw) - set(WHEN_KEYS)):
        errors.append(f"{where}.when.{key} is not a condition")
    if "order_status" in raw and "order_status_not" in raw:
        errors.append(f"{where}.when cannot set both order_status and order_status_not")

    order_status = _status(raw.get("order_status"), f"{where}.when.order_status", errors)
    order_status_not = _status(
        raw.get("order_status_not"), f"{where}.when.order_status_not", errors
    )
    max_total = None
    if "max_total_usd" in raw:
        max_total = _positive_money(raw.get("max_total_usd"), f"{where}.when.max_total_usd", errors)
    max_days = None
    if "max_days_since_delivery" in raw:
        max_days = _whole(
            raw.get("max_days_since_delivery"),
            f"{where}.when.max_days_since_delivery",
            errors,
        )
    max_prior = None
    if "max_prior_refunds" in raw:
        max_prior = _whole(
            raw.get("max_prior_refunds"), f"{where}.when.max_prior_refunds", errors
        )
    if errors:
        return None, errors
    return (
        RuleWhen(
            order_status=order_status,
            order_status_not=order_status_not,
            max_total_usd=max_total,
            max_days_since_delivery=max_days,
            max_prior_refunds=max_prior,
        ),
        [],
    )


def _check_shape(rules: list[RefundRule], errors: list[str]) -> None:
    if not rules:
        return
    for rule in rules[:-1]:
        if rule.when is None:
            errors.append(f"{rule.id} omits when, but only the last rule may match everything")
    last = rules[-1]
    if last.when is not None or last.outcome != "ESCALATE":
        errors.append("The last rule must omit when and use outcome ESCALATE")

    refunded = _position(rules, lambda rule: rule.when is not None and rule.when.order_status == "refunded")
    cancelled = _position(rules, lambda rule: rule.when is not None and rule.when.order_status == "cancelled")
    not_delivered = _position(
        rules,
        lambda rule: rule.when is not None and rule.when.order_status_not == "delivered",
    )
    if refunded is None:
        errors.append("A rule must match order_status refunded with outcome ALREADY_REFUNDED")
    elif rules[refunded].outcome != "ALREADY_REFUNDED":
        errors.append(f"{rules[refunded].id} matches refunded orders but is not ALREADY_REFUNDED")
    if cancelled is None:
        errors.append("A rule must match order_status cancelled with outcome NOT_APPLICABLE")
    elif rules[cancelled].outcome != "NOT_APPLICABLE":
        errors.append(f"{rules[cancelled].id} matches cancelled orders but is not NOT_APPLICABLE")
    if not_delivered is None:
        errors.append("A rule must match order_status_not delivered with outcome NOT_ELIGIBLE_YET")
    elif rules[not_delivered].outcome != "NOT_ELIGIBLE_YET":
        errors.append(f"{rules[not_delivered].id} matches undelivered orders but is not NOT_ELIGIBLE_YET")
    if refunded is not None and not_delivered is not None and refunded > not_delivered:
        errors.append("The refunded-order rule must come before the not-delivered rule")
    if cancelled is not None and not_delivered is not None and cancelled > not_delivered:
        errors.append("The cancelled-order rule must come before the not-delivered rule")


def _position(rules: list[RefundRule], predicate: Any) -> int | None:
    for index, rule in enumerate(rules):
        if predicate(rule):
            return index
    return None


def _status(value: Any, where: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    if value not in ORDER_STATUSES:
        errors.append(f"{where} must be one of {', '.join(ORDER_STATUSES)}")
        return None
    return value


def _text(value: Any, where: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where} must be a non-empty string")
        return ""
    return value.strip()


def _positive_money(value: Any, where: str, errors: list[str]) -> Decimal | None:
    try:
        amount = _money(value, where)
    except (PolicyError, ValueError):
        errors.append(f"{where} must be a non-negative amount")
        return None
    if amount < 0:
        errors.append(f"{where} must be a non-negative amount")
        return None
    return amount


def _whole(value: Any, where: str, errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"{where} must be a non-negative whole number")
        return None
    return value


def _money(value: Any, where: str) -> Decimal:
    to_decimal = getattr(value, "to_decimal", None)
    if callable(to_decimal):
        value = to_decimal()
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{where} is missing")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{where} is not a money amount") from exc


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.date()
    if isinstance(value, date):
        return value
    raise ValueError("delivered_at must be a date or datetime")


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {key}")
    return mapping[key]
