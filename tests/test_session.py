"""The sign-in box, not the chat, collects the email."""

from agent.replies import fixed_reply
from agent.session import sign_in_feedback
from policy.refund_policy import load_refund_policy


def test_sign_in_offers_a_person_after_two_misses():
    policy = load_refund_policy()
    assert sign_in_feedback(0, policy.support_email, policy.support_phone) is None
    again = sign_in_feedback(1, policy.support_email, policy.support_phone)
    assert again is not None
    assert policy.support_phone not in again
    offered = sign_in_feedback(2, policy.support_email, policy.support_phone)
    assert policy.support_email in offered
    assert policy.support_phone in offered


def test_only_the_sign_in_notice_is_a_fixed_sentence():
    orders = [{"title": "Piranesi", "order_id": "BK-10236", "ordered_label": "16 days ago"}]
    for kind in ("choose_order", "confirm_order", "ask_reason", "ask_rating", "refund_declined"):
        assert fixed_reply({"kind": kind, "orders": orders}, "voice") is None
    assert fixed_reply({"kind": "sign_in_required"}, "chat") is not None
