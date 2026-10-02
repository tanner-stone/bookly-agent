"""Sign-in for the chat channels.

The email is collected in a box (or a prompt before the CLI loop), resolved
to a customer, and passed into the graph as customer_id. The transcript
never asks for it.
"""

from __future__ import annotations

from policy.config import MAX_EMAIL_ATTEMPTS


def first_name(full_name: str | None) -> str | None:
    if not full_name:
        return None
    token = full_name.strip().split()[0]
    return token or None


def greeting(name: str | None) -> str:
    """Opening line. ``name`` is already the first name, or None."""

    if name:
        return f"{name}, how can I assist you today?"
    return "How can I assist you today?"


def sign_in_feedback(failures: int, support_email: str, support_phone: str) -> str | None:
    """Message to show on the sign-in box after a failed lookup. None before any try."""

    if failures <= 0:
        return None
    if failures >= MAX_EMAIL_ATTEMPTS:
        return (
            "No account for that email. "
            f"You can reach a person at {support_email} or {support_phone}."
        )
    return "No account for that email. Check the spelling and try the email on the order."
