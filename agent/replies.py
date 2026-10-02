"""The one sentence the graph says without a model call.

Everything else, including order cards, follow-ups, handoffs, and the rating
question, is phrased by the model from a directive.
"""

from __future__ import annotations


def fixed_reply(directive: dict, channel: str) -> str | None:
    if directive.get("kind") == "sign_in_required":
        return "Please sign in with the email on your Bookly account before I look up an order."
    return None
