"""Technical knobs.

Refund tiers, dollar limits, day limits, and escalation contacts do not belong
here. Those live in refund_policy.yaml so a support lead can edit them.
The values below change how the agent searches and how many times it asks,
not what a customer is owed.
"""

# Atlas vector score at or above this counts as a grounded FAQ hit.
FAQ_SCORE_THRESHOLD = 0.75

# Failed sign-in attempts before the email box offers a person.
# The address is not collected inside the chat.
MAX_EMAIL_ATTEMPTS = 2

# Read-only tool calls the model may make in a single turn.
TOOL_LOOP_CAP = 4

# Orders listed when the customer has not named one.
RECENT_ORDERS_LIMIT = 4

# Orders placed in the model's context each turn. A larger account is reached
# through the search_orders tool instead.
ORDER_CONTEXT_LIMIT = 20

# Voyage rate-limit handling for FAQ queries. A free key allows a few requests
# a minute, so a busy demo waits briefly instead of reporting "no article".
VOYAGE_RETRIES = 4
VOYAGE_MAX_WAIT_SECONDS = 25
