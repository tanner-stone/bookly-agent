# Assumptions

These are the choices a support lead would want named before arguing with a trace.

## Whole-order refunds

A refund covers the whole order. There is no partial refund, no per-item amount, and no "refund the shipping only" path. The amount written is the order total, and the tier recorded is the tier from the rule that matched.

## Already refunded

"Already refunded" means the order status is `refunded`. The engine does not scan the refunds collection. Casey's seed row exists so the write has a record, but the rule matches the status. A second `issue_refund` on that order raises and does not write again.

## Inclusive limits

Dollar and day limits use `<=`. An order total equal to the cap still matches. A delivery age equal to the day cap still matches. A future delivery date does not match a day rule.

## Standard tier ignores prior refunds

`standard_refund` does not look at `prior_refund_count`. A customer who has been refunded before still gets that tier when the total is within the cap and the delivery is recent enough. The repeat-customer case in the demo escalates because it misses standard (over the dollar cap) and then misses goodwill (prior refunds are not zero). It does not escalate merely because they have been refunded before.

## Cancelled is not "still on the way"

A cancelled order is `NOT_APPLICABLE` (`cancelled`). That rule is above "not delivered", so a cancelled order is not described as too early to refund. No ticket is opened. The loader rejects a policy file that puts either the refunded rule or the cancelled rule after the not-delivered rule.

## Goodwill is one note

`first_time_goodwill` carries the customer note from the YAML file. The phrasing prompt is told to include that note. The note is not a second policy decision.

## Identity

The email is collected in the sign-in box, or at the CLI prompt before the loop. It is resolved to `customer_id` and stored on the thread. Changing the email starts a new thread. FAQ questions do not require an account. Order status and refunds do. Two failed lookups on the box offer the support phone from the YAML file. The box still accepts another try after that.

## FAQ vectors

This process calls Voyage. The database stores the vectors and runs `$vectorSearch`. It does not embed on its own, and the Voyage key is not given to the Mongo container. Document vectors are computed at seed time (`input_type=document`). Query vectors are computed at search time (`input_type=query`). A score under the threshold is not shown to the model.

## Dates in the seed

Seed dates are offsets from the day the seed runs. Delivery timestamps are stored at noon with no timezone, so converting to UTC cannot move the calendar day. The policy engine compares dates, not timestamps.

## What the eval does not check

`python3 -m pytest` never calls the model or Voyage. `python3 -m evals.run` does call the model, checks rule ids and writes, and does not assert the assistant's sentences. It uses an in-memory store so the demo database stays as seeded.
