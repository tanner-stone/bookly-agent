# Bookly

Customer support for a fictional online bookstore, over chat and voice. One LangGraph serves both channels. The model understands the customer and explains outcomes. Deterministic code decides refunds, tickets, and other writes.

Read-only lookups (orders, FAQ) are tools the model may call. Refund eligibility, issuing a refund, and opening a ticket are graph nodes. The model cannot call them.

Business rules a support lead can change — tiers, dollar and day limits, escalation contacts — live in `policy/refund_policy.yaml`. Code loads that file and stops if it is invalid. Comparisons are inclusive.

## Prerequisites

- Python 3.11+
- Docker with Compose

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
docker compose up -d
python3 scripts/check_mongo.py
python3 -m data.seed
python3 -m pytest
python3 -m agent.cli
python3 -m streamlit run app/streamlit_app.py
```

`check_mongo.py` prints `ping ok` and the replica set name. The container hostname is `mongodb`, which does not resolve on the host, so `MONGODB_URI` sets `directConnection=true`.

`python3 -m agent.cli` is the chat channel. It asks for the account email before the conversation starts, then prints a `thread` id and a one-line trace after each reply (intent, tool calls, and the refund rule when one fired). Resume that conversation with `python3 -m agent.cli --thread <id>`. The CLI calls the model. `python3 -m pytest` does not.

`python3 -m streamlit run app/streamlit_app.py` is the same graph with an email box. The transcript does not ask for the address. The Agent trace panel is the collapsible record of each turn.

Chat uses `grok-4.3` with `XAI_REASONING_EFFORT=none`. These turns read a message against the customer's orders or phrase a decision the code already made, so the larger reasoning model is not worth the wait. Change `XAI_MODEL` if you want a different one.

The image tag in `docker-compose.yml` is a dated Atlas Local build. It includes `mongot` so FAQ search can compare embeddings this app computes. Do not switch it to `latest` or `preview`. `python3 -m data.seed` embeds the articles in `data/faq/` when `VOYAGE_API_KEY` is set.

## Demo customers

`python3 -m data.seed` reloads these customers. Dates are computed from the day the script runs. Refunds are for the whole order. The standard tier does not look at prior refunds: a repeat customer with a cheap recent order still qualifies. The $60 case escalates because it is over $50 and the customer has refunded before.

| Customer | Email | What to try | Expected rule |
| --- | --- | --- | --- |
| Maya Chen | maya.chen@bookly.example | Refund `BK-10231`, $32, delivered 10 days ago, no prior refunds | `standard_refund` |
| Jordan Lee | jordan.lee@bookly.example | Refund `BK-10232`, $120, delivered 40 days ago, first refund | `first_time_goodwill` |
| Sam Patel | sam.patel@bookly.example | Refund `BK-10233`, $60, delivered 20 days ago, already refunded once | `escalate` |
| Riley Nguyen | riley.nguyen@bookly.example | Status for `BK-10234` (in transit, tracking and ETA). Then ask for a refund | Status from the order record. Refund rule `not_delivered` |
| Alex Rivera | alex.rivera@bookly.example | Ask about "my order" with no title or id. Three recent orders: `BK-10235` The Left Hand of Darkness (4 days), `BK-10236` Piranesi (12 days), `BK-10237` A Wizard of Earthsea (21 days) | Agent lists them and asks which one |
| Casey Brooks | casey.brooks@bookly.example | Refund `BK-10238`, already refunded | `already_refunded` |
| Taylor Kim | taylor.kim@bookly.example | Refund `BK-10239`, $200, delivered 5 days ago | `escalate` |
| Quinn Alvarez | quinn.alvarez@bookly.example | Refund `BK-10240`, order cancelled | `cancelled` (`NOT_APPLICABLE`) |
| Tanner Stone | tanner.stone@bookly.example | Ask about "my order" without a title. Six orders: two with "Night" in the title, two by Ursula K. Le Guin, `BK-10244` The Dispossessed and `BK-10245` The Overstory still on the way, and `BK-10246` The Goldfinch delivered 60 days ago | "night" lists only the two Night books. "Le Guin" lists only the two Le Guin books. "hasn't arrived" lists both shipments and naming one shows where it is. "the older one" then selects the earlier of the books just listed. Refunding The Goldfinch escalates |

Escalations point at `support@bookly.example` and `1-800-555-0148`.

## Layout

```
agent/          LangGraph (Phase 2)
tools/          Order, FAQ, refund, and ticket functions
policy/         YAML rules and technical config
voice/          Speech-to-text and text-to-speech
data/           Seed script and FAQ markdown
app/            Streamlit UI (Phase 4)
evals/          Opt-in real-model scenarios (Phase 6)
tests/          Pytest. No live model calls
docs/           Architecture and assumptions (Phase 6)
```

## Phases

Phase 0 is the skeleton and the database connection check. Phase 1 is the refund policy and the seed data. Phase 2 is the graph and the CLI. Phase 3 is FAQ embeddings and vector search. Phase 4 is the Streamlit chat, the email box, and the trace panel. Phase 5 is the voice cascade on that same thread. Phase 6 is `python3 -m evals.run` plus `docs/ARCHITECTURE.md` and `docs/ASSUMPTIONS.md`. `python3 -m pytest` does not call a model and does not need the database. `python3 -m evals.run` calls the real model and is not part of `pytest`.
