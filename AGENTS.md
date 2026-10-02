# Bookly agent

Customer support agent for Bookly, a fictional online bookstore. Chat and voice share one LangGraph. The model handles language. Code handles anything with money or policy consequences.

## Stack

- Python 3.11+. Invoke it as `python3`, never `python`.
- LangGraph state graph (nodes and edges). No prebuilt agent.
- `ChatXAI` (`langchain-xai`). Model id from `XAI_MODEL` (default `grok-4.3`, `XAI_REASONING_EFFORT=none`).
- Voice is a cascade: xAI `/v1/stt` → this graph → xAI `/v1/tts`. Do not use the realtime speech-to-speech API.
- MongoDB via `mongodb/mongodb-atlas-local` in Docker. Pin the dated image tag in `docker-compose.yml`. Do not float `latest` or `preview`.
- Conversation memory: `MongoDBSaver` from `langgraph-checkpoint-mongodb`, keyed by `thread_id`.
- Embeddings: Voyage AI (`VOYAGE_API_KEY`, `VOYAGE_MODEL`, default `voyage-4`). This process computes vectors. The database does not.
- UI: Streamlit.

## Risk split

The model reads each turn (`TurnRead`: goal, order, reason, yes/no, stars) and may call read-only tools (order lookup, order filter search, FAQ search) through a hand-rolled tool loop. `decide_refund`, `issue_refund`, and `create_ticket` are not model tools. Graph nodes call them. Code keeps only order ids the customer owns. `issue_refund` runs only when a refund was offered last turn, the model reads a yes this turn for that same order, and a fresh `decide_refund` still says AUTO_REFUND.

A bad read is a wrong sentence. A bad write moves money or opens a ticket.

## Rules a person can edit

Refund tiers, amount and day thresholds, and escalation contacts live in `policy/refund_policy.yaml`. Code loads that file, validates it, and refuses to start on a bad value. Comparisons are inclusive (`<=`). Technical knobs (FAQ score threshold, email-attempt cap, tool-loop cap) live in `policy/config.py`.

## Trace

Each turn records intent, read-only tool calls, and, when a refund is decided, the outcome plus the YAML rule id that fired. Streamlit shows this in a collapsible "Agent trace" panel.

## Phases

- Phase 0: skeleton, Cursor rules, Atlas Local connection. Done when `python3 scripts/check_mongo.py` pings a replica set.
- Phase 1: YAML policy, `decide_refund`, seed data, pytest. No LLM. `python3 -m pytest` then `python3 -m data.seed`.
- Phase 2: graph, CLI (`python3 -m agent.cli`), MongoDB checkpointer, turn trace. Pytest uses a scripted model and an in-memory checkpointer.
- Phase 3: Voyage embeddings and Atlas vector search. `python3 -m data.seed` embeds `data/faq/` when `VOYAGE_API_KEY` is set.
- Phase 4: Streamlit chat, email box, and the trace panel. `python3 -m streamlit run app/streamlit_app.py`
- Phase 5: voice cascade on the same `thread_id`. `st.audio_input` is WAV. Speech-to-text and text-to-speech are separate HTTP calls around this graph.
- Phase 6: model-free scenario tests stay in pytest. `python3 -m evals.run` is the opt-in real-model pass. `docs/ARCHITECTURE.md` and `docs/ASSUMPTIONS.md` record the risk split and the policy assumptions.

Pytest never calls xAI. The eval script is not part of CI.
