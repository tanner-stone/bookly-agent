"""Chat with the Bookly agent in the terminal.

One thread_id is one conversation. The checkpointer keeps it in Mongo, so
the same id can be resumed after the process exits. Pass --thread to do that.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from agent.graph import build_graph
from agent.language import XAILanguage
from agent.session import sign_in_feedback
from policy.refund_policy import PolicyError, load_refund_policy
from tools.faq_search import VectorFaqIndex, VoyageEmbedder
from tools.store import MongoStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bookly support, chat channel")
    parser.add_argument("--thread", help="Resume this conversation id")
    args = parser.parse_args(argv)

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        print("MONGODB_URI is not set. Copy .env.example to .env.", file=sys.stderr)
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except PyMongoError as exc:
        print(f"MongoDB is not reachable: {exc}", file=sys.stderr)
        client.close()
        return 1

    db_name = os.environ.get("MONGODB_DB", "bookly")
    database = client[db_name]
    try:
        graph = build_graph(
            XAILanguage(),
            MongoStore(database),
            MongoDBSaver(client, db_name=db_name),
            faq=VectorFaqIndex(database, VoyageEmbedder()),
        )
    except PolicyError as exc:
        print(exc, file=sys.stderr)
        client.close()
        return 1

    thread_id = args.thread or uuid.uuid4().hex
    print(f"thread {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    customer_id = _checkpoint_customer(graph, config)
    if customer_id:
        print("Resuming a signed-in conversation. Type quit to exit.")
    else:
        try:
            policy = load_refund_policy()
        except PolicyError as exc:
            print(exc, file=sys.stderr)
            client.close()
            return 1
        customer = _sign_in(MongoStore(database), policy)
        if customer is None:
            client.close()
            return 0
        customer_id = customer["_id"]
        print("Type quit to exit.")
    if not _checkpoint_messages(graph, config):
        opened = graph.invoke(
            {"channel": "chat", "customer_id": customer_id},
            config,
        )
        print(_last_reply(opened["messages"]))
    pending_customer = None if _checkpoint_customer(graph, config) else customer_id
    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                print()
                break
            if not text:
                continue
            if text.lower() in {"quit", "exit"}:
                break
            payload = {"messages": [HumanMessage(content=text)], "channel": "chat"}
            if pending_customer:
                payload["customer_id"] = pending_customer
                pending_customer = None
            result = graph.invoke(payload, config)
            print(_last_reply(result["messages"]))
            trace = (result.get("trace") or [])[-1:]
            if trace:
                print(_format_trace(trace[0]))
    finally:
        client.close()
    return 0


def _checkpoint_customer(graph, config) -> str | None:
    values = _checkpoint_values(graph, config)
    return values.get("customer_id")


def _checkpoint_messages(graph, config) -> list:
    values = _checkpoint_values(graph, config)
    return values.get("messages") or []


def _checkpoint_values(graph, config) -> dict:
    snapshot = graph.get_state(config)
    return getattr(snapshot, "values", None) or {}


def _sign_in(store, policy) -> dict | None:
    print("Sign in with the email on your Bookly account.")
    failures = 0
    while True:
        try:
            email = input("Email: ").strip()
        except EOFError:
            print()
            return None
        if not email:
            continue
        if email.lower() in {"quit", "exit"}:
            return None
        customer = store.get_customer_by_email(email)
        if customer:
            print(f"Signed in as {customer['name']}.")
            return customer
        failures += 1
        print(sign_in_feedback(failures, policy.support_email, policy.support_phone))


def _last_reply(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and not message.tool_calls and message.content:
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _format_trace(record: dict) -> str:
    tools = record.get("tool_calls") or []
    if tools:
        rendered = ", ".join(
            f"{call['name']} ({call['caller']}{'' if call.get('ok') else ', refused'})"
            for call in tools
        )
    else:
        rendered = "none"
    policy = record.get("policy")
    if policy:
        policy_bit = f"{policy.get('outcome')} {policy.get('rule_id')}"
    else:
        policy_bit = "none"
    read = record.get("read") or {}
    read_bit = " ".join(f"{key}={value}" for key, value in read.items()) or "none"
    return (
        f"read   {read_bit}\n"
        f"trace  intent={record.get('intent')}  tools={rendered}  "
        f"policy={policy_bit}  awaiting={record.get('awaiting') or 'none'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
