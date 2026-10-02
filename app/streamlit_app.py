"""Bookly support chat.

Sign in with the email box. The transcript never asks for the address.
One browser session is one thread_id, shared with the CLI and with voice.
"""

from __future__ import annotations

import base64
import hashlib
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from agent.graph import build_graph
from agent.language import XAILanguage
from agent.session import sign_in_feedback
from app.tracking import render_tracking
from app.voice_widget import listen_status, voice_bar
from policy.refund_policy import load_refund_policy
from tools.faq_search import NullFaq, VectorFaqIndex, VoyageEmbedder
from tools.store import MongoStore
from voice.xai_speech import SpeechError, synthesize, transcribe

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

CLOTH = ("#1f3d34", "#5c3b32", "#24324a", "#6e4c2f")
STATUS_LABELS = {
    "delivered": "Delivered",
    "in_transit": "On the way",
    "cancelled": "Canceled",
    "refunded": "Refunded",
}

st.set_page_config(page_title="Bookly", layout="centered")
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,640&family=Nunito+Sans:wght@400;600&display=swap');
      html, body, [class*="css"] {
        font-family: "Nunito Sans", sans-serif;
      }
      h1, h2, h3, .wordmark {
        font-family: Fraunces, serif;
        font-weight: 560;
        letter-spacing: -0.02em;
      }
      [data-testid="stAppViewContainer"] {
        background: #f4efe6;
      }
      [data-testid="stHeader"] {
        background: transparent;
      }
      .wordmark {
        font-size: 3rem;
        line-height: 1;
        margin: 0.4rem 0 0;
        color: #1c1915;
      }
      .rule {
        width: 4.5rem;
        height: 2px;
        background: #1f3d34;
        margin: 0.7rem 0 0.4rem;
      }
      .tagline, .desk {
        color: #5c564c;
        margin-top: 0;
      }
      .cover {
        min-height: 7.5rem;
        padding: 0.9rem;
        color: #f4efe6;
        font-family: Fraunces, serif;
        font-size: 1.15rem;
        line-height: 1.25;
      }
      .card-meta {
        color: #5c564c;
        font-size: 0.85rem;
        margin: 0.15rem 0;
      }
      [data-testid="stHorizontalBlock"] {
        overflow-x: auto;
        flex-wrap: nowrap;
        gap: 0.8rem;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #fbf8f3;
        min-width: 14rem;
      }
      .st-key-mode-bar {
        position: fixed;
        top: 0.35rem;
        left: 0;
        right: 0;
        z-index: 1000001;
        background: #f4efe6;
        padding-bottom: 0.2rem;
        border-bottom: 1px solid #e4ddd0;
      }
      .mode-spacer {
        height: 3.1rem;
      }
      .working {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #5c564c;
      }
      .book {
        width: 22px;
        height: 16px;
        border: 1.5px solid #1f3d34;
        border-radius: 2px 3px 3px 2px;
        position: relative;
        background: #fbf8f3;
        flex: none;
      }
      .book::before {
        content: "";
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 1.5px;
        background: #1f3d34;
      }
      .book::after {
        content: "";
        position: absolute;
        top: 2px;
        bottom: 2px;
        right: 2px;
        width: 8px;
        background: #ebe4d6;
        transform-origin: left center;
        animation: turn-page 1.1s ease-in-out infinite;
      }
      @keyframes turn-page {
        0% { transform: scaleX(1); }
        50% { transform: scaleX(0.15); }
        100% { transform: scaleX(1); }
      }
      .book.shut::after {
        animation: none;
        transform: scaleX(1);
      }
      .ended {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.2rem 0 0.4rem;
        color: #5c564c;
      }
      .ended p {
        margin: 0;
      }
      [class*="-sources"] [data-testid="stHorizontalBlock"],
      [class*="-sources"] {
        flex-wrap: wrap;
        gap: 0.4rem;
      }
      [class*="-sources"] button {
        border-radius: 999px;
        font-size: 0.82rem;
        padding: 0.1rem 0.75rem;
        min-height: 0;
      }
      .st-key-faq-panel {
        position: fixed;
        top: 0;
        right: 0;
        bottom: 0;
        width: min(26rem, 92vw);
        overflow-y: auto;
        z-index: 1000002;
        background: #fbf8f3;
        border-left: 1px solid #e4ddd0;
        box-shadow: -12px 0 32px rgba(28, 25, 21, 0.12);
        padding: 1.4rem 1.5rem 2rem;
        animation: panel-in 0.18s ease-out;
      }
      @keyframes panel-in {
        from { transform: translateX(1.5rem); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      .panel-kicker {
        margin: 0;
        color: #5c564c;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def resources():
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set")
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    client.admin.command("ping")
    db_name = os.environ.get("MONGODB_DB", "bookly")
    database = client[db_name]
    policy = load_refund_policy()
    faq = (
        VectorFaqIndex(database, VoyageEmbedder())
        if os.environ.get("VOYAGE_API_KEY")
        else NullFaq()
    )
    graph = build_graph(
        XAILanguage(),
        MongoStore(database),
        MongoDBSaver(client, db_name=db_name),
        faq=faq,
    )
    return graph, MongoStore(database), policy


def main() -> None:
    try:
        graph, store, policy = resources()
    except Exception as exc:
        _brand()
        st.error(f"Bookly is not available right now. {type(exc).__name__}: {exc}")
        return

    if "email_failures" not in st.session_state:
        st.session_state.email_failures = 0
    if "mode" not in st.session_state:
        st.session_state.mode = "chat"

    if "customer" not in st.session_state:
        _brand()
        _sign_in_box(store, policy)
        return

    customer = st.session_state.customer
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    if not (values.get("messages") or []):
        graph.invoke(
            {"channel": "chat", "customer_id": customer["_id"]},
            config,
        )
        st.rerun()
    snapshot = graph.get_state(config)
    values = snapshot.values or {}
    directive = values.get("directive") or {}
    hold_listen = bool(directive.get("closed") or directive.get("needs_review"))
    with st.container(key="mode-bar"):
        heard = voice_bar(
            mode=st.session_state.mode,
            play_b64=st.session_state.get("play_b64") or "",
            play_token=st.session_state.get("play_token") or "",
            listen_token=st.session_state.get("listen_token") or "",
            listen_after=not hold_listen,
        )
    st.markdown('<div class="mode-spacer"></div>', unsafe_allow_html=True)
    _apply_voice_event(heard, hold_listen)
    _brand()
    st.caption(f"{customer['name']} · {customer['email']}")
    if st.button("Use a different email"):
        for key in (
            "customer",
            "thread_id",
            "play_b64",
            "play_token",
            "listen_token",
            "heard_token",
            "mode_token",
            "mode",
            "tracking_order_id",
            "faq_article",
            "draft",
            "draft_shown",
        ):
            st.session_state.pop(key, None)
        st.session_state.email_failures = 0
        st.rerun()

    messages = values.get("messages") or []
    trace = values.get("trace") or []
    draft = st.session_state.get("draft")
    if draft and st.session_state.get("draft_shown"):
        state = _turn(graph, config, customer["_id"], draft["text"], draft["channel"])
        st.session_state.draft = None
        st.session_state.draft_shown = False
        if draft["channel"] == "voice":
            closing = bool((state.get("directive") or {}).get("closed"))
            closing = closing or bool((state.get("directive") or {}).get("needs_review"))
            _remember_speech(_last_assistant(state["messages"]), uuid.uuid4().hex)
            if not closing:
                st.session_state.listen_token = st.session_state.play_token
        st.rerun()

    choice, star_pick = _render_messages(messages, values.get("awaiting"))
    _article_panel()
    listen_status()
    channel = st.session_state.mode
    if choice:
        _queue_turn(choice, channel)
        _show_working(choice, quiet=star_pick)
        st.rerun()
    if draft and not st.session_state.get("draft_shown"):
        _show_working(draft["text"])
        st.session_state.draft_shown = True
        st.rerun()

    closed = bool(directive.get("closed"))
    needs_review = bool(directive.get("needs_review")) and not closed
    if needs_review:
        with st.form("review-note"):
            note = st.text_area("Leave a note about what went wrong")
            submitted = st.form_submit_button("Send note")
        if submitted and note.strip():
            _queue_turn(note.strip(), channel)
            _show_working(note.strip())
            st.rerun()
    if closed or needs_review:
        _ended()
    elif channel == "chat":
        prompt = st.chat_input("How can I assist you today?")
        if prompt:
            _queue_turn(prompt, "chat")
            _show_working(prompt)
            st.rerun()

    with st.expander("Agent trace"):
        if not trace:
            st.caption("No turns yet.")
        for index, record in enumerate(trace, start=1):
            st.markdown(f"**Turn {index}**")
            st.text(_format_trace(record))


def _apply_voice_event(heard, hold_listen: bool) -> None:
    if not isinstance(heard, dict):
        return
    token = str(heard.get("token") or "")
    if heard.get("kind") == "mode" and token and token != st.session_state.get("mode_token"):
        st.session_state.mode_token = token
        st.session_state.mode = "voice" if heard.get("mode") == "voice" else "chat"
        if st.session_state.mode == "voice":
            st.session_state.play_b64 = ""
            st.session_state.play_token = ""
            st.session_state.listen_token = "" if hold_listen else token
        st.rerun()
    if heard.get("kind") != "audio" or not heard.get("wav_b64"):
        return
    if not token or token == st.session_state.get("heard_token"):
        return
    st.session_state.heard_token = token
    if st.session_state.mode != "voice" or hold_listen:
        return
    raw = base64.b64decode(heard["wav_b64"])
    digest = hashlib.sha256(raw).hexdigest()
    if digest == st.session_state.get("last_audio"):
        return
    st.session_state.last_audio = digest
    with st.spinner("One moment."):
        try:
            text = transcribe(raw)
        except SpeechError as exc:
            st.error(str(exc))
            st.session_state.listen_token = token + "-again"
        else:
            st.session_state.draft = {"text": text, "channel": "voice"}
            st.session_state.draft_shown = False
    st.rerun()


def _brand() -> None:
    st.markdown('<p class="wordmark">Bookly</p><div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">A quieter desk for readers</p>', unsafe_allow_html=True)


def _sign_in_box(store, policy) -> None:
    st.markdown('<p class="desk">Sign in with the email on your account.</p>', unsafe_allow_html=True)
    feedback = sign_in_feedback(
        st.session_state.email_failures,
        policy.support_email,
        policy.support_phone,
    )
    if feedback:
        st.error(feedback)
    with st.form("sign-in"):
        email = st.text_input("Email", placeholder="tanner.stone@bookly.example")
        submitted = st.form_submit_button("Continue")
    if submitted:
        customer = store.get_customer_by_email(email)
        if customer is None:
            st.session_state.email_failures += 1
            st.rerun()
        st.session_state.customer = {
            "_id": customer["_id"],
            "name": customer["name"],
            "email": customer["email"],
        }
        st.session_state.thread_id = uuid.uuid4().hex
        st.session_state.email_failures = 0
        st.rerun()


def _order_cards(orders: list[dict], key_prefix: str) -> str | None:
    if not orders:
        return None
    choice = None
    shown = None
    with st.container(horizontal=True, wrap=False):
        for index, order in enumerate(orders):
            with st.container(border=True):
                cloth = CLOTH[index % len(CLOTH)]
                st.markdown(
                    f'<div class="cover" style="background:{cloth}">{order["title"]}</div>',
                    unsafe_allow_html=True,
                )
                if order.get("author"):
                    st.markdown(f'<p class="card-meta">{order["author"]}</p>', unsafe_allow_html=True)
                label = STATUS_LABELS.get(order.get("status") or "", order.get("status") or "")
                st.markdown(
                    f'<p class="card-meta">{label} · ordered {order.get("ordered_label") or ""}</p>',
                    unsafe_allow_html=True,
                )
                tracking = order.get("tracking") or {}
                where_key = f"{key_prefix}-where-{order['order_id']}"
                if tracking.get("points") and st.button("Where is it", key=where_key):
                    current = st.session_state.get("tracking_order_id")
                    st.session_state.tracking_order_id = (
                        None if current == order["order_id"] else order["order_id"]
                    )
                if st.session_state.get("tracking_order_id") == order["order_id"]:
                    shown = tracking
                if st.button("This order", key=f"{key_prefix}-pick-{order['order_id']}"):
                    choice = order["order_id"]
    if shown:
        render_tracking(shown)
    return choice


def _sources(articles: list[dict], key_prefix: str) -> None:
    open_article = st.session_state.get("faq_article") or {}
    with st.container(horizontal=True, key=f"{key_prefix}-sources"):
        for article in articles:
            article_id = article.get("id") or article.get("title")
            showing = open_article.get("id") == article_id
            if st.button(
                article.get("title") or "Help article",
                key=f"{key_prefix}-source-{article_id}",
                icon=":material/menu_book:",
                type="primary" if showing else "secondary",
                help="Open the help article",
            ):
                st.session_state.faq_article = (
                    None if showing else {"id": article_id, **article}
                )
                st.rerun()


def _article_panel() -> None:
    article = st.session_state.get("faq_article")
    if not article:
        return
    with st.container(key="faq-panel"):
        top = st.container(horizontal=True, vertical_alignment="center")
        top.markdown('<p class="panel-kicker">Help article</p>', unsafe_allow_html=True)
        if top.button("Close", key="faq-panel-close", icon=":material/close:", type="tertiary"):
            st.session_state.faq_article = None
            st.rerun()
        st.markdown(f"### {article.get('title') or ''}")
        st.markdown(article.get("body") or "")


def _stars(key_prefix: str, selected: int | None) -> str | None:
    picked = None
    columns = st.columns(5)
    for number, column in enumerate(columns, start=1):
        label = "★" * number
        if selected is not None:
            column.button(
                label,
                key=f"{key_prefix}-star-{number}",
                disabled=True,
                type="primary" if number == selected else "secondary",
            )
            continue
        if column.button(label, key=f"{key_prefix}-star-{number}"):
            picked = str(number)
    return picked


def _ended() -> None:
    st.markdown(
        '<div class="ended"><div class="book shut"></div><p>This conversation has ended.</p></div>',
        unsafe_allow_html=True,
    )
    if st.button("New conversation"):
        st.session_state.thread_id = uuid.uuid4().hex
        for key in (
            "play_b64",
            "play_token",
            "listen_token",
            "heard_token",
            "draft",
            "draft_shown",
            "tracking_order_id",
            "faq_article",
            "last_audio",
        ):
            st.session_state.pop(key, None)
        if st.session_state.mode == "voice":
            st.session_state.listen_token = uuid.uuid4().hex
        st.rerun()


def _queue_turn(text: str, channel: str) -> None:
    st.session_state.draft = {"text": text, "channel": channel}
    st.session_state.draft_shown = True


def _show_working(text: str, quiet: bool = False) -> None:
    if not quiet:
        with st.chat_message("user"):
            st.write(text)
    with st.chat_message("assistant"):
        st.markdown(
            '<div class="working"><div class="book"></div><span>One moment.</span></div>',
            unsafe_allow_html=True,
        )


def _remember_speech(text: str, token: str) -> None:
    if not text:
        return
    try:
        audio = synthesize(text)
    except SpeechError as exc:
        st.session_state.play_b64 = ""
        st.error(str(exc))
        return
    st.session_state.play_b64 = base64.b64encode(audio).decode()
    st.session_state.play_token = token


def _turn(graph, config, customer_id: str, text: str, channel: str) -> dict:
    return graph.invoke(
        {
            "messages": [HumanMessage(content=text)],
            "channel": channel,
            "customer_id": customer_id,
        },
        config,
    )


def _last_assistant(messages) -> str:
    for message in reversed(messages):
        if getattr(message, "type", None) != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _message_exhibit(message) -> dict | None:
    extra = getattr(message, "additional_kwargs", None) or {}
    exhibit = extra.get("exhibit")
    return exhibit if isinstance(exhibit, dict) else None


def _score_after(messages, index: int) -> int | None:
    for message in messages[index + 1 :]:
        exhibit = _message_exhibit(message) or {}
        if exhibit.get("kind") == "ask_rating":
            return None
        if exhibit.get("kind") == "rated":
            return exhibit.get("stars")
    return None


def _is_star_click(messages, index: int) -> bool:
    """A bare score that the graph saved as a rating. The stars show it instead."""

    content = getattr(messages[index], "content", "")
    for message in messages[index + 1 :]:
        if getattr(message, "type", None) == "human":
            return False
        exhibit = _message_exhibit(message) or {}
        if exhibit.get("kind") == "rated":
            return isinstance(content, str) and content.strip() == str(exhibit.get("stars"))
    return False


def _render_messages(messages, awaiting: str | None) -> tuple[str | None, bool]:
    last_rating = None
    for index, message in enumerate(messages):
        exhibit = _message_exhibit(message)
        if exhibit and exhibit.get("kind") == "ask_rating":
            last_rating = index
    choice = None
    star_pick = False
    for index, message in enumerate(messages):
        role = getattr(message, "type", None)
        content = getattr(message, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "human":
            if _is_star_click(messages, index):
                continue
            with st.chat_message("user"):
                st.write(content)
            continue
        if role != "ai" or getattr(message, "tool_calls", None):
            continue
        with st.chat_message("assistant"):
            st.write(content)
        exhibit = _message_exhibit(message)
        if not exhibit:
            continue
        prefix = f"m{index}"
        if exhibit.get("kind") in {"choose_order", "confirm_order"}:
            picked = _order_cards(exhibit.get("orders") or [], prefix)
            if picked:
                choice = picked
        elif exhibit.get("kind") == "order_status":
            render_tracking((exhibit.get("order") or {}).get("tracking"))
        elif exhibit.get("kind") == "faq_sources":
            _sources(exhibit.get("articles") or [], prefix)
        elif exhibit.get("kind") == "ask_rating":
            selected = _score_after(messages, index)
            if selected is None and not (index == last_rating and awaiting == "rating"):
                continue
            picked = _stars(prefix, selected)
            if picked:
                choice = picked
                star_pick = True
    return choice, star_pick


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
    policy_bit = (
        f"{policy.get('outcome')} {policy.get('rule_id')}" if policy else "none"
    )
    read = record.get("read") or {}
    read_bit = " ".join(f"{key}={value}" for key, value in read.items()) or "none"
    return (
        f"read: {read_bit}\n"
        f"intent={record.get('intent')}  tools={rendered}  "
        f"policy={policy_bit}  awaiting={record.get('awaiting') or 'none'}"
    )


main()
