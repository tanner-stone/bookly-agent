"""A stand-in for the chat model. Pytest must not call xAI.

Each test scripts what the model would read for each customer line. The graph
under test still checks that read before it acts.
"""

from collections.abc import Callable

from langchain_core.messages import AIMessage

Read = dict | Callable[[dict, Callable], dict]


class ScriptedLanguage:
    def __init__(self, script: dict[str, Read] | None = None, tool_behavior: str = "text"):
        self.script = dict(script or {})
        self.tool_behavior = tool_behavior
        self.views: list[dict] = []

    @property
    def understand_calls(self) -> int:
        return len(self.views)

    def understand(self, view: dict, conversation: list, run_read) -> dict:
        self.views.append(view)
        latest = ""
        for message in reversed(conversation):
            if getattr(message, "type", None) == "human":
                latest = message.content
                break
        entry = self.script.get(latest, {"goal": "chat"})
        if callable(entry):
            entry = entry(view, run_read)
        return {"candidate_ids": [], "confirmation": "none", **entry}

    def reply(self, state: dict) -> AIMessage:
        if self.tool_behavior != "text" and state.get("tool_steps", 0) == 0:
            call = self._tool_call(state)
            if call is not None:
                return AIMessage(content="", tool_calls=[call])
        kind = (state.get("directive") or {}).get("kind") or "clarify"
        return AIMessage(content=kind)

    def _tool_call(self, state: dict) -> dict | None:
        if self.tool_behavior == "call_get_order":
            order = (state.get("directive") or {}).get("order") or {}
            order_id = order.get("order_id") or state.get("selected_order_id")
            return {
                "name": "get_order",
                "args": {"order_id": order_id},
                "id": "call-get",
                "type": "tool_call",
            }
        if self.tool_behavior == "call_issue_refund":
            return {
                "name": "issue_refund",
                "args": {},
                "id": "call-refund",
                "type": "tool_call",
            }
        if self.tool_behavior == "call_other_order":
            return {
                "name": "get_order",
                "args": {"order_id": "BK-10239"},
                "id": "call-other",
                "type": "tool_call",
            }
        return None


class StaticFaq:
    """In-memory FAQ stand-in. Tests must not call Voyage."""

    def __init__(self, grounded: bool = False, error: str | None = None):
        self.grounded = grounded
        self.error = error
        self.queries: list[str] = []

    def search(self, query: str):
        from tools.faq_search import FaqHit, FaqResult

        self.queries.append(query)
        if self.error:
            return FaqResult(hits=[], grounded=False, best_score=None, error=self.error)
        if not self.grounded:
            return FaqResult(hits=[], grounded=False, best_score=0.2)
        hit = FaqHit(
            title="Shipping times and cost",
            body="Orders inside the United States ship in 2 to 5 business days.",
            score=0.91,
            article_id="shipping-times",
        )
        return FaqResult(hits=[hit], grounded=True, best_score=0.91)
