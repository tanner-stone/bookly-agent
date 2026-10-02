"""FAQ grounding and the customer-facing articles.

These tests do not call Voyage. They check the score gate and that the
articles still quote the numbers in the refund file.
"""

from pathlib import Path

import httpx
import pytest

from policy.config import FAQ_SCORE_THRESHOLD, VOYAGE_MAX_WAIT_SECONDS, VOYAGE_RETRIES
from policy.refund_policy import load_refund_policy
from tools.faq_search import FaqHit, VectorFaqIndex, VoyageEmbedder, apply_threshold
from data.faq_seed import load_faq_documents

ROOT = Path(__file__).resolve().parents[1]


def test_score_equal_to_the_threshold_is_grounded():
    result = apply_threshold(
        [FaqHit(title="Shipping", body="2 to 5 business days", score=FAQ_SCORE_THRESHOLD)],
        FAQ_SCORE_THRESHOLD,
    )
    assert result.grounded is True
    assert result.hits[0].body == "2 to 5 business days"


def test_score_under_the_threshold_drops_the_article():
    result = apply_threshold(
        [FaqHit(title="Shipping", body="2 to 5 business days", score=FAQ_SCORE_THRESHOLD - 0.01)],
        FAQ_SCORE_THRESHOLD,
    )
    assert result.grounded is False
    assert result.hits == []
    assert result.best_score == FAQ_SCORE_THRESHOLD - 0.01


def test_return_article_quotes_the_yaml_limits():
    policy = load_refund_policy()
    text = (ROOT / "data" / "faq" / "return-policy.md").read_text()
    standard = policy.get_rule("standard_refund").when
    goodwill = policy.get_rule("first_time_goodwill").when
    assert str(int(standard.max_total_usd)) in text
    assert str(standard.max_days_since_delivery) in text
    assert str(int(goodwill.max_total_usd)) in text
    assert str(goodwill.max_days_since_delivery) in text
    assert "one-time" in text.lower()
    contact = (ROOT / "data" / "faq" / "contact.md").read_text()
    assert policy.support_email in contact
    assert policy.support_phone in contact


class _Response:
    def __init__(self, status: int, headers: dict | None = None):
        self.status_code = status
        self.headers = headers or {}
        self.request = httpx.Request("POST", "https://api.voyageai.com/v1/embeddings")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=self.request, response=self)

    def json(self):
        return {"data": [{"index": 0, "embedding": [0.1] * 4}]}


def _voyage(monkeypatch, statuses: list[int], headers: dict | None = None):
    calls = []

    def post(*_args, **_kwargs):
        calls.append(1)
        return _Response(statuses[min(len(calls), len(statuses)) - 1], headers)

    monkeypatch.setattr(httpx, "post", post)
    slept: list[float] = []
    return VoyageEmbedder(api_key="test", sleep=slept.append), calls, slept


def test_a_rate_limit_waits_and_retries(monkeypatch):
    embedder, calls, slept = _voyage(monkeypatch, [429, 429, 200], {"retry-after": "3"})
    assert embedder.embed(["shipping"], "query") == [[0.1] * 4]
    assert len(calls) == 3
    assert slept == [3.0, 3.0]


def test_a_rate_limit_past_the_wait_cap_gives_up(monkeypatch):
    embedder, calls, slept = _voyage(monkeypatch, [429], {"retry-after": "20"})
    with pytest.raises(httpx.HTTPStatusError):
        embedder.embed(["shipping"], "query")
    assert sum(slept) <= VOYAGE_MAX_WAIT_SECONDS
    assert len(calls) <= VOYAGE_RETRIES + 1


def test_a_repeated_question_does_not_call_voyage_again(monkeypatch):
    embedder, calls, _slept = _voyage(monkeypatch, [200])
    embedder.embed(["shipping"], "query")
    embedder.embed(["shipping"], "query")
    assert len(calls) == 1


def test_a_failed_search_reports_the_error():
    class Broken:
        def embed(self, texts, input_type):
            raise RuntimeError("down")

    result = VectorFaqIndex(database=None, embedder=Broken()).search("shipping")
    assert result.grounded is False
    assert result.error == "RuntimeError"


def test_faq_library_has_a_dozen_articles():
    docs = load_faq_documents()
    assert len(docs) >= 12
    assert {doc["_id"] for doc in docs} >= {
        "shipping-times",
        "return-policy",
        "password-reset",
        "ebooks",
        "preorders",
        "gift-cards",
        "contact",
        "signing-in",
    }
