"""FAQ retrieval.

Voyage embeds the query in this process. Atlas compares it to vectors stored
at seed time. A score under the config threshold is not passed to the model.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from policy.config import FAQ_SCORE_THRESHOLD, VOYAGE_MAX_WAIT_SECONDS, VOYAGE_RETRIES

FAQ_INDEX = "faq_vector"
FAQ_DIMENSIONS = 1024


@dataclass(frozen=True)
class FaqHit:
    title: str
    body: str
    score: float
    article_id: str = ""


@dataclass(frozen=True)
class FaqResult:
    hits: list[FaqHit]
    grounded: bool
    best_score: float | None
    # Set when the search itself failed, so a miss is not mistaken for "no article".
    error: str | None = None


def apply_threshold(hits: list[FaqHit], threshold: float = FAQ_SCORE_THRESHOLD) -> FaqResult:
    """Inclusive: a score equal to the threshold is grounded. Bodies below it are dropped."""

    if not hits:
        return FaqResult(hits=[], grounded=False, best_score=None)
    ordered = sorted(hits, key=lambda hit: hit.score, reverse=True)
    best = ordered[0].score
    if best < threshold:
        return FaqResult(hits=[], grounded=False, best_score=best)
    kept = [hit for hit in ordered if hit.score >= threshold]
    return FaqResult(hits=kept, grounded=True, best_score=best)


class NullFaq:
    def search(self, query: str) -> FaqResult:
        return FaqResult(hits=[], grounded=False, best_score=None)


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        sleep=time.sleep,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("VOYAGE_API_KEY", "")
        self.model = model or os.environ.get("VOYAGE_MODEL", "voyage-4")
        self.sleep = sleep
        self._queries: dict[str, list[float]] = {}

    def embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        # A repeated question costs no request. Rate limits on a small key are tight.
        if input_type == "query" and len(texts) == 1 and texts[0] in self._queries:
            return [self._queries[texts[0]]]
        waited = 0.0
        for attempt in range(VOYAGE_RETRIES + 1):
            response = httpx.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "input": texts,
                    "model": self.model,
                    "input_type": input_type,
                    "output_dimension": FAQ_DIMENSIONS,
                },
                timeout=60,
            )
            if response.status_code == 429 and attempt < VOYAGE_RETRIES:
                delay = _retry_after(response, attempt)
                if waited + delay > VOYAGE_MAX_WAIT_SECONDS:
                    break
                self.sleep(delay)
                waited += delay
                continue
            response.raise_for_status()
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors = [row["embedding"] for row in rows]
            if input_type == "query" and len(texts) == 1:
                self._queries[texts[0]] = vectors[0]
            return vectors
        response.raise_for_status()
        raise RuntimeError("Voyage did not return embeddings")


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    try:
        if header is not None:
            return max(0.5, float(header))
    except ValueError:
        pass
    return float(2 ** (attempt + 1))


class VectorFaqIndex:
    def __init__(self, database, embedder: VoyageEmbedder, threshold: float = FAQ_SCORE_THRESHOLD):
        self.database = database
        self.embedder = embedder
        self.threshold = threshold

    def search(self, query: str) -> FaqResult:
        text = (query or "").strip()
        if not text:
            return FaqResult(hits=[], grounded=False, best_score=None)
        try:
            vector = self.embedder.embed([text], "query")[0]
            rows = list(
                self.database.faq.aggregate(
                    [
                        {
                            "$vectorSearch": {
                                "index": FAQ_INDEX,
                                "path": "embedding",
                                "queryVector": vector,
                                "numCandidates": 40,
                                "limit": 3,
                            }
                        },
                        {
                            "$project": {
                                "title": 1,
                                "body": 1,
                                "score": {"$meta": "vectorSearchScore"},
                            }
                        },
                    ]
                )
            )
        except Exception as exc:
            return FaqResult(hits=[], grounded=False, best_score=None, error=_short_error(exc))
        hits = [
            FaqHit(
                title=row["title"],
                body=row["body"],
                score=float(row["score"]),
                article_id=str(row.get("_id") or ""),
            )
            for row in rows
        ]
        return apply_threshold(hits, self.threshold)


def _short_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"voyage {exc.response.status_code}"
    return type(exc).__name__
