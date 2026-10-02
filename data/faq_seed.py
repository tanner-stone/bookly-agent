"""Load FAQ markdown, embed it with Voyage, and store vectors for Atlas search.

The process computes the vectors. The database only stores and compares them.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from pymongo.operations import SearchIndexModel

from tools.faq_search import FAQ_DIMENSIONS, FAQ_INDEX, VoyageEmbedder

FAQ_DIR = Path(__file__).resolve().parent / "faq"


def load_faq_documents() -> list[dict]:
    docs = []
    for path in sorted(FAQ_DIR.glob("*.md")):
        title, body = _split(path.read_text())
        docs.append({"_id": path.stem, "title": title, "body": body})
    if len(docs) < 12:
        raise RuntimeError(f"expected at least 12 FAQ articles, found {len(docs)}")
    return docs


def seed_faq(database) -> str:
    if not os.environ.get("VOYAGE_API_KEY"):
        return "FAQ embeddings skipped (VOYAGE_API_KEY is not set)"
    docs = load_faq_documents()
    embedder = VoyageEmbedder()
    vectors = embedder.embed(
        [f"{doc['title']}\n\n{doc['body']}" for doc in docs],
        "document",
    )
    if any(len(vector) != FAQ_DIMENSIONS for vector in vectors):
        raise RuntimeError("Voyage returned a vector with an unexpected length")
    database.faq.delete_many({})
    database.faq.insert_many(
        {**doc, "embedding": vector} for doc, vector in zip(docs, vectors, strict=True)
    )
    _ensure_index(database.faq)
    _wait_until_queryable(database.faq)
    return f"Embedded {len(docs)} FAQ articles"


def _split(text: str) -> tuple[str, str]:
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise RuntimeError("FAQ articles start with a markdown title")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).strip()
    if not title or not body:
        raise RuntimeError("FAQ articles need a title and a body")
    return title, body


def _ensure_index(collection) -> None:
    existing = list(collection.list_search_indexes())
    if any(index.get("name") == FAQ_INDEX for index in existing):
        return
    collection.create_search_index(
        SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": FAQ_DIMENSIONS,
                        "similarity": "cosine",
                    }
                ]
            },
            name=FAQ_INDEX,
            type="vectorSearch",
        )
    )


def _wait_until_queryable(collection, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        for index in collection.list_search_indexes():
            if index.get("name") == FAQ_INDEX and (
                index.get("queryable") or index.get("status") == "READY"
            ):
                return
        time.sleep(2)
    raise TimeoutError(f"{FAQ_INDEX} was not queryable within {timeout_seconds}s")
