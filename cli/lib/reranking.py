import os
from time import sleep

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-3-27b-it"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        score_text = (response.text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]

import json


def llm_rerank_batch(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    if not documents:
        return []

    # Build doc list string
    doc_list = []
    for doc in documents:
        doc_list.append(
            f"{doc['id']}: {doc.get('title', '')} - {doc.get('document', '')}"
        )

    doc_list_str = "\n".join(doc_list)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""

    response = client.models.generate_content(model=model, contents=prompt)

    text = (response.text or "").strip()

    try:
        ranked_ids = json.loads(text)
    except Exception:
        # fallback: return original order if parsing fails
        return documents[:limit]

    # Map doc_id → doc
    doc_map = {doc["id"]: doc for doc in documents}

    reranked = []
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in doc_map:
            doc = {**doc_map[doc_id], "batch_rank": rank}
            reranked.append(doc)

    return reranked[:limit]

def cross_encoder_rerank(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    if not documents:
        return []

    # Build query-doc pairs
    pairs = []
    for doc in documents:
        pairs.append(
            [query, f"{doc.get('title', '')} - {doc.get('document', '')}"]
        )

    # Create model (only once per query)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

    # Predict scores
    scores = cross_encoder.predict(pairs)

    # Attach scores to docs
    scored_docs = []
    for doc, score in zip(documents, scores):
        scored_docs.append({**doc, "cross_encoder_score": float(score)})

    # Sort by score DESC
    scored_docs.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

    return scored_docs[:limit]

def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    elif method == "batch":
        return llm_rerank_batch(query, documents, limit)
    elif method == "cross_encoder":
        return cross_encoder_rerank(query, documents, limit)
    else:
        return documents[:limit]


