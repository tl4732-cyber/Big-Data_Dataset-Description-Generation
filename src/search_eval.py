"""Compare portal descriptions with the AutoDDG search description using BM25."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from src.config import FACT_ID, PROCESSED, SEARCH_QUERIES, read_json, write_json

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def load_catalog() -> list[dict]:
    path = PROCESSED / "catalog.json"
    if not path.exists():
        raise FileNotFoundError("catalog.json is missing. Run the Milestone 1 catalog cell first.")
    datasets = read_json(path)["datasets"]
    usable = [row for row in datasets if row.get("human_description")]
    if len(usable) < 2:
        raise RuntimeError("Need at least two portal descriptions before BM25 is meaningful.")
    return usable


def _rank_table(documents: list[dict], text_field: str, queries: list[dict]) -> dict:
    corpus = [tokenize(doc[text_field]) for doc in documents]
    if any(len(tokens) == 0 for tokens in corpus):
        raise RuntimeError(f"A document in {text_field} tokenized to nothing.")
    engine = BM25Okapi(corpus)
    ids = [doc["id"] for doc in documents]
    per_query = []
    reciprocal = []
    hits = 0
    for item in queries:
        if item["relevant_id"] not in ids:
            continue
        scores = engine.get_scores(tokenize(item["query"]))
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), ids[index]))
        ranked_ids = [ids[index] for index in order]
        rank = ranked_ids.index(item["relevant_id"]) + 1
        reciprocal.append(1.0 / rank)
        hits += int(rank == 1)
        per_query.append(
            {
                "query": item["query"],
                "relevant_id": item["relevant_id"],
                "rank": rank,
                "top_id": ranked_ids[0],
                "scores": {ids[index]: round(float(scores[index]), 4) for index in order},
            }
        )
    n = len(per_query)
    return {
        "index": text_field,
        "n_documents": len(documents),
        "n_queries": n,
        "mrr": None if n == 0 else round(sum(reciprocal) / n, 4),
        "recall_at_1": None if n == 0 else round(hits / n, 4),
        "queries": per_query,
    }


def _queries_for(datasets: list[dict]) -> list[dict]:
    present = {row["id"] for row in datasets}
    return [item for item in SEARCH_QUERIES if item["relevant_id"] in present]


def evaluate_retrieval() -> dict:
    catalog = load_catalog()
    queries = _queries_for(catalog)
    human_docs = [
        {
            "id": row["id"],
            "human": f"{row['name']}. {row['human_description']}",
        }
        for row in catalog
    ]
    human = _rank_table(human_docs, "human", queries)

    sfd_result = None
    description_path = PROCESSED / "autoddg_descriptions.json"
    if description_path.exists():
        sfd = read_json(description_path)["sfd"]
        swapped = []
        for doc in human_docs:
            text = sfd if doc["id"] == FACT_ID else doc["human"]
            swapped.append({"id": doc["id"], "sfd_swap": text})
        sfd_result = _rank_table(swapped, "sfd_swap", queries)

    report = {
        "method": (
            "Pilot BM25 corpus. Each document is one dataset. "
            "The second index replaces only the 311 portal description with the AutoDDG SFD. "
            "Absolute BM25 scores are not compared across indexes. MRR and Recall@1 are."
        ),
        "frozen_queries": True,
        "human_portal_index": human,
        "autoddg_sfd_index": sfd_result,
    }
    write_json(PROCESSED / "eval_bm25.json", report)
    return report
