# retrieval/searcher.py
from qdrant_client import models
from models.clients import get_remote_embedding, get_models
from retrieval.metadata_extractor import extract_metadata_from_query
from config import *
from utils.logger import timing_logger
import time
import os
import logging

# ===================== SETUP EMBED LOG =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

EMBED_LOG_PATH = os.path.join(LOG_DIR, "embed.log")

embed_logger = logging.getLogger("embed_logger")
embed_logger.setLevel(logging.INFO)

if not embed_logger.handlers:
    fh = logging.FileHandler(EMBED_LOG_PATH, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    fh.setFormatter(formatter)
    embed_logger.addHandler(fh)

# ===================== INIT CLIENTS =====================
qdrant_client, hf_client = get_models()

def retrieve_from_qdrant(
    query: str,
    top_k: int,
    exclude_ids: set,
    force_no_filter: bool = False
) -> list[dict]:

    # ---------- EMBEDDING ----------
    t0 = time.time()
    query_vec = get_remote_embedding(query)
    embed_time = time.time() - t0

    timing_logger.info(f"Embedding time: {embed_time:.4f}s")

    embed_logger.info(
        f"query='{query[:200]}' | dim={len(query_vec)} | time={embed_time:.4f}s | source=hf_space"
    )

    # ---------- METADATA ----------
    meta_filters, general_keywords = extract_metadata_from_query(query)
    timing_logger.info(f"Extracted filters: {meta_filters}")

    if force_no_filter:
        meta_filters = {}
        general_keywords = []
        timing_logger.info("force_no_filter=True → bỏ toàn bộ metadata filter")

    must = []
    should = []
    must_not = [
        models.FieldCondition(key="chunk_id", match=models.MatchValue(value=cid))
        for cid in exclude_ids
    ]

    # ---------- FILTER BUILD ----------
    for key, val in meta_filters.items():
        if key == "document_number":
            must.append(models.FieldCondition(
                key="document_number",
                match=models.MatchText(text=val)
            ))
        elif key == "doc_type":
            must.append(models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=val)
            ))
        elif key == "year":
            must.append(models.FieldCondition(
                key="year",
                match=models.MatchValue(value=val)
            ))
        elif key == "issuer":
            must.append(models.FieldCondition(
                key="issuer",
                match=models.MatchText(text=val)
            ))
        elif key == "title_contains":
            must.append(models.FieldCondition(
                key="title",
                match=models.MatchText(text=val)
            ))

    search_filter = models.Filter(
        must=must or None,
        should=should or None,
        must_not=must_not or None
    )

    # ---------- QUERY QDRANT ----------
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=search_filter,
        limit=top_k * 3,
        with_payload=True
    ).points

    # ---------- SOFT HYBRID ----------
    boosted = []
    for hit in results:
        payload = hit.payload
        score = hit.score * VECTOR_SCORE_MULTIPLIER
        boost = 0.0

        if ENABLE_SOFT_HYBRID:
            for key, weight in METADATA_BOOST_WEIGHTS.items():
                if key in meta_filters:
                    filter_val = meta_filters[key]
                    payload_val = payload.get(key.replace("_contains", ""))
                    if payload_val and str(filter_val).lower() in str(payload_val).lower():
                        boost += weight

            if general_keywords:
                source = payload.get("source_file", "").lower()
                matched = sum(1 for kw in general_keywords if kw in source)
                boost += matched * METADATA_BOOST_WEIGHTS.get("source_file_keyword", 0.4)

        final_score = score + boost

        boosted.append({
            "text": payload["text"],
            "final_score": final_score,
            "boost_score": boost,
            "vector_score": score,
            "metadata": {
                "chunk_id": payload.get("chunk_id"),
                "source_file": payload.get("source_file"),
                "document_number": payload.get("document_number"),
                "title": payload.get("title"),
                "doc_type": payload.get("doc_type"),
                "issuer": payload.get("issuer"),
                "year": payload.get("year"),
            }
        })

    boosted.sort(key=lambda x: x["final_score"], reverse=True)
    return boosted[:top_k]
