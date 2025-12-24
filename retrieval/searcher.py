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
) -> tuple[list[dict], dict]:
    """
    Retrieve from Qdrant with timing info
    
    Returns:
        tuple: (chunks, timing_info) where timing_info contains:
            - embed_time_ms: embedding time in milliseconds
            - retrieval_time_ms: qdrant query time in milliseconds
    """

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
    # Chuyển sang SHOULD (OR logic) để linh hoạt hơn, tránh filter quá strict
    for key, val in meta_filters.items():
        if key == "document_number":
            should.append(models.FieldCondition(
                key="document_number",
                match=models.MatchText(text=val)
            ))
        elif key == "doc_type":
            should.append(models.FieldCondition(
                key="doc_type",
                match=models.MatchValue(value=val)
            ))
        elif key == "year":
            should.append(models.FieldCondition(
                key="year",
                match=models.MatchValue(value=val)
            ))
        elif key == "issuer":
            should.append(models.FieldCondition(
                key="issuer",
                match=models.MatchText(text=val)
            ))
        elif key == "title_contains":
            should.append(models.FieldCondition(
                key="title",
                match=models.MatchText(text=val)
            ))
    
    # Log filters để debug
    timing_logger.info(f"Filter build: must={len(must)}, should={len(should)}, must_not={len(must_not)}")

    search_filter = models.Filter(
        must=must or None,
        should=should or None,
        must_not=must_not or None
    )

    # ---------- QUERY QDRANT ----------
    t_retrieval_start = time.time()
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        query_filter=search_filter,
        limit=top_k * 3,
        with_payload=True
    ).points
    retrieval_time = time.time() - t_retrieval_start
    
    # Log kết quả retrieval
    timing_logger.info(f"Qdrant returned {len(results)} chunks in {retrieval_time:.4f}s")
    if len(results) == 0:
        timing_logger.warning(f"NO RESULTS! query='{query}', filters={meta_filters}, exclude_ids={len(exclude_ids)}")

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
    
    # Prepare timing info
    timing_info = {
        'embed_time_ms': embed_time * 1000,
        'retrieval_time_ms': retrieval_time * 1000
    }
    
    return boosted[:top_k], timing_info


# ===================== DEBUG TEST =====================
if __name__ == "__main__":
    print("=== TEST SEARCHER ===")
    test_queries = [
        "Thông tư 38/2021",
        "Luật 50/2014/QH13",
        "quy định về an toàn giao thông",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        chunks, timing = retrieve_from_qdrant(query, top_k=5, exclude_ids=set())
        print(f"⏱️  Embed: {timing['embed_time_ms']:.0f}ms | Retrieval: {timing['retrieval_time_ms']:.0f}ms")
        print(f"📦 Found {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            meta = chunk['metadata']
            print(f"  {i}. [{chunk['final_score']:.3f}] {meta.get('document_number')} | {meta.get('title', '')[:60]}...")
    
    print("\n✅ Test completed!")
