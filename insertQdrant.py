# insertQdrant.py – PHIÊN BẢN HOÀN CHỈNH (Qdrant 1.15.1)
import os
import pickle
import json
import re
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ===================== CẤU HÌNH =====================
from dotenv import load_dotenv
load_dotenv()
EMBEDDED_FILE = "embedded_chunks.pkl"
QDRANT_HOST = "914f5b70-3424-49a0-841f-80c4a1d7dac8.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_PORT = 6333
QDRANT_API_KEY = os.getenv("QDRANT_TOKEN")
COLLECTION_NAME = "vietnamese_legal_chunks"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"

# ===================== KHỞI TẠO =====================
print("Đang tải dữ liệu...")
if not os.path.exists(EMBEDDED_FILE):
    raise FileNotFoundError(f"Không tìm thấy: {EMBEDDED_FILE}")

with open(EMBEDDED_FILE, 'rb') as f:
    all_chunks = pickle.load(f)
print(f"Load {len(all_chunks)} chunks.")

model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
DIMENSION = len(model.encode("test").tolist())
print(f"Vector dim: {DIMENSION}")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=60)

# ===================== TẠO COLLECTION =====================
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"Xóa collection cũ: {COLLECTION_NAME}")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=DIMENSION, distance=Distance.COSINE),
)
print("Tạo collection thành công!")

# ===================== INSERT =====================
batch_size = 100
points = []

print("Đang insert...")
for idx, chunk in enumerate(tqdm(all_chunks, desc="Insert")):
    meta = chunk.get("metadata", {})
    payload = {
        "text": chunk["text"],
        "chunk_id": meta.get("chunk_id"),
        "source_file": meta.get("source_file", "unknown"),
        "document_number": meta.get("document_number", ""),
        "doc_type": meta.get("document_type", "Không rõ"),
        "issuer": meta.get("authority", "Không rõ"),
        "issue_date": meta.get("issued_date", ""),
        "title": meta.get("title", ""),
    }
    points.append(PointStruct(id=idx, vector=chunk["embedding"], payload=payload))
    if len(points) >= batch_size:
        client.upsert(COLLECTION_NAME, points)
        points = []

if points:
    client.upsert(COLLECTION_NAME, points)
print(f"Insert xong {len(all_chunks)} points.")

# ===================== TẠO INDEX =====================
print("\nTạo index cho hybrid search...")
for field in ["document_number", "title", "text"]:
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=models.PayloadSchemaType.TEXT
        )
        print(f"Index: {field}")
    except:
        pass

# ===================== TEST HYBRID =====================
print("\n--- HYBRID SEARCH TEST ---")
test_queries = ["Thông tư 38/2021", "Luật 50/2014/QH13"]

for query in test_queries:
    print(f"\nTìm: '{query}'")
    vec = model.encode(query, normalize_embeddings=True).tolist()

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=5,
        with_payload=True,
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="document_number",
                    match=models.MatchText(text=query)
                ),
                models.FieldCondition(
                    key="title",
                    match=models.MatchText(text=query)
                ),
            ]
        )
    )


    for i, hit in enumerate(result.points):
        p = hit.payload
        print(f"  {i+1}. [{hit.score:.4f}] {p.get('document_number')} | {p.get('title', '')[:80]}...")
    
# Thêm đoạn này vào cuối file insertQdrant.py
print("\nKIỂM TRA DỮ LIỆU CÓ 'Thông tư 38/2021' KHÔNG?")
found = False
for point in client.scroll(collection_name=COLLECTION_NAME, limit=1000, with_payload=True)[0]:
    if "38/2021" in point.payload.get("document_number", ""):
        print("TÌM THẤY:", point.payload["document_number"], "|", point.payload.get("title", ""))
        found = True
        break
if not found:
    print("KHÔNG TÌM THẤY '38/2021' TRONG DỮ LIỆU!")