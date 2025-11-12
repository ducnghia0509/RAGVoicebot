# models/clients.py
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from huggingface_hub import InferenceClient
import torch
import time
from utils.logger import timing_logger
import os
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

_embedder = None
_qdrant_client = None
_hf_client = None

def get_models():
    global _embedder, _qdrant_client, _hf_client
    if _embedder is not None:
        return _embedder, _qdrant_client, _hf_client

    t0 = time.time()
    _embedder = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    timing_logger.info(f"Loaded embedding model: {time.time() - t0:.2f}s")

    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=60)
    timing_logger.info("Connected to Qdrant")

    _hf_client = InferenceClient(token=os.getenv("HF_TOKEN"), model=HF_MODEL)
    timing_logger.info(f"HF InferenceClient ready: {HF_MODEL}")

    return _embedder, _qdrant_client, _hf_client