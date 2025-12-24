# models/clients.py
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests
import time
import os
from utils.logger import timing_logger
import logging
from config import *

load_dotenv()

device = "cpu" 

_qdrant_client = None
_hf_client = None
API_URL = "https://hoangnam5904-embedding.hf.space/embed"
logger = logging.getLogger(__name__)
def get_remote_embedding(query: str, max_retries: int = 3, timeout: int = 15) -> list:
    """
    Get embedding from remote service with retry logic and fallback
    """
    API_URL = "https://hoangnam5904-embedding.hf.space/embed"
    # Thử format đơn giản hơn: chỉ gửi text string
    payload = {"text": query}
    
    for attempt in range(max_retries):
        try:
            # Tăng timeout và thêm retry
            resp = requests.post(
                API_URL, 
                json=payload, 
                timeout=timeout,
                headers={"User-Agent": "RAGVoicebot/1.0"}
            )
            
            # Log để debug
            if resp.status_code != 200:
                logger.error(f"Embedding API error {resp.status_code}: {resp.text[:200]}")
            
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list):
                    return result
                elif isinstance(result, dict) and "embedding" in result:
                    return result["embedding"]
                else:
                    logger.warning(f"Unexpected response format from embedding service: {result}")
            
            logger.warning(f"Embedding service returned status {resp.status_code}, attempt {attempt+1}/{max_retries}")
            
        except requests.exceptions.Timeout as e:
            logger.warning(f"Embedding timeout (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                time.sleep(wait_time)
            continue
            
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Embedding connection error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
            
        except Exception as e:
            logger.error(f"Embedding error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    # Fallback: return a dummy embedding or use local model
    logger.error(f"All {max_retries} embedding attempts failed for query: '{query[:50]}...'")
    
    # Option 1: Return zero vector (simple fallback)
    # return [0.0] * 384  # or whatever dimension your model uses
    
    # Option 2: Use a simple local embedding (if available)
    # return get_local_embedding(query)
    
    # Option 3: Raise exception with more context
    raise ConnectionError(f"Failed to get embedding after {max_retries} attempts")

def get_models():
    """Chỉ trả về các client cần thiết, không có embedder local"""
    global _qdrant_client, _hf_client
    t0 = time.time()
    QDRANT_API_KEY = os.getenv("QDRANT_TOKEN")
    _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY, timeout=60)
    timing_logger.info("Connected to Qdrant")

    _hf_client = InferenceClient(token=os.getenv("HF_TOKEN"), model=HF_MODEL)
    timing_logger.info(f"HF InferenceClient ready: {HF_MODEL}")

    # Chỉ trả về 2 clients, không có embedder
    return _qdrant_client, _hf_client