TOP_K=5 # cho bản basic
GGUF_MODEL_PATH = r"C:\Users\DELL\Desktop\Test\LocalProjectPackage\Project\VoiceBotBasic\Gemma-2-IT-Q6_K.gguf"
HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
QUICK_MODEL = "meta-llama/Llama-3.1-8B-Instruct" 
TOP_K_FINAL = 5
TOP_K_VECTOR = 3
MAX_TOKENS = 512
TEMPERATURE = 0.8
LOG_FILE = "processing"
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
MIN_INPUT_LENGTH_TO_RETRIEVAL = 5
# qdrant config
# QDRANT_HOST = "localhost"
QDRANT_HOST = "914f5b70-3424-49a0-841f-80c4a1d7dac8.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_PORT = 6333
COLLECTION_NAME = "vietnamese_legal_chunks"

DEBOUNCE_TIME = 1.5  # ← Chống spam: số s giữa các lần tìm
ENABLE_HYBRID_SEARCH = False
# === SOFT HYBRID SEARCH BOOST CONFIG ===
METADATA_BOOST_WEIGHTS = {
    "document_number":  8.0,
    "title_contains":   5.0,
    "doc_type":         2.0,
    "year":             1.5,
    "issuer":           1.0,
    "source_file_keyword": 0.4,
}

VECTOR_SCORE_MULTIPLIER = 1.0
ENABLE_SOFT_HYBRID = True

SUBMIT_RETRIEVAL_BOOST = 0.15

# ASR config
ASR_API_URL = "https://Hoangnam5904-STT.hf.space/transcribe_file"
ASR_CHUNK_DURATION = 5.0
ASR_OVERLAP = 0.25
ASR_LANGUAGE = "vi"