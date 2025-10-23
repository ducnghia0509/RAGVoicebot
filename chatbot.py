

import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import torch
import os
from typing import List, Dict
import time
import logging
import sys
from io import StringIO

# Custom StreamHandler to force UTF-8 encoding for console output
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(stream=sys.stdout)
        self.stream = StringIO()
        sys.stdout = self

    def write(self, text):
        if isinstance(text, str):
            sys.__stdout__.write(text.encode('utf-8', errors='replace').decode('utf-8'))
        else:
            sys.__stdout__.write(str(text))

    def flush(self):
        sys.__stdout__.flush()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        UTF8StreamHandler(),
        logging.FileHandler('processing_log.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_JSONL = "embeddings.jsonl"
FAISS_INDEX_PATH = "faiss_index.bin"
# GGUF_MODEL_PATH = r"C:\Users\DELL\Desktop\Test\LocalProjectPackage\Project\VoiceBotBasic\Gemma-2-IT-Q6_K.gguf"  
GGUF_MODEL_PATH = "gemma-3-1b-it-Q8_0.gguf"  
MODEL_NAME = "Alibaba-NLP/gte-multilingual-base"
TOP_K = 5
MAX_TOKENS = 512
TEMPERATURE = 0.7

# Load embedding model
@st.cache_resource
def load_embedding_model():
    start_time = time.time()
    logger.info("Bắt đầu tải embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device, trust_remote_code=True)
    # model = SentenceTransformer(
    #     r"C:\Users\DELL\.cache\huggingface\hub\models--Alibaba-NLP--gte-multilingual-base",
    #     device=device,
    #     trust_remote_code=True
    # )

    elapsed_time = time.time() - start_time
    logger.info(f"Tải embedding model hoàn tất. Thời gian: {elapsed_time:.2f} giây")
    return model

# Load FAISS index and metadata
@st.cache_resource
def load_faiss_and_metadata():
    start_time = time.time()
    logger.info("Bắt đầu tải FAISS index và metadata...")
    metadatas = []
    texts = []
    with open(EMBEDDINGS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            metadatas.append(data["metadata"])
            texts.append(data["text"])
    index = faiss.read_index(FAISS_INDEX_PATH)
    elapsed_time = time.time() - start_time
    logger.info(f"Tải FAISS index và metadata hoàn tất. Thời gian: {elapsed_time:.2f} giây")
    return index, metadatas, texts

# Retrieve top-k
def retrieve_top_k(query: str, index, embedding_model, metadatas, texts, k: int = TOP_K):
    start_time = time.time()
    logger.info("Bắt đầu quá trình truy xuất top-k...")
    
    # Bước 1: Tạo embedding cho query
    encode_start = time.time()
    logger.info("Bước 1: Tạo embedding cho query...")
    query_emb = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    query_emb = query_emb.reshape(1, -1)
    encode_time = time.time() - encode_start
    logger.info(f"Bước 1 hoàn tất. Thời gian tạo embedding: {encode_time:.2f} giây")
    
    # Bước 2: Tìm kiếm FAISS
    search_start = time.time()
    logger.info("Bước 2: Tìm kiếm top-k trong FAISS index...")
    distances, indices = index.search(query_emb, k)
    search_time = time.time() - search_start
    logger.info(f"Bước 2 hoàn tất. Thời gian tìm kiếm FAISS: {search_time:.2f} giây")
    
    # Bước 3: Xử lý kết quả
    process_start = time.time()
    logger.info("Bước 3: Xử lý kết quả truy xuất...")
    results = []
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadatas):
            continue
        metadata = metadatas[idx]
        source_file = metadata.get('source_file', '')
        chunk_index = metadata.get('chunk_index', 0)
        text = texts[idx]
        
        if source_file and os.path.exists(source_file):
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    chunks = data.get('chunks', []) or data.get('articles', [])
                    if chunk_index < len(chunks):
                        text = chunks[chunk_index].get('text', text)
            except Exception as e:
                logger.warning(f"Không thể đọc file nguồn {source_file}: {e}")
        
        results.append({
            "distance": distances[0][i],
            "metadata": metadata,
            "text": text
        })
    process_time = time.time() - process_start
    logger.info(f"Bước 3 hoàn tất. Thời gian xử lý kết quả: {process_time:.2f} giây")
    
    total_retrieve_time = time.time() - start_time
    logger.info(f"Quá trình truy xuất top-k hoàn tất. Tổng thời gian: {total_retrieve_time:.2f} giây")
    return results

# Load LLM
@st.cache_resource
def load_llm():
    start_time = time.time()
    logger.info("Bắt đầu tải LLM...")
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False,
        kv_cache_type="standard"
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Tải LLM hoàn tất. Thời gian: {elapsed_time:.2f} giây")
    return llm

# Streamlit UI
def main():
    st.title("🦙 Chatbot ")
    st.sidebar.info("Cấu hình:\n- Model: Llama GGUF\n- Retrieval: Top-{TOP_K} từ FAISS\n- Embedding: gte-multilingual-base")

    start_time = time.time()
    logger.info("Bắt đầu khởi tạo giao diện và tài nguyên...")
    
    embedding_model = load_embedding_model()
    index, metadatas, texts = load_faiss_and_metadata()
    llm = load_llm()
    
    resource_load_time = time.time() - start_time
    logger.info(f"Khởi tạo giao diện và tài nguyên hoàn tất. Thời gian: {resource_load_time:.2f} giây")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        input_start_time = time.time()
        logger.info("Nhận được input từ người dùng")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và suy nghĩ..."):
                retrieve_start = time.time()
                logger.info("Bắt đầu truy xuất context...")
                top_chunks = retrieve_top_k(prompt, index, embedding_model, metadatas, texts)
                context = "\n".join([f"- {chunk['text']} (Nguồn: {chunk['metadata'].get('source_file', 'Unknown')})" for chunk in top_chunks])
                retrieve_time = time.time() - retrieve_start
                logger.info(f"Truy xuất context hoàn tất. Thời gian: {retrieve_time:.2f} giây")
                st.info(f"Context retrieved (Top-{TOP_K}):\n{context}")
                
                response_start = time.time()
                logger.info("Bắt đầu tạo phản hồi từ LLM...")
                response_placeholder = st.empty()
                full_response = ""
                first_token_time = None
                for chunk in llm(
                    f"""
Dựa hoàn toàn vào nội dung trong phần Context, trả lời câu hỏi của người dùng bằng tiếng Việt sao cho:
- Trả lời thân thiện, rõ ràng, dễ hiểu.
- Đầy đủ ý và chính xác, BẮT BUỘC CÂU TRẢ LỜI KHÔNG BỊ LẶP GÂY KHÓ HIỂU.
- Không suy diễn ngoài Context.
- Nêu rõ căn cứ pháp lý hoặc nguồn trích (ví dụ: "theo Điều/Luật/Nghị đinh,..."). 
Nếu không có, có thể nói: "Theo như thông tin tôi được cung cấp,..."   
- Tránh lặp lại câu trả lời, không nói lan man.

Context:
{context}

Câu hỏi: {prompt}

Trả lời:
"""
,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    stream=True
                ):
                    if 'choices' in chunk and chunk['choices'][0]['text']:
                        if first_token_time is None:
                            first_token_time = time.time()
                            input_to_first_token_time = first_token_time - input_start_time
                            logger.info(f"Thời gian từ khi nhận input đến khi LLM in chữ đầu tiên: {input_to_first_token_time:.2f} giây")
                            st.info(f"Thời gian từ input đến chữ đầu tiên: {input_to_first_token_time:.2f} giây")
                        full_response += chunk['choices'][0]['text']
                        response_placeholder.markdown(full_response + "▌")
                response_time = time.time() - response_start
                logger.info(f"Tạo phản hồi từ LLM hoàn tất. Thời gian: {response_time:.2f} giây")
                
                response_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                total_processing_time = time.time() - input_start_time
                logger.info(f"Tổng thời gian xử lý câu hỏi: {total_processing_time:.2f} giây")
                st.info(f"Tổng thời gian xử lý: {total_processing_time:.2f} giây")

if __name__ == "__main__":
    main()