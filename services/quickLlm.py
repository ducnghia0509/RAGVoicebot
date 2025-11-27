import re
import os
from huggingface_hub import InferenceClient
from config import *

# ================= Quick Templates =================
QUICK_TEMPLATES = {
    "định mức": "Đang tìm kiếm thông tin về định mức kinh tế kỹ thuật cho bạn",
    "thuế": "Để tôi tra cứu quy định về thuế nhé",
    "hợp đồng": "Đang tìm kiếm thông tin về hợp đồng",
    "bảo hiểm": "Đang tra cứu quy định bảo hiểm",
    "lương": "Để tôi kiểm tra quy định về lương",
    "nghỉ phép": "Đang tìm thông tin về chế độ nghỉ phép",
    "sa thải": "Đang tra cứu quy định về chấm dứt hợp đồng",
    "phạt": "Để tôi tìm thông tin về mức phạt",
    "thông tư": "Đang tìm kiếm văn bản pháp luật liên quan",
    "nghị định": "Đang tra cứu nghị định bạn cần",
    "luật": "Đang tìm thông tin trong văn bản luật",
    "quyết định": "Đang tìm kiếm quyết định liên quan",
}
DEFAULT_RESPONSE = "Đang tìm kiếm thông tin cho bạn"

# ================= Rule-based quick response =================
def get_quick_response(question: str, max_length: int = 60) -> str:

    question_lower = question.lower()
    
    # Match keyword templates
    for keyword, template in QUICK_TEMPLATES.items():
        if keyword in question_lower:
            return template

    # Extract topic if no keyword matched
    topic_patterns = [
        r'về\s+([^\?]{3,20})',
        r'([^\?]{3,20})\s+như\s+thế\s+nào',
        r'([^\?]{3,20})\s+là\s+gì',
        r'quy\s+định\s+([^\?]{3,20})',
    ]
    
    for pattern in topic_patterns:
        match = re.search(pattern, question_lower)
        if match:
            topic = match.group(1).strip()
            topic = re.sub(r'\s+', ' ', topic)
            if len(topic) < 25:
                return f"Đang tìm kiếm thông tin về {topic}"

    return DEFAULT_RESPONSE

# ================= LLM-based quick response =================
HF_TOKEN = os.getenv("HF_TOKEN")
_hf_client = InferenceClient(token=HF_TOKEN, model=QUICK_MODEL)


def get_quick_response_llm(question: str) -> str:
    """
    Gọi LLM nhanh, không stream, trả về full output.
    """
    
    messages = [
        {"role": "system", "content": f"Bạn hãy tóm tắt nội dung câu hỏi sau một cách tự nhiên, ngắn gọn trong vòng 10 từ: '{question}'. "
            "Chỉ nói rằng tôi đang tìm kiếm thông tin để trả lời, có thể làm ngắn gọn lại yêu cầu của người dùng, ví dụ như đang tìm kiếm nội dung liên quan tới.... "
            "Bạn không được phép trả lời câu hỏi."},
        # {"role": "user", "content": ""}
    ]
    
    try:
        # Gọi LLM không stream, trả về toàn bộ output
        response = _hf_client.chat_completion(
            messages,
            max_tokens=64,
            temperature=0.3,
            top_p=0.95,
            stream=False
        )
        
        # HF client trả về list hoặc dict, extract text
        if isinstance(response, dict) and "choices" in response:
            text = response["choices"][0]["message"]["content"].strip()
        elif isinstance(response, list):
            text = response[0]["message"]["content"].strip()
        else:
            text = str(response).strip()
        
        return text
    
    except Exception:

        return DEFAULT_RESPONSE


# ================= Router (giữ nguyên logic) =================
def quick_router(question: str) -> str:
    """
    Quyết định dùng rule hay llm, giữ nguyên logic gốc.
    """
    res = get_quick_response(question)
    if res != DEFAULT_RESPONSE:
        return res
    return get_quick_response_llm(question)
