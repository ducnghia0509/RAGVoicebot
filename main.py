# main.py
import gradio as gr
import time
import os
from huggingface_hub import InferenceClient

from models.clients import get_models
from retrieval.searcher import retrieve_from_qdrant
from realtime.handler import realtime_search_on_word_count, reset_realtime, used_chunk_ids
from prompt.builder import build_context_and_prompt
from config import *
from utils.logger import logger, timing_logger

# Load models một lần duy nhất
embedder, qdrant_client, hf_client = get_models()

# Anti-spam submit liên tục
_last_submit_time = 0.0


def process_question(user_question: str, history):
    global _last_submit_time

    # Chống spam submit
    current_time = time.time()
    if current_time - _last_submit_time < 1.0:
        yield history
        return
    _last_submit_time = current_time

    logger.info(f"Nhận câu hỏi: {user_question}")
    t_total_start = time.time()

    # === 1. Retrieval khi ấn Gửi ===
    submit_chunks = retrieve_from_qdrant(user_question, TOP_K_VECTOR * 2, used_chunk_ids)
    for c in submit_chunks:
        c["source"] = "submit"
        c["final_score"] = c.get("final_score", 0) + SUBMIT_RETRIEVAL_BOOST

    # === 2. Merge với realtime chunks ===
    from realtime.handler import realtime_shown_chunks

    combined_chunks = list(submit_chunks)
    seen_ids = used_chunk_ids.copy()
    seen_ids.update(c["metadata"].get("chunk_id") for c in submit_chunks if c["metadata"].get("chunk_id"))

    for c in realtime_shown_chunks:
        cid = c["metadata"].get("chunk_id")
        if cid and cid not in seen_ids:
            c_copy = c.copy()
            c_copy["source"] = "realtime"
            combined_chunks.append(c_copy)
            seen_ids.add(cid)

    # Sắp xếp và lấy top
    combined_chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    final_chunks = combined_chunks[:TOP_K_FINAL]

    # === 3. Không tìm thấy gì ===
    if not final_chunks:
        response = "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu pháp luật hiện tại."
        history.append({"role": "user", "content": user_question})
        history.append({"role": "assistant", "content": response})
        reset_realtime()
        yield history
        return

    # === 4. Tạo context + messages ===
    context, messages = build_context_and_prompt(final_chunks, user_question)

    # Thêm vào lịch sử chat
    history.append({"role": "user", "content": user_question})
    history.append({"role": "assistant", "content": ""})
    yield history

    # === 5. Streaming LLM - ĐÃ FIX HOÀN TOÀN ===
    response = ""
    first_token_time = None

    try:
        stream = hf_client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            stream=True,
        )

        for message in stream:
            # Bảo vệ tuyệt đối: nếu không có choices → bỏ qua
            if not message.choices:
                continue

            delta = message.choices[0].delta

            # Một số chunk chỉ có tool_calls hoặc finish_reason
            if delta.content is not None:
                token = delta.content
                response += token
                history[-1]["content"] = response

                # Ghi log thời gian ra token đầu tiên
                if first_token_time is None:
                    first_token_time = time.time()
                    timing_logger.info(f"First token: {first_token_time - t_total_start:.2f}s")

                yield history

            # Dừng sớm nếu model báo xong
            if getattr(delta, "finish_reason", None) is not None:
                break

    except Exception as e:
        logger.error(f"Lỗi khi gọi mô hình: {e}")
        error_msg = "Đã xảy ra lỗi khi kết nối đến mô hình ngôn ngữ. Vui lòng thử lại sau ít phút."
        history[-1]["content"] = error_msg
        yield history
        return

    finally:
        # Cập nhật chunk đã dùng + reset realtime
        for c in final_chunks:
            if cid := c["metadata"].get("chunk_id"):
                used_chunk_ids.add(cid)
        reset_realtime()

        total_time = time.time() - t_total_start
        timing_logger.info(f"Tổng thời gian xử lý: {total_time:.2f}s")

    # Kết thúc hoàn toàn
    yield history


# ===================== GRADIO UI =====================
with gr.Blocks(title="Luật sư AI Việt Nam", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # ⚖️ Trợ lý Pháp lý Việt Nam  
    Hỏi bất kỳ quy định pháp luật nào – tôi trả lời chính xác theo văn bản hiện hành.
    """)

    chatbot = gr.Chatbot(height=620, type="messages", avatar_images=("👤", "⚖️"))

    with gr.Row():
        txt = gr.Textbox(
            label="Câu hỏi của bạn",
            placeholder="Ví dụ: Thuế TNCN với chuyên gia nước ngoài theo nghị định nào năm 2024?",
            lines=3,
            scale=8,
            container=False
        )
        send_btn = gr.Button("🚀 Gửi", variant="primary", scale=1)

    # Realtime search khi đang gõ
    txt.change(
        fn=realtime_search_on_word_count,
        inputs=txt,
        outputs=None
    )

    # Gửi câu hỏi
    send_btn.click(
        fn=process_question,
        inputs=[txt, chatbot],
        outputs=chatbot
    ).then(
        lambda: "",  # Xóa ô input sau khi gửi
        outputs=txt
    )

    # Enter để gửi
    txt.submit(
        fn=process_question,
        inputs=[txt, chatbot],
        outputs=chatbot
    ).then(
        lambda: "",
        outputs=txt
    )

if __name__ == "__main__":
    print("Khởi động Trợ lý Pháp lý Việt Nam...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="favicon.ico" if os.path.exists("favicon.ico") else None
    )