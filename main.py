# main.py
import gradio as gr
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Generator
from collections import deque
import threading
import struct
from typing import Optional

from models.clients import get_models
from retrieval.searcher import retrieve_from_qdrant
from realtime.handler import realtime_search_on_word_count, reset_realtime, used_chunk_ids, realtime_shown_chunks
from prompt.builder import build_context_and_prompt
from config import *
from utils.logger import logger, timing_logger

# === Services ===
from services.quickLlm import get_quick_response, get_quick_response_llm
from services.tts import tts_client

# ========== Setup logs directory ==========
os.makedirs("logs", exist_ok=True)

# ===================== PIPELINE LOGGER (general) =====================
PIPELINE_LOG_FILE = "logs/pipeline.log"
import logging
pipeline_logger = logging.getLogger("pipeline")
pipeline_logger.setLevel(logging.INFO)
p_handler = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
p_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
pipeline_logger.handlers = [p_handler]

# ===================== DETAILED LOGGERS (per step) =====================
def create_step_logger(name: str, filename: str, level=logging.INFO):
    lg = logging.getLogger(name)
    lg.setLevel(level)
    h = logging.FileHandler(os.path.join("logs", filename), encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    lg.handlers = [h]
    return lg

retrieval_logger = create_step_logger("retrieval", "retrieval.log", logging.DEBUG)
context_logger   = create_step_logger("context", "context.log", logging.DEBUG)
stream_logger    = create_step_logger("stream", "stream.log", logging.DEBUG)
tts_send_logger  = create_step_logger("tts_send", "tts_send.log", logging.DEBUG)
tts_resp_logger  = create_step_logger("tts_resp", "tts_resp.log", logging.DEBUG)
playback_logger  = create_step_logger("playback", "playback.log", logging.DEBUG)
quick_logger     = create_step_logger("quick", "quick.log", logging.DEBUG)

# ===================== INIT MODELS & EXECUTOR =====================
embedder, qdrant_client, hf_client = get_models()
_executor = ThreadPoolExecutor(max_workers=3)
_last_submit_time = 0.0

# ===================== AUDIO QUEUE (single queue) =====================
audio_queue = deque()           # items: (filepath, duration, meta_dict)
queue_lock = threading.Lock()
playback_thread_running = False
playback_thread = None

def safe_preview(text: str, max_len: int = 200):
    s = text.replace("\n", " ")
    return s
    # return (s[:max_len] + "...") if len(s) > max_len else s

def get_wav_duration(audio_bytes: bytes) -> float:
    if not audio_bytes or len(audio_bytes) < 44:
        return max(0.8, len(audio_bytes) / 30000.0)

    if audio_bytes[:4] != b'RIFF' or audio_bytes[8:12] != b'WAVE':
        return len(audio_bytes) / 38500.0

    try:
        fmt_size = struct.unpack_from('<I', audio_bytes, 16)[0]
        byte_rate_offset = 20 + (fmt_size - 16)
        if byte_rate_offset + 4 > len(audio_bytes):
            raise ValueError()
        byte_rate = struct.unpack_from('<I', audio_bytes, byte_rate_offset)[0]

        data_pos = audio_bytes.find(b'data', 12)
        if data_pos == -1 or data_pos + 8 > len(audio_bytes):
            raise ValueError()
        data_size = struct.unpack_from('<I', audio_bytes, data_pos + 4)[0]

        if byte_rate > 0 and data_size > 0:
            duration = data_size / byte_rate
            if 0.1 < duration < 300:
                return duration

    except:
        pass

    duration = len(audio_bytes) / 38500.0
    return max(0.8, duration)


def cleanup_wav_file(filepath: str):
    """Xóa file WAV sau khi phát xong"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            playback_logger.debug(f"CLEANUP | deleted={filepath}")
    except Exception as e:
        playback_logger.error(f"CLEANUP_ERROR | file={filepath} | error={e}")

# ========= UNIFIED TTS THREAD (with detailed logs) =========
def _unified_tts_thread(text: str, is_quick: bool = False):
    clean_text = text.strip()
    if not clean_text:
        pipeline_logger.info("TTS called with empty text -> skip")
        return

    start = time.time()
    speed = 1.2
    source = "quick" if is_quick else "batch"

    tts_send_logger.info(f"SEND | source={source} | len={len(clean_text)} | preview='{safe_preview(clean_text,120)}' | speed={speed}")

    audio_buffer = None
    try:
        audio_buffer = tts_client.speak_to_buffer(clean_text, 'vi', speed=speed)
    except Exception as e:
        tts_resp_logger.error(f"ERROR calling tts_client: {e}")

    audio_bytes = None
    if audio_buffer:
        try:
            audio_bytes = audio_buffer.getvalue() if audio_buffer.getbuffer().nbytes > 0 else None
        except Exception:
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read() if audio_buffer else None

    if not audio_bytes:
        pipeline_logger.info(f"END | TTS {source} | FAILED | '{clean_text[:40]}...'")
        return

    duration = get_wav_duration(audio_bytes)
    print("DEBUG: ", duration)
    if duration is None:
        duration = len(clean_text) / 18.0

    # Tạo file WAV ngay trong thread này
    tmp_path = f"tmp_audio_{time.time_ns()}.wav"
    try:
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        tts_resp_logger.error(f"ERROR writing WAV file: {e}")
        return

    pipeline_logger.info(f"END    | TTS {source} | {duration:.2f}s | OK | file={tmp_path}")

    meta = {
        "source": source,
        "text_preview": safe_preview(clean_text, 200),
        "size_bytes": len(audio_bytes),
        "timestamp": time.time(),
    }

    # Push filepath vào queue
    with queue_lock:
        if is_quick:
            audio_queue.appendleft((tmp_path, duration, meta))
        else:
            audio_queue.append((tmp_path, duration, meta))
        total_waiting = sum(d for _, d, _ in audio_queue)
        playback_logger.info(f"QUEUE_ADD | source={source} | queue_len={len(audio_queue)} | total_waiting≈{total_waiting:.1f}s | file={tmp_path}")

def _playback_worker():
    """Thread worker để quản lý phát file WAV tuần tự"""
    global playback_thread_running
    playback_logger.info("PLAYBACK_THREAD | Started")
    
    while playback_thread_running:
        filepath = None
        duration = None
        
        with queue_lock:
            if len(audio_queue) > 1:  # Chỉ dequeue khi có ít nhất 2 file (file đầu đang phát)
                # Xóa file đầu tiên (đã phát xong)
                old_file, old_duration, old_meta = audio_queue.popleft()
                playback_logger.info(f"CLEANUP_OLD | file={old_file}")
                # cleanup_wav_file(old_file)
        
        time.sleep(0.5)  # Check mỗi 0.5s
    
    # Cleanup remaining files khi stop
    with queue_lock:
        while audio_queue:
            filepath, _, _ = audio_queue.popleft()
            # cleanup_wav_file(filepath)
    
    playback_logger.info("PLAYBACK_THREAD | Stopped")

def start_playback_thread():
    """Khởi động playback thread"""
    global playback_thread_running, playback_thread
    if not playback_thread_running:
        playback_thread_running = True
        playback_thread = threading.Thread(target=_playback_worker, daemon=True)
        playback_thread.start()
        playback_logger.info("PLAYBACK_THREAD | Initialized")

def stop_playback_thread():
    """Dừng playback thread"""
    global playback_thread_running
    playback_thread_running = False
    if playback_thread:
        playback_thread.join(timeout=2.0)


# ========= GET NEXT AUDIO (for Gradio yield) =========
def get_next_audio():
    """Lấy file audio tiếp theo từ queue để yield cho Gradio"""
    with queue_lock:
        if not audio_queue:
            return None
        # Lấy file đầu tiên trong queue (không xóa khỏi queue)
        filepath, duration, meta = audio_queue[0]
    
    playback_logger.info(f"YIELD_AUDIO | file={filepath} | duration≈{duration:.2f}s")
    return filepath


# ========= WAIT FOR QUEUE EMPTY (yield generator) =========
def wait_for_audio_queue_empty(history):
    """Chờ queue rỗng và yield audio files"""
    last_yielded = None
    
    while True:
        with queue_lock:
            if not audio_queue:
                break
            current_file, _, _ = audio_queue[0]
        
        # Chỉ yield khi có file mới
        if current_file != last_yielded:
            yield history, current_file
            last_yielded = current_file
            playback_logger.info(f"YIELDED | file={current_file}")
        
        time.sleep(0.3)
    
    # Yield cuối cùng với audio None
    yield history, gr.Audio(value=None, visible=False)

# ========= PROCESS QUESTION (main pipeline) =========
def process_question(user_question: str, history) -> Generator:
    global _last_submit_time
    now = time.time()
    if now - _last_submit_time < 1.0:
        yield history, gr.Audio(value=None, visible=False)
        return
    _last_submit_time = now

    t_start = time.time()
    pipeline_logger.info(f"{'='*80}")
    pipeline_logger.info(f"START  | Pipeline lớn | '{safe_preview(user_question,200)}'")

    # clear old queue
    with queue_lock:
        audio_queue.clear()
        playback_logger.info("QUEUE_CLEARED at new question start")

    # 1. append user message
    history.append({"role": "user", "content": user_question})
    yield history, gr.Audio(value=None, visible=False)

    # Append placeholder cho assistant response
    history.append({"role": "assistant", "content": ""})
    yield history, gr.Audio(value=None, visible=False)

    # 2. Quick response (fast LLM)
    quick_text = None
    try:
        quick_text = get_quick_response_llm(user_question)
    except Exception as e:
        quick_logger.error(f"Quick LLM error: {e}")

    if quick_text and quick_text.strip():
        quick_display = f"[Tìm kiếm nhanh] {quick_text}"
        history[-1]["content"] = quick_display
        yield history, gr.Audio(value=None, visible=False)
        quick_logger.info(f"QUICK_TEXT | preview='{safe_preview(quick_text,200)}' | len={len(quick_text)}")
        _executor.submit(_unified_tts_thread, quick_text, is_quick=True)
        pipeline_logger.info("Đã gửi Quick TTS (ưu tiên cao nhất)")
        
        # Chờ quick TTS vào queue và yield
        for _ in range(20):  # Chờ tối đa 2s
            time.sleep(0.1)
            with queue_lock:
                if audio_queue:
                    quick_audio, _, _ = audio_queue[0]
                    yield history, quick_audio
                    playback_logger.info(f"QUICK_AUDIO_YIELDED | file={quick_audio}")
                    break

    # 3. Retrieval (detailed logging)
    priority_chunks = []
    seen = set()

    retrieval_logger.info(f"RETRIEVE_START | question_preview='{safe_preview(user_question,200)}' | TOP_K={TOP_K_VECTOR*3}")
    # realtime shown chunks first
    for c in realtime_shown_chunks:
        cid = c["metadata"].get("chunk_id")
        if cid and cid not in used_chunk_ids and cid not in seen:
            c = c.copy()
            c["source"] = "realtime"
            c["final_score"] = c.get("final_score", 0) + 0.05
            priority_chunks.append(c)
            seen.add(cid)
            retrieval_logger.debug(f"RETRIEVE_ADD realtime | cid={cid} | score={c.get('final_score')}")
    # main vector retrieval
    try:
        chunks = retrieve_from_qdrant(user_question, top_k=TOP_K_VECTOR*3, exclude_ids=used_chunk_ids, force_no_filter=False)
        retrieval_logger.info(f"RETRIEVE_RESULT main | got={len(chunks)}")
        for c in chunks:
            cid = c["metadata"].get("chunk_id")
            if cid and cid not in seen:
                c = c.copy()
                c["source"] = "filtered"
                c["final_score"] = c.get("final_score", 0) + SUBMIT_RETRIEVAL_BOOST
                priority_chunks.append(c)
                seen.add(cid)
                retrieval_logger.debug(f"RETRIEVE_ADD filtered | cid={cid} | score={c.get('final_score')}")
    except Exception as e:
        retrieval_logger.error(f"retrieve_from_qdrant error: {e}")

    # fallback retrieval if not enough
    if len(priority_chunks) < 6:
        try:
            fallback = retrieve_from_qdrant(user_question, top_k=15, exclude_ids=used_chunk_ids | seen, force_no_filter=True)
            retrieval_logger.info(f"RETRIEVE_RESULT fallback | got={len(fallback)}")
            for c in fallback:
                cid = c["metadata"].get("chunk_id")
                if cid and cid not in seen:
                    c = c.copy()
                    c["source"] = "fallback"
                    priority_chunks.append(c)
                    seen.add(cid)
                    retrieval_logger.debug(f"RETRIEVE_ADD fallback | cid={cid} | score={c.get('final_score', 0)}")
        except Exception as e:
            retrieval_logger.error(f"fallback retrieve error: {e}")

    # log top priority chunks (ids & preview)
    priority_chunks.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    final_chunks = priority_chunks[:6]
    retrieval_logger.info(f"PRIORITY_CHUNKS | picked={len(final_chunks)}")
    for i, c in enumerate(final_chunks):
        cid = c["metadata"].get("chunk_id")
        text_preview = safe_preview(c["payload"].get("text", "") if "payload" in c else c.get("text", ""), 200)
        retrieval_logger.info(f"CHUNK[{i}] | cid={cid} | src={c.get('source')} | score={c.get('final_score')} | preview='{text_preview}'")

    if not final_chunks:
        no_info = "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn."
        history[-1]["content"] = no_info
        _executor.submit(_unified_tts_thread, no_info, is_quick=False)
        yield history, get_next_audio()
        reset_realtime()
        return

    # 4. Build context & prompt (log resulting context and messages)
    try:
        context, messages = build_context_and_prompt(final_chunks, user_question)
        context_logger.info("CONTEXT_BUILD_SUCCESS")
        context_logger.debug(f"CONTEXT_PREVIEW:\n{safe_preview(context,2000)}")
        # messages can be a list of dicts; log each element
        for idx, m in enumerate(messages):
            context_logger.debug(f"MESSAGE[{idx}] role={m.get('role')} | preview='{safe_preview(m.get('content',''),500)}'")
    except Exception as e:
        context_logger.error(f"build_context_and_prompt error: {e}")
        history.append({"role": "assistant", "content": "Lỗi khi xây dựng ngữ cảnh."})
        yield history, gr.Audio(value=None, visible=False)
        return

    # 5. Stream LLM response + chunked TTS for sentences
    response = ""
    pending_text = ""
    sent_texts = set()
    yield history, get_next_audio()

    # smart sentence splitter (tweakable)
    sentence_pattern = r'[^\.!\?\n][^\.?!]*[.?!]\s*'
    def extract_sentences(text: str):
        return [s.strip() for s in re.findall(sentence_pattern, text) if len(s.strip().split()) >= 4]

    try:
        stream_logger.info("LLM_STREAM_START")
        stream_logger.debug(f"LLM_CALL messages_count={len(messages)}")
        for i, m in enumerate(messages):
            # already logged above; keep brief here
            pass

        stream = hf_client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stream=True
        )

        for idx, msg in enumerate(stream):
            # Each msg is a partial delta
            if not msg.choices or not msg.choices[0].delta.content:
                continue

            delta = msg.choices[0].delta.content
            response += delta
            pending_text += delta
            history[-1]["content"] = response

            stream_logger.debug(f"STREAM_DELTA | len={len(delta)} | preview='{safe_preview(delta,200)}'")

            # Extract complete sentences in pending_text
            sentences = extract_sentences(pending_text)
            for sentence in sentences:
                clean = sentence.strip(".\n ").strip()
                if clean and clean not in sent_texts and len(clean.split()) >= 6:
                    sent_texts.add(clean)
                    # Log which sentence is queued for TTS
                    stream_logger.info(f"SEND_TTS_CHUNK | preview='{safe_preview(clean,200)}' | len_words={len(clean.split())}")
                    _executor.submit(_unified_tts_thread, clean, is_quick=False)
                    # Remove only the first occurrence
                    pending_text = pending_text.replace(sentence, "", 1)

            yield history, get_next_audio()

            if getattr(msg.choices[0].delta, "finish_reason", None):
                remaining = pending_text.strip(".\n ").strip()
                if remaining and remaining not in sent_texts and len(remaining.split()) >= 6:
                    stream_logger.info(f"SEND_TTS_REMAINING | preview='{safe_preview(remaining,200)}' | len_words={len(remaining.split())}")
                    _executor.submit(_unified_tts_thread, remaining, is_quick=False)
                break

    except Exception as e:
        stream_logger.error(f"Stream error: {e}")
        history[-1]["content"] = "Đã xảy ra lỗi khi xử lý câu hỏi của bạn."
        yield history, gr.Audio(value=None, visible=False)

    finally:
        # mark used chunk ids
        for c in final_chunks:
            if cid := c["metadata"].get("chunk_id"):
                used_chunk_ids.add(cid)
        reset_realtime()

        total_time = time.time() - t_start
        pipeline_logger.info(f"END    | Total: {total_time:.2f}s | Bắt đầu chờ phát hết TTS...")
        pipeline_logger.info(f"{'='*80}")

        # wait for audio queue to be fully played (yield)
        for final_yield in wait_for_audio_queue_empty(history):
            yield final_yield

    pipeline_logger.info(f"END    | Total: {time.time()-t_start:.2f}s | ĐÃ PHÁT HẾT ÂM THANH")

# ===================== GRADIO UI =====================
with gr.Blocks(title="Luật sư AI Việt Nam", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Trợ lý Pháp lý AI Việt Nam")
    chatbot = gr.Chatbot(height=620, type="messages", avatar_images=("Người dùng", "Luật sư"), show_label=False)

    audio_output = gr.Audio(
        label="Âm thanh",
        type="filepath",   # <-- đổi từ 'bytes' sang 'filepath'
        autoplay=True,
        visible=False,
        elem_classes="tts-player",
        interactive=False
    )

    with gr.Row():
        txt = gr.Textbox(
            label="Câu hỏi của bạn",
            placeholder="Ví dụ: Ly hôn đơn phương cần giấy tờ gì?",
            lines=3,
            scale=8
        )
        send = gr.Button("Gửi", variant="primary")

    # Realtime search khi gõ
    txt.change(realtime_search_on_word_count, txt, None)

    # Gửi câu hỏi
    send.click(process_question, [txt, chatbot], [chatbot, audio_output]) \
        .then(lambda: "", outputs=txt)
    txt.submit(process_question, [txt, chatbot], [chatbot, audio_output]) \
        .then(lambda: "", outputs=txt)

    # JS: quản lý audio queue trên client để phát tuần tự (giữ nguyên)
    demo.load(None, None, None, js="""
    () => {
        let queue = [];
        let playing = false;

        const playNext = () => {
            if (playing || queue.length === 0) return;
            playing = true;
            const src = queue.shift();

            const container = document.querySelector('.tts-player');
            if (!container) return;
            container.innerHTML = `<audio src="${src}" autoplay></audio>`;
            container.style.display = 'block';

            const audio = container.querySelector('audio');
            const cleanup = () => {
                playing = false;
                container.style.display = 'none';
                container.innerHTML = '';
                playNext();
            };
            audio.onended = cleanup;
            audio.onerror = cleanup;
            audio.onstalled = cleanup;
        };

        new MutationObserver(mutations => {
            mutations.forEach(m => {
                m.addedNodes.forEach(node => {
                    if (node.tagName === 'AUDIO' && node.src && node.src.startsWith('data:audio')) {
                        queue.push(node.src);
                        if (!playing) playNext();
                    }
                });
            });
        }).observe(document.body, { childList: true, subtree: true });
    }
    """)

if __name__ == "__main__":
    print("Khởi động Trợ lý Pháp lý AI Việt Nam...")
    print(f"Log pipeline: {os.path.abspath(PIPELINE_LOG_FILE)}")
    print(f"TTS Endpoint: {tts_client.endpoint}")
    
    # Khởi động playback thread
    start_playback_thread()
    
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            favicon_path="favicon.ico" if os.path.exists("favicon.ico") else None
        )
    finally:
        stop_playback_thread()