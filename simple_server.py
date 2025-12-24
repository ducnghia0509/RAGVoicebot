import os
import time
import uuid
import logging
from pydub import AudioSegment, silence
# Explicit ffmpeg binary (Windows) provided by user
# --- FFMPEG SETUP FIXED ---
FFMPEG_DIR = "./ffmpeg-7.1.1-essentials_build/bin"
FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")

# Kiểm tra file tồn tại
if not os.path.exists(FFMPEG_PATH):
    logger = logging.getLogger("simple_server")
    logger.error(f"FFMPEG not found at: {FFMPEG_PATH}")
    raise FileNotFoundError(f"FFMPEG not found at: {FFMPEG_PATH}")

if not os.path.exists(FFPROBE_PATH):
    logger = logging.getLogger("simple_server")
    logger.warning(f"FFPROBE not found at: {FFPROBE_PATH}")

# Thiết lập đường dẫn cho pydub
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH
os.environ["FFPROBE_BINARY"] = FFPROBE_PATH

try:
    AudioSegment.converter = FFMPEG_PATH
    AudioSegment.ffprobe = FFPROBE_PATH
    logger = logging.getLogger("simple_server")
    logger.info(f"Using ffmpeg: {FFMPEG_PATH}")
    logger.info(f"Using ffprobe: {FFPROBE_PATH}")
except Exception as e:
    logger = logging.getLogger("simple_server")
    logger.error(f"Failed to set ffmpeg paths: {e}")
from typing import List
import threading
import re
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
from pydantic import BaseModel
from models.clients import get_models
from services.quickLlm import get_quick_response_llm
from services.tts import tts_client
from services.asr import get_asr_client
from retrieval.searcher import retrieve_from_qdrant
from prompt.builder import build_context_and_prompt
from typing import Optional 
from config import *


# --- Logging setup ---
LOG_FILE = "logs/simple_server.log"
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("simple_server")


# --- App & static ---
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_AUDIO_DIR = os.path.join(BASE_DIR, "tmp_audio")
os.makedirs(TMP_AUDIO_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# --- Models / clients init ---
logger.debug("Initializing models (embedder, qdrant_client, hf_client)")
embedder, qdrant_client, hf_client = get_models()
logger.debug("Models initialized")

# --- Background task management & tasks store ---
executor = ThreadPoolExecutor(max_workers=3)
tasks = {}  # id -> task info
tasks_lock = threading.Lock()


def _safe_update_task(task_id: str, **kwargs):
    with tasks_lock:
        t = tasks.get(task_id, {})
        t.update(kwargs)
        tasks[task_id] = t


TRACE_LOG = os.path.join('logs', 'trace.log')


def trace_event(task_id: str, event: str, data: dict = None):
    """Append a JSON-line to global trace log and per-task log."""
    rec = {
        'ts': _now(),
        'task_id': task_id,
        'event': event,
        'data': data or {}
    }
    line = json.dumps(rec, ensure_ascii=False)
    try:
        with open(TRACE_LOG, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        logger.exception('Failed to write global trace log')
    # per-task file
    try:
        pth = os.path.join('logs', f'task_{task_id}.log')
        with open(pth, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        logger.exception('Failed to write task trace log')


def _now():
    return int(time.time() * 1000)


def _chunk_text_to_batches(text: str, target_words: int = 10, max_words: int = 30):
    """
    Split text into ordered batches suitable for TTS:
    - First split by sentence delimiters (.!?), preserving tokens.
    - Merge adjacent sentences until roughly `target_words` reached, but never exceed `max_words`.
    - If a single sentence is longer than `max_words`, split it by words into sub-batches.

    Returns (batches, remainder) where remainder is the tail text that wasn't large enough to emit yet.
    """
    if not text or not text.strip():
        return [], ''

    # split by sentence delimiters, keep delimiter
    parts = re.split(r'([\.\!\?]+)\s*', text)
    # rebuild sentences: combine text + following delimiter
    sentences = []
    i = 0
    while i < len(parts):
        if parts[i].strip() == '':
            i += 1
            continue
        if i + 1 < len(parts) and re.match(r'[\.\!\?]+', parts[i+1]):
            sent = (parts[i] + parts[i+1]).strip()
            i += 2
        else:
            sent = parts[i].strip()
            i += 1
        if sent:
            sentences.append(sent)

    batches = []
    current = []
    current_words = 0

    def flush_current():
        nonlocal current, current_words
        if current:
            batches.append(' '.join(current).strip())
            current = []
            current_words = 0

    for sent in sentences:
        words = sent.split()
        wcount = len(words)
        if wcount >= max_words:
            # split long sentence into word chunks
            for start in range(0, wcount, max_words):
                sub = words[start:start+max_words]
                if current:
                    # flush existing first to keep order
                    flush_current()
                batches.append(' '.join(sub).strip())
        else:
            if current_words + wcount <= max_words:
                current.append(sent)
                current_words += wcount
                # if reached target, flush
                if current_words >= target_words:
                    flush_current()
            else:
                # flush current and start new
                flush_current()
                current.append(sent)
                current_words = wcount
                if current_words >= target_words:
                    flush_current()

    # remainder is anything left in current
    remainder = ''
    if current:
        remainder = ' '.join(current).strip()

    return batches, remainder

def _extract_complete_sentences(text: str):
    """
    Return (sentences, remainder)
    FIXED: Normalize whitespace để tránh trùng lặp do space khác nhau
    """
    if not text:
        return [], ''

    import re as _re
    
    # Normalize: strip leading/trailing, collapse multiple spaces
    text = ' '.join(text.split())
    
    pause_chars = r"\.\!\?,;:\-\u2013\u2014\u2026"
    pattern = fr"(.+?(?:[{pause_chars}]+)(?:['\"\)\]]+)?)\s*"

    matches = [m.group(1).strip() for m in _re.finditer(pattern, text, flags=_re.S)]

    if matches:
        last_match = matches[-1]
        end_pos = None
        for m in _re.finditer(pattern, text, flags=_re.S):
            if m.group(1).strip() == last_match:
                end_pos = m.end(1)

        if end_pos is None:
            end_pos = text.rfind(last_match) + len(last_match)

        remainder = text[end_pos:].strip()  # STRIP remainder
        cleaned = [s.strip() for s in matches]

        return cleaned, remainder
    else:
        return [], text
    
class AskRequest(BaseModel):
    question: str


def _write_audio_buffer_to_file(buffer, prefix="audio"):
    ts = int(time.time() * 1000)
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}.wav"
    path = os.path.join(TMP_AUDIO_DIR, fname)
    try:
        with open(path, "wb") as f:
            buffer.seek(0)
            f.write(buffer.read())
        logger.debug(f"Wrote TTS to file: {path}")
        # Trim silence to avoid gaps at boundaries
        try:
            _trim_silence_file(path)
        except Exception:
            logger.exception("Trimming silence failed")
        return fname, path
    except Exception as e:
        logger.exception("Failed to write audio file")
        raise


def _trim_silence_file(path: str, silence_thresh: int = -40, min_silence_len: int = 120) -> str:
    """Trim leading/trailing silence from WAV file in-place. Returns path."""
    try:
        seg = AudioSegment.from_file(path, format="wav")
        dur = len(seg)
        silences = silence.detect_silence(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        start_trim = 0
        end_trim = dur
        if silences:
            # leading
            if silences[0][0] == 0:
                start_trim = silences[0][1]
            # trailing
            if silences[-1][1] >= dur:
                end_trim = silences[-1][0]
        # guard
        if start_trim > 0 or end_trim < dur:
            trimmed = seg[start_trim:end_trim]
            trimmed.export(path, format="wav")
            logger.debug(f"Trimmed silence for {path}: start={start_trim}ms end={end_trim}ms")
    except Exception:
        logger.exception(f"Error trimming silence for {path}")
    return path


def _merge_audio_files(file_paths: List[str], out_path: str, crossfade_ms: int = 30) -> str:
    """Merge list of WAV files into one with small crossfade. Returns out_path."""
    try:
        combined = AudioSegment.empty()
        for p in file_paths:
            if not os.path.exists(p):
                logger.debug(f"Merge: missing file {p}, skipping")
                continue
            seg = AudioSegment.from_file(p, format="wav")
            if len(combined) == 0:
                combined = seg
            else:
                # append with crossfade
                try:
                    combined = combined.append(seg, crossfade=crossfade_ms)
                except Exception:
                    # fallback without crossfade
                    combined = combined + seg
        combined.export(out_path, format="wav")
        logger.debug(f"Merged {len(file_paths)} files -> {out_path}")
        # trim resulting file
        try:
            _trim_silence_file(out_path)
        except Exception:
            pass
        return out_path
    except Exception:
        logger.exception("Failed to merge audio files")
        raise


def _background_process_all(task_id: str, question: str,audio_data: Optional[bytes] = None):
    
    logger.debug(f"[BG] Start background process for id={task_id}")
    if audio_data:
        try:
            asr_client = get_asr_client()
            transcription = asr_client.transcribe_bytes(audio_data, 16000, stream=False)
            
            if transcription.get('success'):
                question = transcription.get('text', question)
                _safe_update_task(task_id, original_audio=True, transcribed_text=question)
                trace_event(task_id, 'asr_complete', {'transcribed_text': question})
        except Exception as e:
            logger.exception(f"ASR failed for task {task_id}")
    try:
        _safe_update_task(task_id, bg_started_at=_now())
        # 1. quick audio (if quick_text exists)
        with tasks_lock:
            quick_text = tasks.get(task_id, {}).get('quick_text')
        if quick_text:
            logger.debug(f"[BG] Generating quick audio for id={task_id} len={len(quick_text)}")
            try:
                trace_event(task_id, 'tts_quick_request', {'text_preview': quick_text})
                buf = tts_client.speak_to_buffer(quick_text, 'vi', speed=1.4)
                if buf:
                    fname, path = _write_audio_buffer_to_file(buf, prefix=f"quick_{task_id}")
                    quick_url = f"/audio/{fname}"
                    _safe_update_task(task_id, quick_audio_url=quick_url, quick_audio_ready_at=_now())
                    logger.debug(f"[BG] quick audio ready for id={task_id} -> {fname}")
                    trace_event(task_id, 'audio_created', {'type': 'quick', 'file': fname, 'url': quick_url})
            except Exception:
                logger.exception(f"[BG] quick TTS failed for id={task_id}")
                trace_event(task_id, 'audio_error', {'type': 'quick'})

        # 2. Full pipeline: retrieval -> build context -> LLM -> chunk -> TTS per chunk
        _safe_update_task(task_id, status='full_processing', full_started_at=_now())
        logger.debug(f"[BG] Retrieval for id={task_id}")
        try:
            chunks = retrieve_from_qdrant(question, top_k=TOP_K_VECTOR*3, exclude_ids=set(), force_no_filter=False)
            logger.debug(f"[BG] Retrieved {len(chunks) if chunks else 0} chunks for id={task_id}")
        except Exception:
            logger.exception(f"[BG] Retrieval failed for id={task_id}")
            chunks = []

        try:
            context, messages = build_context_and_prompt(chunks, question)
            logger.debug(f"[BG] Built context/messages for id={task_id}")
        except Exception:
            logger.exception(f"[BG] build_context_and_prompt failed for id={task_id}")
            messages = []

        # Try streaming LLM if available to reduce latency: as deltas arrive, chunk and TTS immediately
        final_text = ""
        audio_urls = []
        try:
            logger.debug(f"[BG] Attempting streaming LLM for id={task_id}")
            stream = hf_client.chat_completion(messages, max_tokens=512, temperature=0.7, top_p=0.95, stream=True)
            trace_event(task_id, 'llm_request', {'messages_count': len(messages)})
            
            pending = ""
            sent_texts = set()  # Set để track text đã gửi TTS
            
            for msg in stream:
                if not msg.choices or not getattr(msg.choices[0].delta, 'content', None):
                    if getattr(msg.choices[0], 'finish_reason', None):
                        break
                    continue
                    
                delta = msg.choices[0].delta.content
                trace_event(task_id, 'llm_delta', {'delta_preview': (delta or '')})
                final_text += delta
                pending += delta
                
                # Extract completed sentences
                sentences, remainder = _extract_complete_sentences(pending)
                
                for sent in sentences:
                    # NORMALIZE before checking duplicates
                    clean = ' '.join(sent.strip().split())  # Collapse multiple spaces
                    
                    if not clean:
                        continue
                        
                    # Check duplicate với normalized text
                    if clean in sent_texts:
                        logger.debug(f"[BG] SKIP duplicate sentence: {clean[:50]}...")
                        continue
                        
                    sent_texts.add(clean)
                    
                    try:
                        logger.debug(f"[BG] Streaming TTS for sentence (len {len(clean.split())} words) for id={task_id}")
                        trace_event(task_id, 'tts_request', {'text_preview': clean})
                        
                        buf = tts_client.speak_to_buffer(clean, 'vi', speed=1.0)
                        if buf:
                            fname, path = _write_audio_buffer_to_file(buf, prefix=f"full_{task_id}")
                            url = f"/audio/{fname}"
                            audio_urls.append(url)
                            _safe_update_task(
                                task_id, 
                                full_audio_urls=audio_urls.copy(), 
                                full_text=final_text, 
                                last_part_ready_at=_now()
                            )
                            logger.debug(f"[BG] Streaming sentence audio ready -> {fname}")
                            trace_event(task_id, 'audio_created', {'type': 'full_sentence', 'file': fname, 'url': url})
                    except Exception:
                        logger.exception(f"[BG] Streaming TTS failed for id={task_id}")
                        trace_event(task_id, 'audio_error', {'type': 'full_sentence'})
                
                pending = remainder  # Update pending với remainder đã normalized
            
            # Process remaining text SAU KHI stream kết thúc
            remaining = ' '.join(pending.strip().split())  # Normalize
            
            if remaining and remaining not in sent_texts:  # CHECK duplicate
                try:
                    trace_event(task_id, 'tts_request', {'text_preview': remaining[:200]})
                    buf = tts_client.speak_to_buffer(remaining, 'vi', speed=1.0)
                    if buf:
                        fname, path = _write_audio_buffer_to_file(buf, prefix=f"full_{task_id}")
                        url = f"/audio/{fname}"
                        audio_urls.append(url)
                        sent_texts.add(remaining)  # Add to set
                        _safe_update_task(
                            task_id, 
                            full_audio_urls=audio_urls.copy(), 
                            full_text=final_text, 
                            last_part_ready_at=_now()
                        )
                        logger.debug(f"[BG] Final remaining part audio ready -> {fname}")
                        trace_event(task_id, 'audio_created', {'type': 'final_remaining', 'file': fname, 'url': url})
                except Exception:
                    logger.exception(f"[BG] TTS for remaining failed for id={task_id}")
                    trace_event(task_id, 'audio_error', {'type': 'final_remaining'})
            elif remaining:
                logger.debug(f"[BG] SKIP duplicate remaining text: {remaining[:50]}...")

        except Exception:
            trace_event(task_id, 'llm_error', {})
            logger.exception(f"[BG] Streaming LLM failed for id={task_id}, falling back to batch LLM")
            # fallback to non-stream LLM then chunk+TTS
            try:
                resp = hf_client.chat_completion(messages, max_tokens=512, temperature=0.7, top_p=0.95, stream=False)
                if hasattr(resp, 'choices'):
                    final_text = ''.join([ (c.get('message') or {}).get('content','') if isinstance(c, dict) else getattr(c, 'message', {}).get('content','') for c in resp.choices ])
                elif isinstance(resp, dict):
                    final_text = resp.get('content') or resp.get('text')
                else:
                    final_text = str(resp)
            except Exception:
                logger.exception(f"[BG] batch LLM failed for id={task_id}")
                final_text = get_quick_response_llm(question) or "Không thể tạo phản hồi lúc này."

            trace_event(task_id, 'llm_complete', {'final_text_preview': (final_text or '')})

            # Fallback: split full_text into sentences by punctuation and TTS per sentence
            import re as _re
            sentences = [s.strip() for s in _re.findall(r'[^.!?\n]+[.!?]+["\')\]]*\s*', final_text)]
            if not sentences:
                sentences = [final_text]
            logger.debug(f"[BG] Chunked into {len(sentences)} sentences for id={task_id} (fallback)")
            for idx, part in enumerate(sentences):
                if not part.strip():
                    continue
                try:
                    logger.debug(f"[BG] TTS sentence {idx+1}/{len(sentences)} for id={task_id}")
                    buf = tts_client.speak_to_buffer(part, 'vi', speed=1.0)
                    if buf:
                        fname, path = _write_audio_buffer_to_file(buf, prefix=f"full_{task_id}_{idx}")
                        url = f"/audio/{fname}"
                        audio_urls.append(url)
                        _safe_update_task(task_id, full_audio_urls=audio_urls.copy(), full_text=final_text, last_part_ready_at=_now())
                        logger.debug(f"[BG] sentence {idx} audio ready -> {fname}")
                        trace_event(task_id, 'audio_created', {'type': 'full_sentence', 'file': fname, 'url': url})
                except Exception:
                    logger.exception(f"[BG] TTS sentence {idx} failed for id={task_id}")

        # mark finished
        _safe_update_task(task_id, status='full_ready', full_audio_urls=audio_urls, full_finished_at=_now())
        logger.debug(f"[BG] Full processing finished for id={task_id} with {len(audio_urls)} parts")

        # Create merged audio file for smoother playback (reduce network/seek gaps) -> xóa do có thể gây conflict
        # try:
        #     files = []
        #     for u in audio_urls:
        #         if u.startswith('/audio/'):
        #             fname = u.split('/').pop()
        #             files.append(os.path.join(TMP_AUDIO_DIR, fname))
        #     if files:
        #         merged_name = f"full_merged_{task_id}.wav"
        #         merged_path = os.path.join(TMP_AUDIO_DIR, merged_name)
        #         _merge_audio_files(files, merged_path, crossfade_ms=30)
        #         merged_url = f"/audio/{merged_name}"
        #         _safe_update_task(task_id, full_merged_url=merged_url, merged_created_at=_now())
        #         logger.debug(f"[BG] Created merged full audio for id={task_id} -> {merged_name}")
        # except Exception:
        #     logger.exception(f"[BG] Failed to create merged audio for id={task_id}")

    except Exception:
        logger.exception(f"[BG] Unexpected error in background for id={task_id}")
        _safe_update_task(task_id, status='failed')

@app.post("/transcribe_base64")
def transcribe_base64(payload: dict):
    """
    Transcribe audio from base64 encoded string
    """
    try:
        audio_base64 = payload.get("audio_data")
        sample_rate = payload.get("sample_rate", 16000)
        stream = payload.get("stream", False)
        mime_type = payload.get("mime_type", None)
        
        if not audio_base64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Use new transcribe_base64 method
        asr_client = get_asr_client()
        result = asr_client.transcribe_base64(audio_base64, sample_rate, stream, mime_type)
        return result
        
    except Exception as e:
        logger.exception(f"Base64 transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...), stream: bool = Form(False)):
    """
    Transcribe audio file to text
    """
    try:
        logger.debug(f"Transcribing audio file: {file.filename}")
        
        # Save temporarily
        temp_path = os.path.join(TMP_AUDIO_DIR, f"transcribe_{uuid.uuid4().hex[:8]}.wav")
        with open(temp_path, "wb") as f:
            content = file.file.read()
            f.write(content)
        
        # Transcribe
        asr_client = get_asr_client()
        result = asr_client.transcribe_file(temp_path, stream=stream)
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_bytes")
def transcribe_bytes(payload: dict):
    """
    Transcribe audio from base64 encoded bytes
    """
    try:
        import base64
        audio_b64 = payload.get("audio_data")
        sample_rate = payload.get("sample_rate", 16000)
        stream = payload.get("stream", False)
        mime_type = payload.get("mime_type", None)
        if not audio_b64:
            raise HTTPException(status_code=400, detail="No audio data provided")
        # Decode base64
        audio_bytes = base64.b64decode(audio_b64)
        # Transcribe
        asr_client = get_asr_client()
        result = asr_client.transcribe_bytes(audio_bytes, sample_rate, stream, mime_type)
        return result
    except Exception as e:
        logger.exception(f"Bytes transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    index_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "msg": "Simple server running"})


@app.post("/ask")
def ask(req: AskRequest):
    q = req.question.strip()
    logger.debug(f"/ask received question='{q}'")
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Create task id and set initial task state
    task_id = uuid.uuid4().hex[:12]
    logger.debug(f"/ask assigned id={task_id}")

    # Synchronously get quick text (fast; keep UX snappy)
    try:
        quick_text = get_quick_response_llm(q)
        logger.debug(f"Quick response length={len(quick_text) if quick_text else 0}")
    except Exception:
        logger.exception("Quick LLM failed")
        quick_text = None

    # initialize task record
    now = time.time()
    with tasks_lock:
        tasks[task_id] = {
            'id': task_id,
            'question': q,
            'created': now,
            'status': 'quick_ready' if quick_text else 'pending',
            'quick_text': quick_text,
            'quick_audio_url': None,
            'full_text': None,
            'full_audio_urls': [],
        }
    # trace start
    trace_event(task_id, 'start', {'question': q, 'created_ms': int(now*1000)})
    if quick_text:
        trace_event(task_id, 'quick_text_ready', {'quick_text': quick_text})

    # spawn background worker to generate quick audio (if any) and full processing
    executor.submit(_background_process_all, task_id, q)

    # return quick text + task id immediately
    return {"id": task_id, "text": quick_text}

LOG_RETRIEVAL_FILE = "logs/retrieval.log"
def log_retrieval(question: str, chunks, context: str):
    try:
        with open(LOG_RETRIEVAL_FILE, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"QUESTION:\n{question}\n\n")
            f.write(f"CHUNKS RETRIEVED: {len(chunks) if chunks else 0}\n\n")
            f.write("CONTEXT RETRIEVED:\n")
            f.write(context)
            f.write("\n")
    except Exception:
        logger.exception("Failed to write retrieval log")

@app.post("/ask_full")
def ask_full(req: AskRequest):
    q = req.question.strip()
    logger.debug(f"/ask_full received question='{q}'")
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Retrieval
    try:
        chunks = retrieve_from_qdrant(
            q,
            top_k=TOP_K_VECTOR * 3,
            exclude_ids=set(),
            force_no_filter=False
        )
        logger.debug(f"retrieve_from_qdrant returned {len(chunks) if chunks else 0} chunks")
    except Exception:
        logger.exception("Retrieval failed")
        chunks = []

    # Build context & messages
    try:
        context, messages = build_context_and_prompt(chunks, q)
        logger.debug(f"Built context, messages count={len(messages)}")

        # 🔽 LOG QUESTION + CONTEXT RA FILE RIÊNG
        log_retrieval(q, chunks, context)

    except Exception:
        logger.exception("build_context_and_prompt failed")
        messages = []
        context = ""

    # LLM call (non-stream). Try hf_client first, fallback to quick
    final_text = None
    try:
        logger.debug("Calling hf_client.chat_completion (non-stream) ...")
        resp = hf_client.chat_completion(
            messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stream=False
        )

        if hasattr(resp, 'choices'):
            final_text = ''.join([
                c.get('message', {}).get('content', '')
                if isinstance(c, dict)
                else getattr(c, 'message', {}).get('content', '')
                for c in resp.choices
            ])
        elif isinstance(resp, dict):
            final_text = resp.get('content') or resp.get('text')
        else:
            final_text = str(resp)

        logger.debug(f"LLM final_text length={len(final_text) if final_text else 0}")

    except Exception:
        logger.exception("hf_client.chat_completion failed, falling back to quick response")
        try:
            final_text = get_quick_response_llm(q)
        except Exception:
            logger.exception("Fallback quick response failed")
            final_text = "Không thể tạo phản hồi lúc này."

    # TTS final_text
    audio_url = None
    try:
        buf = tts_client.speak_to_buffer(final_text, 'vi', speed=1.0)
        if buf:
            fname, path = _write_audio_buffer_to_file(buf, prefix="full")
            audio_url = f"/audio/{fname}"
    except Exception:
        logger.exception("TTS for final_text failed")

    return {"text": final_text, "audio_url": audio_url}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    with tasks_lock:
        t = tasks.get(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    # return task summary (do not expose internal buffers)
    return {
        'id': t.get('id'),
        'status': t.get('status'),
        'quick_text': t.get('quick_text'),
        'quick_audio_url': t.get('quick_audio_url'),
        'full_text': t.get('full_text'),
        'full_audio_urls': t.get('full_audio_urls', []),
        'full_merged_url': t.get('full_merged_url')
    }


@app.get("/audio/{fname}")
def get_audio(fname: str):
    path = os.path.join(TMP_AUDIO_DIR, fname)
    logger.debug(f"GET /audio/{fname} -> path={path}")
    if not os.path.exists(path):
        logger.debug("File not found")
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/wav")


@app.delete("/audio/{fname}")
def delete_audio(fname: str):
    path = os.path.join(TMP_AUDIO_DIR, fname)
    logger.debug(f"DELETE /audio/{fname} -> path={path}")
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug("Deleted audio file")
            return {"deleted": True}
        else:
            return {"deleted": False, "reason": "not_exist"}
    except Exception:
        logger.exception("Failed to delete audio")
        raise HTTPException(status_code=500, detail="Failed to delete")


@app.post("/report_play")
def report_play(payload: dict):
    # payload: { task_id, url, event: 'start'|'end', ts(optional ms) }
    task_id = payload.get('task_id')
    url = payload.get('url')
    event = payload.get('event')
    ts = payload.get('ts') or _now()
    logger.debug(f"REPORT_PLAY | task={task_id} | url={url} | event={event} | ts={ts}")
    if not task_id or not url or event not in ('start', 'end'):
        raise HTTPException(status_code=400, detail="invalid payload")
    with tasks_lock:
        t = tasks.get(task_id)
        if not t:
            raise HTTPException(status_code=404, detail="task not found")
        # record
        plays = t.get('plays', [])
        plays.append({'url': url, 'event': event, 'ts': ts})
        t['plays'] = plays
        tasks[task_id] = t
    # After recording, check if all expected audio files have an 'end' event
    with tasks_lock:
        t = tasks.get(task_id)
        if t.get('completed'):
            return {'ok': True}
        expected = set()
        if t.get('quick_audio_url'):
            expected.add(t.get('quick_audio_url'))
        for u in t.get('full_audio_urls', []):
            expected.add(u)
        if t.get('full_merged_url'):
            expected = {t.get('full_merged_url')}

        # if no expected audio, consider completed when status is full_ready
        if not expected:
            if t.get('status') == 'full_ready':
                t['completed'] = True
                t['completed_at'] = _now()
                trace_event(task_id, 'task_end', {'reason': 'no_audio', 'completed_at': t['completed_at']})
                tasks[task_id] = t
                return {'ok': True}

        # check plays for end events per expected url
        ended_urls = set(p['url'] for p in t.get('plays', []) if p.get('event') == 'end')
        if expected.issubset(ended_urls):
            t['completed'] = True
            t['completed_at'] = _now()
            # compute duration from start
            start_ms = int(t.get('created', 0) * 1000)
            duration_ms = t['completed_at'] - start_ms if start_ms else None
            trace_event(task_id, 'task_end', {'completed_at': t['completed_at'], 'duration_ms': duration_ms, 'ended_urls': list(ended_urls)})
            tasks[task_id] = t
    return {'ok': True}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting simple_server via uvicorn on 0.0.0.0:8000")
    uvicorn.run("simple_server:app", host="0.0.0.0", port=8000, log_level="info")
