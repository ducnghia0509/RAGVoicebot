# realtime/handler.py
from retrieval.searcher import retrieve_from_qdrant
from config import *
from utils.helpers import count_words
from utils.logger import timing_logger
import time

used_chunk_ids = set()
realtime_shown_chunks = []
realtime_shown_chunk_ids = set()
last_realtime_time = 0
last_word_count = 0
MAX_REALTIME_BUFFER = 50

def realtime_search_on_word_count(input_text: str):
    global last_realtime_time, last_word_count, realtime_shown_chunks, realtime_shown_chunk_ids

    current_time = time.time()
    current_word_count = count_words(input_text)

    if (current_word_count < MIN_INPUT_LENGTH_TO_RETRIEVAL or
        current_word_count - last_word_count < MIN_INPUT_LENGTH_TO_RETRIEVAL or
        current_time - last_realtime_time < DEBOUNCE_TIME):
        return

    last_realtime_time = current_time
    last_word_count = current_word_count

    chunks = retrieve_from_qdrant(input_text, TOP_K_VECTOR, used_chunk_ids)

    for c in chunks:
        cid = c["metadata"].get("chunk_id")
        if cid and cid not in used_chunk_ids and cid not in realtime_shown_chunk_ids:
            realtime_shown_chunks.append(c)
            realtime_shown_chunk_ids.add(cid)

    if len(realtime_shown_chunks) > MAX_REALTIME_BUFFER:
        realtime_shown_chunks = realtime_shown_chunks[-MAX_REALTIME_BUFFER:]
        realtime_shown_chunk_ids = {c["metadata"]["chunk_id"] for c in realtime_shown_chunks if c["metadata"]["chunk_id"]}

def reset_realtime():
    global realtime_shown_chunks, realtime_shown_chunk_ids, last_word_count, last_realtime_time
    realtime_shown_chunks.clear()
    realtime_shown_chunk_ids.clear()
    last_word_count = last_realtime_time = 0