# TTSClient.py
import io
import time
import logging
import requests
from typing import Optional, Literal

logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, endpoint: str, max_retries: int = 2):
        self.endpoint = endpoint.rstrip("/")
        self.max_retries = max_retries
        self.session = requests.Session()

        # Rate limit
        self._last_call_time = 0
        self._min_interval = 0.1

        # Failure handling
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
        self._cooldown_until = 0

        logger.info(f"TTS Client initialized (FastAPI streaming): {self.endpoint}")

    def _rate_limit_wait(self):
        now = time.time()
        if now < self._cooldown_until:
            wait_time = self._cooldown_until - now
            logger.warning(f"TTS cooldown {wait_time:.1f}s")
            time.sleep(wait_time)
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def _enter_cooldown(self, duration=8.0):
        self._cooldown_until = time.time() + duration
        logger.warning(f"Entering short cooldown {duration}s (failures={self._consecutive_failures})")

    def speak_to_buffer(
        self,
        text: str,
        language: Literal["vi", "en"] = "vi",
        speed: float = 1.5,  # tham số này giờ không dùng nữa, nhưng vẫn giữ để không lỗi code cũ
        timeout_total: float = 20.0,
        timeout_network: float = 5.0,
    ) -> Optional[io.BytesIO]:

        if not text or len(text.strip()) < 2:
            return None

        text = text.strip()
        start_time = time.time()

        if self._consecutive_failures < self._max_consecutive_failures:
            for attempt in range(self.max_retries):
                try:
                    self._rate_limit_wait()
                    if time.time() - start_time > timeout_total:
                        logger.warning("Hard timeout (total) reached before request")
                        break

                    logger.info(f"[FastAPI Streaming TTS] Attempt {attempt+1}/{self.max_retries}")
                    response = self.session.post(
                        f"{self.endpoint}/speak_stream",
                        json={"text": text, "lang": language},
                        stream=True,
                        timeout=timeout_network,
                    )

                    if response.status_code != 200:
                        raise Exception(f"Status {response.status_code}: {response.text}")

                    mp3_buf = io.BytesIO()
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            mp3_buf.write(chunk)

                    if mp3_buf.getbuffer().nbytes < 200:
                        raise Exception("Audio too small")

                    mp3_buf.seek(0)
                    self._consecutive_failures = 0
                    logger.info(f"Streaming TTS success → MP3 ({len(mp3_buf.getvalue())} bytes)")
                    return mp3_buf

                except requests.exceptions.ConnectionError:
                    logger.error("Local network error → skipping failure count")
                    time.sleep(0.3)
                    break

                except Exception as e:
                    logger.warning(f"FastAPI Streaming TTS error: {e}")
                    self._consecutive_failures += 1
                    if attempt < self.max_retries - 1:
                        time.sleep(0.8 + attempt * 0.5)
                    else:
                        if self._consecutive_failures >= self._max_consecutive_failures:
                            self._enter_cooldown(8)
                        break

        logger.info("Falling back to gTTS...")
        return self._gtts_fallback(text, language)

    def _gtts_fallback(self, text, language):
        from gtts import gTTS
        for attempt in range(2):
            try:
                tts = gTTS(text=text, lang=language)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                logger.info(f"gTTS fallback success → MP3 (attempt {attempt+1})")
                return buf
            except Exception as e:
                logger.error(f"gTTS failed: {e}")
                time.sleep(0.5)
        return None

    def health_check(self):
        status = {
            "api": "unknown",
            "gtts": "unknown",
            "consecutive_failures": self._consecutive_failures,
            "in_cooldown": time.time() < self._cooldown_until,
        }
        try:
            r = self.session.get(f"{self.endpoint}/")
            status["api"] = "ok" if r.status_code == 200 else "failed"
        except:
            status["api"] = "failed"

        try:
            g = self._gtts_fallback("test", "en")
            status["gtts"] = "ok" if g else "failed"
        except:
            status["gtts"] = "failed"

        return status


# Global instance (giữ nguyên như cũ)
tts_client = TTSClient("https://hoangnam5904-tts.hf.space")