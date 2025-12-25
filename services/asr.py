# # file name: services/asr.py
# import io
# import time
# import requests
# import soundfile as sf
# import numpy as np
# from typing import Optional, List, Dict, Any
# import logging
# import wave
# import struct
# import base64

# logger = logging.getLogger(__name__)

# class ASRClient:
#     """
#     Client for Automatic Speech Recognition (ASR) service
#     Supports both streaming and file-based transcription
#     """
    
#     def __init__(self, 
#                  api_url: str = "https://Hoangnam5904-STT.hf.space/transcribe_file",
#                  chunk_duration: float = 3.0,
#                  overlap: float = 0.25,
#                  language: str = "vi"):
#         """
#         Initialize ASR client
        
#         Args:
#             api_url: Endpoint for ASR service
#             chunk_duration: Duration of each audio chunk in seconds
#             overlap: Overlap between chunks in seconds
#             language: Language code for transcription
#         """
#         self.api_url = api_url
#         self.chunk_duration = chunk_duration
#         self.overlap = overlap
#         self.language = language
#         self.step = chunk_duration - overlap
        
#         logger.info(f"ASR client initialized with URL: {api_url}")
    
#     def _split_audio(self, data: np.ndarray, sr: int) -> List[np.ndarray]:
#         """
#         Split audio into chunks with overlap
#         """
#         chunk_samples = int(self.chunk_duration * sr)
#         step_samples = int(self.step * sr)
        
#         chunks = []
#         i = 0
        
#         while i + chunk_samples <= len(data):
#             chunk = data[i:i + chunk_samples]
#             chunks.append(chunk)
#             i += step_samples
        
#         remaining = len(data) - i
#         if remaining > 0:
#             # Nếu còn lại >= 1s thì xử lý riêng
#             if remaining >= sr:  # >= 1 second
#                 # Lấy phần còn lại chính xác thay vì overlap
#                 last_chunk = data[i:]
#                 chunks.append(last_chunk)
#             else:
#                 # Nếu < 1s thì overlap với chunk trước (tránh mất âm cuối câu)
#                 if len(chunks) > 0:
#                     # Extend chunk cuối cùng thay vì tạo chunk mới
#                     last_idx = len(chunks) - 1
#                     overlap_start = max(0, len(data) - chunk_samples)
#                     chunks[last_idx] = data[overlap_start:]
        
#         return chunks
    
#     def _bytes_to_wav(self, audio_bytes: bytes, sr: int = 16000) -> bytes:
#         """
#         Convert raw audio bytes to WAV format
#         """
#         # Try to detect format
#         if audio_bytes[:4] == b'RIFF':
#             # Already WAV
#             return audio_bytes
        
#         # Assume raw PCM (16-bit mono)
#         # Create WAV header
#         num_channels = 1
#         sampwidth = 2  # 16-bit
#         nframes = len(audio_bytes) // sampwidth
        
#         # Create in-memory WAV file
#         buffer = io.BytesIO()
#         with wave.open(buffer, 'wb') as wav_file:
#             wav_file.setnchannels(num_channels)
#             wav_file.setsampwidth(sampwidth)
#             wav_file.setframerate(sr)
#             wav_file.writeframes(audio_bytes)
        
#         buffer.seek(0)
#         return buffer.read()
    
#     def _webm_to_wav(self, webm_bytes: bytes, sr: int = 16000) -> Optional[bytes]:
#         """
#         WebM/Opus conversion - no longer supported without ffmpeg
#         Return None to let API handle the raw bytes
#         """
#         logger.warning("WebM conversion skipped - sending raw bytes to API")
#         return None
    
#     def _decode_audio_bytes(self, audio_bytes: bytes, mime_type: str = None) -> Optional[tuple]:
#         """
#         Decode audio bytes to (data, sample_rate) tuple
#         Returns None if cannot decode
#         """
#         buffer = io.BytesIO(audio_bytes)
        
#         # Try soundfile first
#         try:
#             data, sr = sf.read(buffer)
#             return data, sr
#         except Exception as e:
#             logger.debug(f"Soundfile read failed: {e}")
        
#         # Try wave module for WAV files
#         buffer.seek(0)
#         try:
#             if audio_bytes[:4] == b'RIFF' or (mime_type and 'wav' in mime_type):
#                 with wave.open(buffer, 'rb') as wav:
#                     sr = wav.getframerate()
#                     n_frames = wav.getnframes()
#                     audio_data = wav.readframes(n_frames)
#                     # Convert bytes to numpy array
#                     dtype = np.int16 if wav.getsampwidth() == 2 else np.int8
#                     data = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 32768.0
#                     return data, sr
#         except Exception as e:
#             logger.debug(f"Wave read failed: {e}")
        
#         # If WebM/Opus and pydub available, try conversion
#         buffer.seek(0)
#         if mime_type and ('webm' in mime_type or 'opus' in mime_type or 'ogg' in mime_type):
#             wav_bytes = self._webm_to_wav(audio_bytes)
#             if wav_bytes:
#                 return self._decode_audio_bytes(wav_bytes, 'audio/wav')
        
#         return None
    
#     def transcribe_file(self, 
#                        audio_path: str, 
#                        stream: bool = False) -> Dict[str, Any]:
#         """
#         Transcribe an audio file
#         """
#         try:
#             logger.debug(f"Transcribing file: {audio_path}")
            
#             # Load audio using soundfile
#             data, sr = sf.read(audio_path)
            
#             # Convert to mono if stereo
#             if len(data.shape) > 1:
#                 data = data.mean(axis=1)
            
#             if stream:
#                 return self._transcribe_streaming(data, sr)
#             else:
#                 return self._transcribe_single(data, sr)
                
#         except Exception as e:
#             logger.error(f"Error transcribing file {audio_path}: {e}")
#             return {"text": "", "error": str(e), "success": False}
    
#     def _transcribe_single(self, data: np.ndarray, sr: int) -> Dict[str, Any]:
#         """
#         Transcribe entire audio as a single request
#         """
#         try:
#             # Create in-memory WAV file
#             buf = io.BytesIO()
#             sf.write(buf, data, sr, format='WAV')
#             buf.seek(0)
            
#             # Send to ASR API
#             response = requests.post(
#                 self.api_url,
#                 files={"file": ("audio.wav", buf, "audio/wav")},
#                 data={"language": self.language},
#                 timeout=30
#             )
            
#             if response.status_code == 200:
#                 result = response.json()
#                 return {
#                     "text": result.get("text", ""),
#                     "language": result.get("language", self.language),
#                     "success": True
#                 }
#             else:
#                 logger.error(f"ASR API error: {response.status_code}")
#                 return {
#                     "text": "",
#                     "error": f"API error: {response.status_code}",
#                     "success": False
#                 }
                
#         except requests.exceptions.RequestException as e:
#             logger.error(f"Request failed: {e}")
#             return {"text": "", "error": str(e), "success": False}
    
#     def _transcribe_streaming(self, data: np.ndarray, sr: int) -> Dict[str, Any]:
#         """
#         Transcribe audio by streaming chunks sequentially
#         """
#         try:
#             chunks = self._split_audio(data, sr)
#             logger.debug(f"Split audio into {len(chunks)} chunks")
            
#             all_texts = []
            
#             for i, chunk in enumerate(chunks):
#                 # Create chunk buffer
#                 buf = io.BytesIO()
#                 sf.write(buf, chunk, sr, format='WAV')
#                 buf.seek(0)
                
#                 # Send chunk
#                 response = requests.post(
#                     self.api_url,
#                     files={"file": (f"chunk_{i}.wav", buf, "audio/wav")},
#                     data={"language": self.language},
#                     timeout=10
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     chunk_text = result.get("text", "")
#                     if chunk_text:
#                         all_texts.append(chunk_text)
#                         logger.debug(f"Chunk {i}: {chunk_text[:50]}...")
                
#                 # Simulate continuous speech
#                 if i < len(chunks) - 1:
#                     time.sleep(self.step)
            
#             final_text = " ".join(all_texts)
#             return {
#                 "text": final_text,
#                 "chunks": len(chunks),
#                 "language": self.language,
#                 "success": True
#             }
            
#         except Exception as e:
#             logger.error(f"Streaming transcription failed: {e}")
#             return {"text": "", "error": str(e), "success": False}
    
#     def transcribe_bytes(self, 
#                         audio_bytes: bytes, 
#                         sample_rate: int = 16000,
#                         stream: bool = False,
#                         mime_type: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Transcribe audio from bytes - SIMPLIFIED VERSION
#         """
#         try:
#             logger.debug(f"Transcribing bytes, length: {len(audio_bytes)}, mime: {mime_type}")
            
#             # Try to decode the audio
#             decoded = self._decode_audio_bytes(audio_bytes, mime_type)
            
#             if decoded is None:
#                 # If cannot decode, try to send raw bytes directly to API
#                 logger.warning(f"Cannot decode audio, sending raw bytes to API")
                
#                 # Determine file extension
#                 ext = "wav"
#                 if mime_type:
#                     if "webm" in mime_type:
#                         ext = "webm"
#                     elif "ogg" in mime_type:
#                         ext = "ogg"
#                     elif "mp3" in mime_type:
#                         ext = "mp3"
                
#                 response = requests.post(
#                     self.api_url,
#                     files={"file": (f"audio.{ext}", audio_bytes, mime_type or "audio/wav")},
#                     data={"language": self.language},
#                     timeout=30
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     return {
#                         "text": result.get("text", ""),
#                         "language": result.get("language", self.language),
#                         "success": True
#                     }
#                 else:
#                     return {
#                         "text": "",
#                         "error": f"API rejected audio with status {response.status_code}",
#                         "success": False
#                     }
            
#             data, sr = decoded
            
#             # Use provided sample rate or detected one
#             sr = sample_rate if sample_rate else sr
            
#             # Convert to mono if stereo
#             if len(data.shape) > 1:
#                 data = data.mean(axis=1)
            
#             if stream:
#                 return self._transcribe_streaming(data, sr)
#             else:
#                 return self._transcribe_single(data, sr)
                
#         except Exception as e:
#             logger.error(f"Error transcribing bytes: {e}")
#             return {"text": "", "error": str(e), "success": False}
    
#     def transcribe_base64(self,
#                          audio_base64: str,
#                          sample_rate: int = 16000,
#                          stream: bool = False,
#                          mime_type: Optional[str] = None) -> Dict[str, Any]:
#         """
#         Transcribe audio from base64 string
#         """
#         try:
#             # Decode base64
#             audio_bytes = base64.b64decode(audio_base64)
#             return self.transcribe_bytes(audio_bytes, sample_rate, stream, mime_type)
#         except Exception as e:
#             logger.error(f"Error decoding base64: {e}")
#             return {"text": "", "error": str(e), "success": False}


# # Singleton instance
# _asr_client = None

# def get_asr_client() -> ASRClient:
#     """Get or create ASR client instance"""
#     global _asr_client
#     if _asr_client is None:
#         _asr_client = ASRClient()
#     return _asr_client

from google.cloud import speech
import queue
import threading
import logging
import io
import base64
import soundfile as sf
import numpy as np
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)

class GoogleStreamingASR:
    """
    Optimized Google Cloud Speech-to-Text streaming client
    - Uses enhanced model for better accuracy and speed
    - Optimized for low latency
    - Supports real-time transcription with interim results
    """
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 lang: str = "vi-VN",
                 use_enhanced: bool = True):
        """
        Initialize Google Streaming ASR
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            lang: Language code (default: vi-VN for Vietnamese)
            use_enhanced: Use enhanced model for better quality (default: True)
        """
        self.client = speech.SpeechClient()
        self.audio_queue = queue.Queue()
        self.sample_rate = sample_rate
        self.lang = lang
        self.is_running = False
        self._lock = threading.Lock()

        # Optimized config for low latency and high accuracy
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code=lang,
            # Enable enhanced model for better speed and accuracy
            use_enhanced=use_enhanced,
            # Use latest_short model for lowest latency
            model="latest_short" if not use_enhanced else "default",
            # Enable automatic punctuation for better results
            enable_automatic_punctuation=True,
            # Enable word time offsets for debugging (optional)
            # enable_word_time_offsets=True,
            # Audio channel count
            audio_channel_count=1,
        )

        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
            # Single utterance mode off for continuous streaming
            single_utterance=False,
        )
        
        logger.info(f"GoogleStreamingASR initialized: lang={lang}, sr={sample_rate}, enhanced={use_enhanced}")

    def audio_generator(self):
        """Generator that yields audio chunks from queue"""
        while True:
            chunk = self.audio_queue.get()
            if chunk is None:
                logger.debug("Audio generator received stop signal")
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    def push_audio(self, pcm_bytes: bytes):
        """
        Push audio chunk to processing queue
        
        Args:
            pcm_bytes: Raw PCM audio bytes (LINEAR16)
        """
        if not self.is_running:
            logger.warning("ASR not running, audio chunk ignored")
            return
            
        with self._lock:
            self.audio_queue.put(pcm_bytes)

    def stop(self):
        """Stop the ASR processing"""
        logger.info("Stopping ASR...")
        with self._lock:
            self.is_running = False
            self.audio_queue.put(None)

    def run(self, 
            on_partial: Callable[[str], None], 
            on_final: Callable[[str], None],
            on_error: Optional[Callable[[Exception], None]] = None):
        """
        Start streaming recognition
        
        Args:
            on_partial: Callback for interim results (快速反馈)
            on_final: Callback for final results (确认结果)
            on_error: Optional callback for errors
        """
        self.is_running = True
        logger.info("Starting streaming recognition...")
        
        try:
            requests = self.audio_generator()
            responses = self.client.streaming_recognize(
                self.streaming_config,
                requests
            )

            for response in responses:
                if not response.results:
                    continue
                    
                for result in response.results:
                    if not result.alternatives:
                        continue
                        
                    text = result.alternatives[0].transcript
                    confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0.0
                    
                    if result.is_final:
                        logger.debug(f"Final result: {text} (confidence: {confidence:.2f})")
                        on_final(text)
                    else:
                        logger.debug(f"Interim result: {text}")
                        on_partial(text)
                        
        except Exception as e:
            logger.error(f"Streaming recognition error: {e}")
            if on_error:
                on_error(e)
            else:
                raise
        finally:
            self.is_running = False
            logger.info("Streaming recognition stopped")

    def run_async(self,
                  on_partial: Callable[[str], None],
                  on_final: Callable[[str], None],
                  on_error: Optional[Callable[[Exception], None]] = None) -> threading.Thread:
        """
        Run ASR in a separate thread for non-blocking operation
        
        Returns:
            Thread object running the ASR
        """
        thread = threading.Thread(
            target=self.run,
            args=(on_partial, on_final, on_error),
            daemon=True
        )
        thread.start()
        logger.info("ASR thread started")
        return thread

    def transcribe_file(self, 
                       audio_path: str, 
                       stream: bool = False) -> Dict[str, Any]:
        """
        Transcribe an audio file using Google Cloud Speech-to-Text (batch mode)
        
        Args:
            audio_path: Path to audio file
            stream: If True, use streaming (currently ignored, uses batch)
            
        Returns:
            Dict with 'text', 'success', and optional 'error' keys
        """
        try:
            logger.debug(f"Transcribing file: {audio_path}")
            
            # Detect sample rate from audio file
            try:
                data, detected_sr = sf.read(audio_path)
                logger.debug(f"Detected sample rate: {detected_sr} Hz")
                
                # Create config with detected sample rate
                file_config = speech.RecognitionConfig(
                    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=detected_sr,
                    language_code=self.lang,
                    use_enhanced=True,
                    model="latest_short",
                    enable_automatic_punctuation=True,
                    audio_channel_count=1 if len(data.shape) == 1 else data.shape[1],
                )
            except Exception as e:
                logger.warning(f"Could not detect sample rate, using config default: {e}")
                file_config = self.config
            
            # Load audio
            with io.open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            
            # Use synchronous recognition for file-based transcription
            audio = speech.RecognitionAudio(content=content)
            
            response = self.client.recognize(
                config=file_config,
                audio=audio
            )
            
            # Combine all results
            transcript = ' '.join([
                result.alternatives[0].transcript 
                for result in response.results
                if result.alternatives
            ])
            
            logger.debug(f"Transcription result: {transcript[:100]}...")
            
            return {
                "text": transcript,
                "language": self.lang,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error transcribing file {audio_path}: {e}")
            return {"text": "", "error": str(e), "success": False}
    
    def transcribe_bytes(self, 
                        audio_bytes: bytes, 
                        sample_rate: int = 16000,
                        stream: bool = False,
                        mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from bytes using Google Cloud Speech-to-Text
        
        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Sample rate (ignored, uses config)
            stream: If True, use streaming (currently ignored)
            mime_type: MIME type of audio (ignored)
            
        Returns:
            Dict with 'text', 'success', and optional 'error' keys
        """
        try:
            logger.debug(f"Transcribing bytes, length: {len(audio_bytes)}")
            
            # Use synchronous recognition
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            response = self.client.recognize(
                config=self.config,
                audio=audio
            )
            
            # Combine all results
            transcript = ' '.join([
                result.alternatives[0].transcript 
                for result in response.results
                if result.alternatives
            ])
            
            return {
                "text": transcript,
                "language": self.lang,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error transcribing bytes: {e}")
            return {"text": "", "error": str(e), "success": False}
    
    def transcribe_base64(self,
                         audio_base64: str,
                         sample_rate: int = 16000,
                         stream: bool = False,
                         mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from base64 string
        
        Args:
            audio_base64: Base64 encoded audio
            sample_rate: Sample rate (ignored)
            stream: If True, use streaming (ignored)
            mime_type: MIME type (ignored)
            
        Returns:
            Dict with 'text', 'success', and optional 'error' keys
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            return self.transcribe_bytes(audio_bytes, sample_rate, stream, mime_type)
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return {"text": "", "error": str(e), "success": False}


# Singleton instance
_google_asr_client = None

def get_asr_client(sample_rate: int = 16000, lang: str = "vi-VN") -> GoogleStreamingASR:
    """Get or create Google ASR client instance"""
    global _google_asr_client
    if _google_asr_client is None:
        _google_asr_client = GoogleStreamingASR(sample_rate=sample_rate, lang=lang)
    return _google_asr_client


# Example usage callbacks
def on_partial(text: str):
    """Callback for interim results"""
    print(f"⏳ {text}")

def on_final(text: str):
    """Callback for final results"""
    print(f"✅ {text}")

def on_error(error: Exception):
    """Callback for errors"""
    print(f"❌ Error: {error}")


# Example usage:
# asr = get_asr_client()
# # Start async
# thread = asr.run_async(on_partial, on_final, on_error)
# # Push audio from mic/websocket/rtc
# asr.push_audio(pcm_chunk)
# # Stop when done
# asr.stop()
# thread.join()
