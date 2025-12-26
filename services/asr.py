from google.cloud import speech
import queue
import threading
import logging
import io
import base64
import soundfile as sf
import numpy as np
from typing import Callable, Optional, Dict, Any
from scipy import signal

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
            # sample_rate_hertz removed - will be set per request based on actual audio
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

    def resample_audio(self, 
                      audio_bytes: bytes, 
                      original_sample_rate: int,
                      target_sample_rate: int = 16000) -> Optional[bytes]:
        """
        Resample audio from original sample rate to target sample rate
        
        Args:
            audio_bytes: Raw audio bytes (any format)
            original_sample_rate: Original sample rate in Hz
            target_sample_rate: Target sample rate in Hz (default: 16000)
            
        Returns:
            Resampled audio bytes in WAV format, or None if failed
        """
        try:
            # Tạo file tạm thời để xử lý
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Đọc audio bytes
                if audio_bytes[:4] == b'RIFF':
                    # Nếu đã là WAV, ghi thẳng ra file
                    with open(tmp_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Đọc bằng soundfile
                    data, sr = sf.read(tmp_path)
                else:
                    # Giả định là raw PCM 16-bit mono
                    dtype = np.int16
                    data = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32) / 32768.0
                    sr = original_sample_rate
                
                # Kiểm tra nếu cần resample
                if sr == target_sample_rate:
                    logger.debug(f"No resampling needed: {sr}Hz == {target_sample_rate}Hz")
                    return audio_bytes
                
                # Tính tỷ lệ resample
                resample_ratio = target_sample_rate / sr
                num_samples = int(len(data) * resample_ratio)
                
                # Thực hiện resample
                resampled_data = signal.resample(data, num_samples)
                
                # Chuyển về dạng int16 cho WAV
                resampled_int16 = (resampled_data * 32767).astype(np.int16)
                
                # Tạo WAV file trong memory
                with io.BytesIO() as wav_buffer:
                    with sf.SoundFile(wav_buffer, mode='w', samplerate=target_sample_rate,
                                     channels=1, format='WAV', subtype='PCM_16') as audio_file:
                        audio_file.write(resampled_int16)
                    
                    wav_buffer.seek(0)
                    wav_bytes = wav_buffer.read()
                
                logger.info(f"Resampled audio: {sr}Hz → {target_sample_rate}Hz")
                return wav_bytes
                
            finally:
                # Xóa file tạm
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"Error resampling audio: {e}")
            return None

    def transcribe_bytes_with_resample(self,
                                      audio_bytes: bytes,
                                      original_sample_rate: int,
                                      target_sample_rate: int = 16000,
                                      mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio with automatic resampling to target sample rate
        
        Args:
            audio_bytes: Raw audio bytes
            original_sample_rate: Original sample rate
            target_sample_rate: Target sample rate for ASR (default: 16000)
            mime_type: MIME type of audio
            
        Returns:
            Dict with transcription results
        """
        try:
            # Resample nếu cần
            if original_sample_rate != target_sample_rate:
                logger.info(f"Resampling from {original_sample_rate}Hz to {target_sample_rate}Hz")
                resampled_bytes = self.resample_audio(
                    audio_bytes, 
                    original_sample_rate, 
                    target_sample_rate
                )
                
                if resampled_bytes:
                    # Sử dụng audio đã resample
                    return self.transcribe_bytes(
                        resampled_bytes, 
                        target_sample_rate, 
                        False,  # stream
                        'audio/wav'  # MIME type sau khi resample luôn là WAV
                    )
            
            # Nếu không cần resample hoặc resample thất bại
            return self.transcribe_bytes(
                audio_bytes,
                original_sample_rate,
                False,
                mime_type
            )
            
        except Exception as e:
            logger.error(f"Error in transcribe_bytes_with_resample: {e}")
            return {"text": "", "error": str(e), "success": False}

    def transcribe_base64_with_resample(self,
                                       audio_base64: str,
                                       original_sample_rate: int,
                                       target_sample_rate: int = 16000,
                                       mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe base64 audio with automatic resampling
        
        Args:
            audio_base64: Base64 encoded audio
            original_sample_rate: Original sample rate
            target_sample_rate: Target sample rate (default: 16000)
            mime_type: MIME type
            
        Returns:
            Dict with transcription results
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            return self.transcribe_bytes_with_resample(
                audio_bytes,
                original_sample_rate,
                target_sample_rate,
                mime_type
            )
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return {"text": "", "error": str(e), "success": False}

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
            sample_rate: Sample rate of the audio
            stream: If True, use streaming (currently ignored)
            mime_type: MIME type of audio (used to detect encoding)
            
        Returns:
            Dict with 'text', 'success', and optional 'error' keys
        """
        try:
            logger.debug(f"Transcribing bytes, length: {len(audio_bytes)}, sample_rate: {sample_rate}, mime: {mime_type}")
            
            # Detect encoding from MIME type
            encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16  # Default
            if mime_type:
                if 'webm' in mime_type or 'opus' in mime_type:
                    encoding = speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
                elif 'ogg' in mime_type:
                    encoding = speech.RecognitionConfig.AudioEncoding.OGG_OPUS
                elif 'flac' in mime_type:
                    encoding = speech.RecognitionConfig.AudioEncoding.FLAC
                elif 'mp3' in mime_type:
                    encoding = speech.RecognitionConfig.AudioEncoding.MP3
            
            # For WEBM_OPUS, OGG_OPUS, FLAC - let Google auto-detect sample rate
            # Only set sample_rate_hertz for LINEAR16 and MP3
            config_params = {
                'encoding': encoding,
                'language_code': self.lang,
                'use_enhanced': self.config.use_enhanced,
                'model': self.config.model,
                'enable_automatic_punctuation': True,
                'audio_channel_count': 1,
            }
            
            # Only add sample_rate_hertz for formats that require it
            if encoding in [
                speech.RecognitionConfig.AudioEncoding.LINEAR16,
                speech.RecognitionConfig.AudioEncoding.MP3
            ]:
                config_params['sample_rate_hertz'] = sample_rate
            
            # Create config with dynamic encoding
            config = speech.RecognitionConfig(**config_params)
            
            logger.debug(f"Using encoding: {encoding}, sample_rate: {sample_rate if encoding == speech.RecognitionConfig.AudioEncoding.LINEAR16 else 'auto-detect'}")
            
            # Use synchronous recognition
            audio = speech.RecognitionAudio(content=audio_bytes)
            
            response = self.client.recognize(
                config=config,  # Use dynamic config
                audio=audio
            )
            
            # Combine all results
            transcript = ' '.join([
                result.alternatives[0].transcript 
                for result in response.results
                if result.alternatives
            ])
            
            # Check if transcript is empty (silent audio or no speech)
            if not transcript or not transcript.strip():
                logger.warning("No speech detected in audio")
                return {
                    "text": "",
                    "language": self.lang,
                    "success": False,
                    "error": "No speech detected in audio"
                }
            
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
