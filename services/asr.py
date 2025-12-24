# file name: services/asr.py
import io
import time
import requests
import soundfile as sf
import numpy as np
from typing import Optional, List, Dict, Any
import logging
import wave
import struct
import base64

logger = logging.getLogger(__name__)

class ASRClient:
    """
    Client for Automatic Speech Recognition (ASR) service
    Supports both streaming and file-based transcription
    """
    
    def __init__(self, 
                 api_url: str = "https://Hoangnam5904-STT.hf.space/transcribe_file",
                 chunk_duration: float = 3.0,
                 overlap: float = 0.25,
                 language: str = "vi"):
        """
        Initialize ASR client
        
        Args:
            api_url: Endpoint for ASR service
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between chunks in seconds
            language: Language code for transcription
        """
        self.api_url = api_url
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.language = language
        self.step = chunk_duration - overlap
        
        logger.info(f"ASR client initialized with URL: {api_url}")
    
    def _split_audio(self, data: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split audio into chunks with overlap
        """
        chunk_samples = int(self.chunk_duration * sr)
        step_samples = int(self.step * sr)
        
        chunks = []
        i = 0
        
        while i + chunk_samples <= len(data):
            chunk = data[i:i + chunk_samples]
            chunks.append(chunk)
            i += step_samples
        
        remaining = len(data) - i
        if 0 < remaining < chunk_samples:
            last_chunk = data[-chunk_samples:]
            chunks.append(last_chunk)
        
        return chunks
    
    def _bytes_to_wav(self, audio_bytes: bytes, sr: int = 16000) -> bytes:
        """
        Convert raw audio bytes to WAV format
        """
        # Try to detect format
        if audio_bytes[:4] == b'RIFF':
            # Already WAV
            return audio_bytes
        
        # Assume raw PCM (16-bit mono)
        # Create WAV header
        num_channels = 1
        sampwidth = 2  # 16-bit
        nframes = len(audio_bytes) // sampwidth
        
        # Create in-memory WAV file
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(num_channels)
            wav_file.setsampwidth(sampwidth)
            wav_file.setframerate(sr)
            wav_file.writeframes(audio_bytes)
        
        buffer.seek(0)
        return buffer.read()
    
    def _webm_to_wav(self, webm_bytes: bytes, sr: int = 16000) -> Optional[bytes]:
        """
        WebM/Opus conversion - no longer supported without ffmpeg
        Return None to let API handle the raw bytes
        """
        logger.warning("WebM conversion skipped - sending raw bytes to API")
        return None
    
    def _decode_audio_bytes(self, audio_bytes: bytes, mime_type: str = None) -> Optional[tuple]:
        """
        Decode audio bytes to (data, sample_rate) tuple
        Returns None if cannot decode
        """
        buffer = io.BytesIO(audio_bytes)
        
        # Try soundfile first
        try:
            data, sr = sf.read(buffer)
            return data, sr
        except Exception as e:
            logger.debug(f"Soundfile read failed: {e}")
        
        # Try wave module for WAV files
        buffer.seek(0)
        try:
            if audio_bytes[:4] == b'RIFF' or (mime_type and 'wav' in mime_type):
                with wave.open(buffer, 'rb') as wav:
                    sr = wav.getframerate()
                    n_frames = wav.getnframes()
                    audio_data = wav.readframes(n_frames)
                    # Convert bytes to numpy array
                    dtype = np.int16 if wav.getsampwidth() == 2 else np.int8
                    data = np.frombuffer(audio_data, dtype=dtype).astype(np.float32) / 32768.0
                    return data, sr
        except Exception as e:
            logger.debug(f"Wave read failed: {e}")
        
        # If WebM/Opus and pydub available, try conversion
        buffer.seek(0)
        if mime_type and ('webm' in mime_type or 'opus' in mime_type or 'ogg' in mime_type):
            wav_bytes = self._webm_to_wav(audio_bytes)
            if wav_bytes:
                return self._decode_audio_bytes(wav_bytes, 'audio/wav')
        
        return None
    
    def transcribe_file(self, 
                       audio_path: str, 
                       stream: bool = False) -> Dict[str, Any]:
        """
        Transcribe an audio file
        """
        try:
            logger.debug(f"Transcribing file: {audio_path}")
            
            # Load audio using soundfile
            data, sr = sf.read(audio_path)
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            if stream:
                return self._transcribe_streaming(data, sr)
            else:
                return self._transcribe_single(data, sr)
                
        except Exception as e:
            logger.error(f"Error transcribing file {audio_path}: {e}")
            return {"text": "", "error": str(e), "success": False}
    
    def _transcribe_single(self, data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Transcribe entire audio as a single request
        """
        try:
            # Create in-memory WAV file
            buf = io.BytesIO()
            sf.write(buf, data, sr, format='WAV')
            buf.seek(0)
            
            # Send to ASR API
            response = requests.post(
                self.api_url,
                files={"file": ("audio.wav", buf, "audio/wav")},
                data={"language": self.language},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "text": result.get("text", ""),
                    "language": result.get("language", self.language),
                    "success": True
                }
            else:
                logger.error(f"ASR API error: {response.status_code}")
                return {
                    "text": "",
                    "error": f"API error: {response.status_code}",
                    "success": False
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"text": "", "error": str(e), "success": False}
    
    def _transcribe_streaming(self, data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Transcribe audio by streaming chunks sequentially
        """
        try:
            chunks = self._split_audio(data, sr)
            logger.debug(f"Split audio into {len(chunks)} chunks")
            
            all_texts = []
            
            for i, chunk in enumerate(chunks):
                # Create chunk buffer
                buf = io.BytesIO()
                sf.write(buf, chunk, sr, format='WAV')
                buf.seek(0)
                
                # Send chunk
                response = requests.post(
                    self.api_url,
                    files={"file": (f"chunk_{i}.wav", buf, "audio/wav")},
                    data={"language": self.language},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    chunk_text = result.get("text", "")
                    if chunk_text:
                        all_texts.append(chunk_text)
                        logger.debug(f"Chunk {i}: {chunk_text[:50]}...")
                
                # Simulate continuous speech
                if i < len(chunks) - 1:
                    time.sleep(self.step)
            
            final_text = " ".join(all_texts)
            return {
                "text": final_text,
                "chunks": len(chunks),
                "language": self.language,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return {"text": "", "error": str(e), "success": False}
    
    def transcribe_bytes(self, 
                        audio_bytes: bytes, 
                        sample_rate: int = 16000,
                        stream: bool = False,
                        mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from bytes - SIMPLIFIED VERSION
        """
        try:
            logger.debug(f"Transcribing bytes, length: {len(audio_bytes)}, mime: {mime_type}")
            
            # Try to decode the audio
            decoded = self._decode_audio_bytes(audio_bytes, mime_type)
            
            if decoded is None:
                # If cannot decode, try to send raw bytes directly to API
                logger.warning(f"Cannot decode audio, sending raw bytes to API")
                
                # Determine file extension
                ext = "wav"
                if mime_type:
                    if "webm" in mime_type:
                        ext = "webm"
                    elif "ogg" in mime_type:
                        ext = "ogg"
                    elif "mp3" in mime_type:
                        ext = "mp3"
                
                response = requests.post(
                    self.api_url,
                    files={"file": (f"audio.{ext}", audio_bytes, mime_type or "audio/wav")},
                    data={"language": self.language},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("text", ""),
                        "language": result.get("language", self.language),
                        "success": True
                    }
                else:
                    return {
                        "text": "",
                        "error": f"API rejected audio with status {response.status_code}",
                        "success": False
                    }
            
            data, sr = decoded
            
            # Use provided sample rate or detected one
            sr = sample_rate if sample_rate else sr
            
            # Convert to mono if stereo
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            if stream:
                return self._transcribe_streaming(data, sr)
            else:
                return self._transcribe_single(data, sr)
                
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
        """
        try:
            # Decode base64
            audio_bytes = base64.b64decode(audio_base64)
            return self.transcribe_bytes(audio_bytes, sample_rate, stream, mime_type)
        except Exception as e:
            logger.error(f"Error decoding base64: {e}")
            return {"text": "", "error": str(e), "success": False}


# Singleton instance
_asr_client = None

def get_asr_client() -> ASRClient:
    """Get or create ASR client instance"""
    global _asr_client
    if _asr_client is None:
        _asr_client = ASRClient()
    return _asr_client