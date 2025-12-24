import os
import csv
import threading
from datetime import datetime
from typing import Optional

class CSVLogger:
    """
    Thread-safe CSV logger for tracking RAG performance metrics
    """
    def __init__(self, log_file: str = None):
        if log_file is None:
            # Use /tmp for serverless, otherwise logs/
            if os.environ.get("VERCEL") or os.environ.get("SERVERLESS"):
                log_dir = "/tmp/logs"
            else:
                log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "performance_metrics.csv")
        
        self.log_file = log_file
        self.lock = threading.Lock()
        
        # Create file with headers if not exists
        if not os.path.exists(self.log_file):
            self._write_headers()
    
    def _write_headers(self):
        """Write CSV headers"""
        headers = [
            'timestamp',
            'task_id',
            'question',
            'answer',
            'context_num',
            'time_receive_to_first_token_ms',
            'time_embedding_ms',
            'time_retrieval_ms',
            'time_llm_stream_ms',
            'time_total_ms',
            'quick_audio_generated',
            'full_audio_parts',
            'status'
        ]
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_task(
        self,
        task_id: str,
        question: str,
        answer: str = "",
        context_num: int = 0,
        time_receive_to_first_token_ms: Optional[float] = None,
        time_embedding_ms: Optional[float] = None,
        time_retrieval_ms: Optional[float] = None,
        time_llm_stream_ms: Optional[float] = None,
        time_total_ms: Optional[float] = None,
        quick_audio_generated: bool = False,
        full_audio_parts: int = 0,
        status: str = "completed"
    ):
        """
        Log a task's performance metrics to CSV
        
        Args:
            task_id: Unique task identifier
            question: User's question
            answer: Generated answer
            context_num: Number of context chunks provided to LLM
            time_receive_to_first_token_ms: Time from receiving WAV to first token
            time_embedding_ms: Time to generate embedding
            time_retrieval_ms: Time for retrieval from Qdrant
            time_llm_stream_ms: Time for LLM streaming
            time_total_ms: Total time from receive to completion
            quick_audio_generated: Whether quick audio was generated
            full_audio_parts: Number of full audio parts
            status: Task status (completed/failed/etc)
        """
        
        row = [
            datetime.now().isoformat(),
            task_id,
            question[:200] if question else "",  # Truncate long questions
            answer[:500] if answer else "",  # Truncate long answers
            context_num,
            round(time_receive_to_first_token_ms, 2) if time_receive_to_first_token_ms else "",
            round(time_embedding_ms, 2) if time_embedding_ms else "",
            round(time_retrieval_ms, 2) if time_retrieval_ms else "",
            round(time_llm_stream_ms, 2) if time_llm_stream_ms else "",
            round(time_total_ms, 2) if time_total_ms else "",
            quick_audio_generated,
            full_audio_parts,
            status
        ]
        
        with self.lock:
            try:
                with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
            except Exception as e:
                # Don't fail the main process if logging fails
                print(f"CSV logging failed: {e}")


# Global singleton instance
_csv_logger = None

def get_csv_logger() -> CSVLogger:
    """Get the global CSV logger instance"""
    global _csv_logger
    if _csv_logger is None:
        _csv_logger = CSVLogger()
    return _csv_logger
