# utils/logger.py
import logging
import os

os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("rag_bot")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# File
fh = logging.FileHandler("logs/gradio_rag.log", encoding="utf-8")
fh.setFormatter(formatter)
logger.addHandler(fh)

# Timing logger riêng
timing_logger = logging.getLogger("timing_context")
timing_logger.setLevel(logging.INFO)
th = logging.FileHandler("logs/rag_timing_context.log", encoding="utf-8")
th.setFormatter(formatter)
timing_logger.addHandler(th)