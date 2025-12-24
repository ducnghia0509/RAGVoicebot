# Vercel serverless function wrapper
import sys
import os
import logging

# Disable file logging for serverless environment
os.environ["VERCEL"] = "1"
os.environ["SERVERLESS"] = "1"

# Use /tmp for any file operations (only writable dir in Vercel)
os.environ["TMP_AUDIO_DIR"] = "/tmp/audio"
os.environ["LOG_DIR"] = "/tmp/logs"

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Suppress file-based logging before importing app
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

from app import app

# Export for Vercel
handler = app
