#!/bin/bash
MODEL_URL="https://huggingface.co/bartowski/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q8_0.gguf"

echo "🔽 Đang tải gemma-3-1b-it-Q8_0.gguf ..."
wget -c "$MODEL_URL" -O "gemma-3-1b-it-Q8_0.gguf"

if [ $? -eq 0 ]; then
    echo "✅ Tải thành công: gemma-3-1b-it-Q8_0.gguf"
else
    echo "❌ Tải thất bại. Vui lòng kiểm tra kết nối hoặc đường dẫn."
fi

streamlit run chatbot.py