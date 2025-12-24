import requests

API_URL = "https://hoangnam5904-embedding.hf.space/embed"

def get_remote_embedding(text: str, timeout: int = 30):
    payload = {"text": text}
    resp = requests.post(API_URL, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Embedding API error: {resp.status_code} {resp.text}")
    data = resp.json()
    return data["embedding"]
