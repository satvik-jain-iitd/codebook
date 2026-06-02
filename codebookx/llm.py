import json
import re
import requests
from typing import Optional, Tuple
from tqdm import tqdm

def check_server(base_url: str) -> tuple[bool, Optional[str]]:
    """Check if an LLM server is running and return (is_running, model_name)."""

    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=3)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            if models:
                model_name = models[0].get("id", "unknown")
                return True, model_name
            return True, None
    except Exception:
        pass
    return False, None

def strip_think_tokens(text: str) -> str:
    """Remove <think>...</think> blocks (used by reasoning models like DeepSeek-R1)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def stream_chat_completion(url: str, model: str, messages: list, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Send messages to LM Studio and stream the response, returning the full text."""
    try:
        response = requests.post(
            f"{url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=(10, 60),
        )
        response.raise_for_status()

        full_text_chunks = []
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            data = response.json()
            full_text = data["choices"][0]["message"]["content"]
            print(full_text)
            return strip_think_tokens(full_text)
        else:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                        full_text_chunks.append(delta)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
        print()
        return strip_think_tokens("".join(full_text_chunks))
    except Exception as e:
        print(f"\n[Error: {e}]")
        return f"[Error: {e}]"
