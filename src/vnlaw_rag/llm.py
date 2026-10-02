"""Client LLM tối giản: Gemini, Groq, hoặc server local tương thích OpenAI (llama.cpp / Ollama).

Gọi thẳng REST bằng requests thay vì cài SDK của từng hãng, để repo nhẹ và
đổi provider chỉ bằng biến môi trường. Provider đầu lỗi (429, 5xx, timeout)
thì tự chuyển sang provider sau.
"""

import logging
import os
import time
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

DEFAULT_MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "groq": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    "local": os.getenv("LOCAL_MODEL", "local"),
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float


def _gemini(system: str, user: str, model: str, temperature: float, json_mode: bool, timeout: int) -> str:
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    cfg = {"temperature": temperature}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": cfg,
    }
    r = requests.post(url, json=body, headers={"x-goog-api-key": key}, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"gemini {r.status_code}: {r.text[:200]}")
    data = r.json()
    try:
        return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError) as e:
        # bị chặn bởi safety filter thì không có 'content'
        raise LLMError(f"gemini: response lạ {str(data)[:200]}") from e


def _openai_compat(base_url: str, key: str | None, system: str, user: str, model: str,
                   temperature: float, json_mode: bool, timeout: int) -> str:
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = requests.post(f"{base_url}/chat/completions", json=body, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise LLMError(f"{base_url} {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"] or ""


def _call(provider: str, system: str, user: str, temperature: float, json_mode: bool, timeout: int) -> str:
    model = DEFAULT_MODELS[provider]
    if provider == "gemini":
        return _gemini(system, user, model, temperature, json_mode, timeout)
    if provider == "groq":
        return _openai_compat("https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"],
                              system, user, model, temperature, json_mode, timeout)
    if provider == "local":
        base = os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8080/v1")
        return _openai_compat(base, None, system, user, model, temperature, json_mode, timeout)
    raise ValueError(f"provider không hỗ trợ: {provider}")


class LLMClient:
    def __init__(self, providers: list[str] | None = None, retries: int = 2, timeout: int = 60):
        env = os.getenv("LLM_PROVIDERS", "gemini,groq")
        self.providers = providers or [p.strip() for p in env.split(",") if p.strip()]
        self.retries = retries
        self.timeout = timeout

    def complete(self, system: str, user: str, temperature: float = 0.2, json_mode: bool = False) -> LLMResponse:
        errors = []
        for provider in self.providers:
            for attempt in range(self.retries + 1):
                t0 = time.perf_counter()
                try:
                    text = _call(provider, system, user, temperature, json_mode, self.timeout)
                    return LLMResponse(text, provider, DEFAULT_MODELS[provider], time.perf_counter() - t0)
                except KeyError as e:
                    errors.append(f"{provider}: thiếu biến môi trường {e}")
                    break  # thiếu key thì retry cũng vô ích
                except (LLMError, requests.RequestException) as e:
                    errors.append(f"{provider}#{attempt}: {e}")
                    log.warning("LLM lỗi %s, thử lại", e)
                    time.sleep(1.5 * (attempt + 1))
        raise LLMError("Tất cả provider đều lỗi:\n" + "\n".join(errors))
