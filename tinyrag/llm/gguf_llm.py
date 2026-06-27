import json
from typing import Optional
from urllib import error, request

try:
    from tinyrag.llm.base_llm import BaseLLM
except ModuleNotFoundError:
    from base_llm import BaseLLM


class GGUFLLM(BaseLLM):
    def __init__(
        self,
        model_id_key: str,
        device: str = "cpu",
        is_api: bool = False,
        base_url: str = "http://127.0.0.1:8080/v1",
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        super().__init__(model_id_key, device, is_api)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        content: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": self.model_id_key,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GGUF HTTP request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"GGUF HTTP request failed: {exc.reason}") from exc

        payload = json.loads(raw)
        message = payload["choices"][0]["message"]["content"]
        return message.strip() if isinstance(message, str) else str(message)
