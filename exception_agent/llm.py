"""LLM clients. Each returns the raw text of one completion."""
import base64
import json
import re
import urllib.request
from pathlib import Path
from typing import Protocol


class LLMError(RuntimeError):
    """The endpoint itself failed (not a bad answer)."""


class LLMClient(Protocol):
    def complete(self, messages: list[dict], schema: dict, building_id: str,
                 timeout_s: float, images: list[tuple[str, str]] = ()) -> str:
        """images: (label, file path) pairs for the last user message; text-only clients ignore them."""


class StubLLM:
    """Replays scripted outputs per building; gives up when the script runs out.

    script: {building_id: [action-dict or raw string, ...]}
    """

    def __init__(self, script: dict):
        self.script = script
        self.cursor: dict[str, int] = {}

    def complete(self, messages, schema, building_id, timeout_s, images=()):
        steps = self.script.get(building_id, [])
        i = self.cursor.get(building_id, 0)
        self.cursor[building_id] = i + 1
        if i >= len(steps):
            return json.dumps({"action": "give_up", "reason": "stub script exhausted", "target": None,
                               "geometry": None, "diagnosis": "stub LLM has no further scripted step"})
        step = steps[i]
        return step if isinstance(step, str) else json.dumps(step)


class OpenAICompatLLM:
    """Chat-completions client for a local OpenAI-compatible server.

    json_mode: "schema" sends response_format json_schema (vLLM, llama.cpp server,
    Ollama, LM Studio); "object" sends json_object; "none" relies on the prompt.
    Output is validated by the harness in every mode.
    vision: attach images (roof image, render of the last build) as image_url parts.
    extra_body: merged into the request, e.g. server-specific switches for thinking mode.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "none",
                 json_mode: str = "schema", temperature: float = 0.2, max_tokens: int = 2048,
                 vision: bool = True, extra_body: dict | None = None):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.json_mode = model, api_key, json_mode
        self.temperature, self.max_tokens = temperature, max_tokens
        self.vision, self.extra_body = vision, extra_body or {}

    def build_body(self, messages, schema, images=()) -> dict:
        if self.vision and images:
            parts = [{"type": "text", "text": messages[-1]["content"]}]
            for label, path in images:
                mime = "image/jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
                data = base64.b64encode(Path(path).read_bytes()).decode()
                parts += [{"type": "text", "text": label},
                          {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]
            messages = messages[:-1] + [{**messages[-1], "content": parts}]
        body = {"model": self.model, "messages": messages,
                "temperature": self.temperature, "max_tokens": self.max_tokens, **self.extra_body}
        if self.json_mode == "schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "agent_action", "strict": True, "schema": schema}}
        elif self.json_mode == "object":
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, messages, schema, building_id, timeout_s, images=()):
        try:
            body = self.build_body(messages, schema, images)
        except OSError as e:
            raise LLMError(f"cannot read image: {e}") from e
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=max(timeout_s, 1.0)) as resp:
                data = json.load(resp)
            text = data["choices"][0]["message"]["content"] or ""
            # Thinking models may put their reasoning inline before the answer.
            return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        except Exception as e:  # network, HTTP, malformed response
            raise LLMError(f"{type(e).__name__}: {e}") from e
