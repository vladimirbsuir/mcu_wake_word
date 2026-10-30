"""OpenAI-compatible HTTP endpoints for all three categories.

Works with OpenAI itself and with anything that mirrors its API: OpenRouter,
Groq, Together, Ollama, LM Studio and llama.cpp's own server (which the local
llama.cpp provider reuses). Only the standard library is used.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid

import numpy as np
import soundfile as sf

from ..base import LlmEngine, SttEngine, TtsEngine, Voice
from ..common import USER_AGENT, VoiceError

DEFAULT_TTS_VOICES = ("alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer")


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str | None, name: str, timeout: float = 180):
        if not base_url:
            raise VoiceError(f"Endpoint '{name}' has no base URL.")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name
        self.timeout = timeout

    def _request(self, path: str, body: bytes, content_type: str) -> bytes:
        headers = {"Content-Type": content_type, "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(f"{self.base_url}{path}", data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise VoiceError(f"{self.name}: HTTP {exc.code} from {path}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VoiceError(f"{self.name}: cannot reach {self.base_url} ({exc.reason}).") from exc

    def post_json(self, path: str, payload: dict) -> bytes:
        return self._request(path, json.dumps(payload).encode("utf-8"), "application/json")

    def post_multipart(self, path: str, fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> bytes:
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for key, value in fields.items():
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
        for key, (filename, data, mime) in files.items():
            body.write(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {mime}\r\n\r\n".encode()
            )
            body.write(data)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        return self._request(path, body.getvalue(), f"multipart/form-data; boundary={boundary}")


def chat_completion(
    client: OpenAICompatClient,
    model: str,
    messages: list[dict],
    schema: dict | None,
    temperature: float,
    extra: dict | None = None,
    max_tokens: int | None = None,
) -> str:
    payload: dict = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "schema": schema, "strict": True},
        }
    payload.update(extra or {})
    try:
        raw = client.post_json("/chat/completions", payload)
    except VoiceError as exc:
        # Some servers do not support json_schema; retry with plain JSON mode.
        unsupported = "response_format" in str(exc) or "json_schema" in str(exc)
        if schema is None or not unsupported:
            raise
        payload["response_format"] = {"type": "json_object"}
        raw = client.post_json("/chat/completions", payload)
    try:
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise VoiceError(f"{client.name}: unexpected chat response: {raw[:300]!r}") from exc


class OpenAICompatLlm(LlmEngine):
    def __init__(self, client: OpenAICompatClient, model: str):
        self._client = client
        self._model = model

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> str:
        return chat_completion(self._client, self._model, messages, schema, temperature, None, max_tokens)


class OpenAICompatTts(TtsEngine):
    def __init__(self, client: OpenAICompatClient, model: str, voices: list[str]):
        self._client = client
        self._model = model
        self._voices = [Voice(name, name) for name in (voices or DEFAULT_TTS_VOICES)]

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        raw = self._client.post_json(
            "/audio/speech",
            {"model": self._model, "input": text, "voice": voice, "speed": speed, "response_format": "wav"},
        )
        try:
            audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception as exc:
            raise VoiceError(f"{self._client.name}: TTS did not return WAV audio.") from exc
        return audio.mean(axis=1), int(sample_rate)


class OpenAICompatStt(SttEngine):
    def __init__(self, client: OpenAICompatClient, model: str, language: str):
        self._client = client
        self._model = model
        self._language = language

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
        raw = self._client.post_multipart(
            "/audio/transcriptions",
            {"model": self._model, "language": self._language, "response_format": "json"},
            {"file": ("take.wav", buffer.getvalue(), "audio/wav")},
        )
        try:
            return str(json.loads(raw).get("text", "")).strip()
        except ValueError:
            return raw.decode("utf-8", "replace").strip()
