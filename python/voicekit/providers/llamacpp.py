"""Local LLM through a llama.cpp server that lives only for the current job."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from ..base import LlmEngine
from ..common import VoiceError, log
from .openai_compat import OpenAICompatClient, chat_completion

STARTUP_TIMEOUT_SECONDS = 180
CONTEXT_TOKENS = 4096


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _server_binary(runtime_dir: Path) -> Path:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    matches = sorted(runtime_dir.rglob(name))
    if not matches:
        raise VoiceError("llama-server was not found in the llama.cpp runtime. Reinstall it.")
    return matches[0]


class LlamaCppLlm(LlmEngine):
    def __init__(self, runtime_dir: Path, gguf: Path, disable_thinking: bool):
        if not gguf.is_file():
            raise VoiceError(f"Model file {gguf.name} is missing. Reinstall the model.")
        self._disable_thinking = disable_thinking
        self._process: subprocess.Popen | None = None
        binary = _server_binary(runtime_dir)
        port = _free_port()
        log(f"Starting llama.cpp server ({gguf.name}) on 127.0.0.1:{port}")
        command = [
            str(binary),
            "--model", str(gguf),
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(CONTEXT_TOKENS),
            "--n-gpu-layers", "999",  # Vulkan build offloads when a GPU exists, CPU otherwise
            "--jinja",
            "--no-webui",
        ]
        flags = 0x0800_0000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        # The server logs every request; a pipe nobody drains would block it.
        self._server_log = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            command,
            cwd=str(binary.parent),
            stdout=subprocess.DEVNULL,
            stderr=self._server_log,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        base = f"http://127.0.0.1:{port}"
        self._wait_ready(base)
        self._client = OpenAICompatClient(f"{base}/v1", None, "llama.cpp", timeout=600)

    def _wait_ready(self, base: str) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process and self._process.poll() is not None:
                self._server_log.seek(0)
                error = self._server_log.read().decode("utf-8", "replace")[-800:]
                raise VoiceError(f"llama.cpp server exited during startup: {error}")
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)
        self.close()
        raise VoiceError("llama.cpp server did not become ready in time.")

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> str:
        # Small local models repeat themselves easily; DRY sampling and a
        # presence penalty keep word lists varied (llama.cpp server options).
        extra: dict = {
            "presence_penalty": 0.6,
            "repeat_penalty": 1.1,
            "dry_multiplier": 0.8,
            "dry_base": 1.75,
            "dry_allowed_length": 2,
        }
        if self._disable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return chat_completion(self._client, "local", messages, schema, temperature, extra, max_tokens)

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        log_file = getattr(self, "_server_log", None)
        if log_file is not None:
            log_file.close()
            self._server_log = None

    def __del__(self) -> None:
        if sys is not None:  # interpreter shutdown guard
            self.close()
