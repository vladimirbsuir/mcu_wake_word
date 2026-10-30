"""Engine interfaces. Every provider, local or remote, implements one of these.

Engines are created for one language and used inside a single job, so they
may hold heavy state (loaded ONNX sessions, a running llama.cpp server) and
must release it in ``close()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Voice:
    id: str
    name: str


class Engine(ABC):
    def close(self) -> None:  # noqa: B027 - optional hook
        """Release models, servers or connections."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class SttEngine(Engine):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return the text spoken in a mono float32 clip."""


class TtsEngine(Engine):
    @abstractmethod
    def voices(self) -> list[Voice]:
        """Voices usable for the engine's language, in a stable order."""

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample rate)."""


class LlmEngine(Engine):
    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        temperature: float = 0.8,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant text. With ``schema`` the answer must be JSON.

        ``max_tokens`` bounds the answer; small models otherwise tend to loop
        on one word until the context is full.
        """
