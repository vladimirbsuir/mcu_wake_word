"""sherpa-onnx engines: transducer/Whisper STT and Supertonic TTS (CPU, no PyTorch)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..base import SttEngine, TtsEngine, Voice
from ..common import VoiceError

THREADS = max(1, min(8, (os.cpu_count() or 4) // 2))


def _pick(folder: Path, pattern: str) -> Path:
    """Prefer the int8 variant of an ONNX file when the archive ships both."""
    matches = sorted(folder.rglob(pattern))
    if not matches:
        raise VoiceError(f"Model file matching '{pattern}' is missing in {folder}. Reinstall the model.")
    int8 = [path for path in matches if ".int8." in path.name]
    return int8[0] if int8 else matches[0]


def _decode(recognizer, audio: np.ndarray, sample_rate: int) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, np.ascontiguousarray(audio, dtype=np.float32))
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


class TransducerStt(SttEngine):
    def __init__(self, folder: Path, model_type: str):
        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(_pick(folder, "encoder*.onnx")),
            decoder=str(_pick(folder, "decoder*.onnx")),
            joiner=str(_pick(folder, "joiner*.onnx")),
            tokens=str(_pick(folder, "tokens.txt")),
            model_type=model_type,
            num_threads=THREADS,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return _decode(self._recognizer, audio, sample_rate)


class WhisperStt(SttEngine):
    def __init__(self, folder: Path, language: str):
        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(_pick(folder, "*encoder*.onnx")),
            decoder=str(_pick(folder, "*decoder*.onnx")),
            tokens=str(_pick(folder, "*tokens.txt")),
            language=language,
            task="transcribe",
            num_threads=THREADS,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return _decode(self._recognizer, audio, sample_rate)


class SupertonicTts(TtsEngine):
    def __init__(self, folder: Path, language: str, voices: tuple[dict, ...]):
        import sherpa_onnx

        self._sherpa = sherpa_onnx
        self._language = language
        self._voices = [Voice(item["id"], item["name"]) for item in voices]
        model = sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(_pick(folder, "duration_predictor*.onnx")),
            text_encoder=str(_pick(folder, "text_encoder*.onnx")),
            vector_estimator=str(_pick(folder, "vector_estimator*.onnx")),
            vocoder=str(_pick(folder, "vocoder*.onnx")),
            tts_json=str(_pick(folder, "tts.json")),
            unicode_indexer=str(_pick(folder, "unicode_indexer.bin")),
            voice_style=str(_pick(folder, "voice.bin")),
        )
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(supertonic=model, num_threads=THREADS)
        )
        self._tts = sherpa_onnx.OfflineTts(config)

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        config = self._sherpa.GenerationConfig()
        config.sid = int(voice)
        config.speed = float(speed)
        config.num_steps = 8
        # ``extra`` is returned as a copy: item assignment would be silently lost.
        config.extra = {"lang": self._language}
        audio = self._tts.generate(text, config)
        samples = np.asarray(audio.samples, dtype=np.float32)
        if samples.size == 0:
            raise VoiceError(f"Supertonic produced no audio for '{text}'.")
        return samples, int(audio.sample_rate)
