"""Piper TTS (piper-tts package, bundles espeak-ng). One catalog entry may hold
several single-speaker voices or one multi-speaker model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import TtsEngine, Voice
from ..common import VoiceError

# Multi-speaker models (LibriTTS-R has 904) expose an evenly spread subset so a
# synthesis run covers many different voices without hundreds of entries.
MAX_SPEAKERS_PER_MODEL = 40


class PiperTts(TtsEngine):
    def __init__(self, folder: Path, voices: tuple[dict, ...]):
        from piper import PiperVoice, SynthesisConfig

        self._config_cls = SynthesisConfig
        self._models: dict[str, object] = {}
        self._voices: list[Voice] = []
        for entry in voices:
            onnx = folder / f"{entry['id']}.onnx"
            if not onnx.is_file():
                raise VoiceError(f"Piper voice file {onnx.name} is missing. Reinstall the model.")
            model = PiperVoice.load(onnx)
            self._models[entry["id"]] = model
            speakers = int(entry.get("speakers") or model.config.num_speakers or 1)
            if speakers <= 1:
                self._voices.append(Voice(entry["id"], entry["name"]))
                continue
            step = max(1, speakers // MAX_SPEAKERS_PER_MODEL)
            for speaker in range(0, speakers, step)[:MAX_SPEAKERS_PER_MODEL]:
                self._voices.append(Voice(f"{entry['id']}#{speaker}", f"{entry['name']} #{speaker}"))

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        model_id, _, speaker = voice.partition("#")
        model = self._models.get(model_id)
        if model is None:
            raise VoiceError(f"Unknown Piper voice '{voice}'.")
        config = self._config_cls(
            speaker_id=int(speaker) if speaker else None,
            length_scale=1.0 / max(0.5, float(speed)),
        )
        chunks = list(model.synthesize(text, syn_config=config))
        if not chunks:
            raise VoiceError(f"Piper produced no audio for '{text}'.")
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        return audio, int(chunks[0].sample_rate)
