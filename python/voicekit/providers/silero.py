"""Silero TTS (Russian). Runs the packaged TorchScript model on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..base import TtsEngine, Voice
from ..common import VoiceError

# Speakers Silero ships for convenience that are not real voices.
SKIPPED_SPEAKERS = {"random"}


class SileroTts(TtsEngine):
    def __init__(self, folder: Path, sample_rate: int):
        import torch
        from torch.package import PackageImporter

        files = sorted(folder.glob("*.pt"))
        if not files:
            raise VoiceError("Silero model file is missing. Reinstall the model.")
        torch.set_num_threads(4)
        importer = PackageImporter(str(files[0]))
        self._model = importer.load_pickle("tts_models", "model")
        self._model.to(torch.device("cpu"))
        self._sample_rate = sample_rate
        speakers = [name for name in getattr(self._model, "speakers", []) if name not in SKIPPED_SPEAKERS]
        if not speakers:
            raise VoiceError("The Silero model does not list any speakers.")
        self._voices = [Voice(name, name) for name in speakers]

    def voices(self) -> list[Voice]:
        return list(self._voices)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        # Silero has no speed control; the synthesis pipeline varies tempo later.
        audio = self._model.apply_tts(
            text=text,
            speaker=voice,
            sample_rate=self._sample_rate,
            put_accent=True,
            put_yo=True,
        )
        samples = audio.detach().cpu().numpy().astype(np.float32)
        if speed and abs(speed - 1.0) > 0.01:
            import librosa

            samples = librosa.effects.time_stretch(samples, rate=float(speed))
        return samples, self._sample_rate
