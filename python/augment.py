#!/usr/bin/env python3
"""Build an embedded-friendly wake-word dataset from clean recordings.

The script intentionally uses a source-level split: every augmented variant of a
recording ends up in the same train/val/test split.  Splitting individual
variants would leak near-duplicate audio across splits and inflate validation
accuracy.

Output layout:
    <output>/train/wake_word/*.wav
    <output>/train/background/*.wav
    <output>/val/...
    <output>/test/...
    <output>/manifest.csv
    <output>/dataset_info.json

Progress is emitted as ``@@PROGRESS@@ {json}`` and the final payload as
``@@RESULT@@ {json}``.  The Tauri backend understands these markers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from voicekit.corpora import Corpus, utterance_split
from voicekit.groups import contains_phrase

try:
    import librosa
except ImportError as exc:  # pragma: no cover - gives a better message than a traceback
    raise SystemExit(
        "librosa is missing. Run: pip install -r python/requirements.txt"
    ) from exc

SAMPLE_RATE = 16_000
LABEL_WAKE = "wake_word"
LABEL_BACKGROUND = "background"
PROGRESS_PREFIX = "@@PROGRESS@@"
RESULT_PREFIX = "@@RESULT@@"

# Positive clips are peak-normalized to a modest level (typical of raw
# microphone input) and then varied over a small range. Normalizing to full
# scale would put training far above the levels the model sees live. The
# range reaches down to about -44 dBFS peak: speech from across a room is that
# quiet, and with a -12 dB floor recall fell from 96% to 14% between -25 and
# -40 dBFS in cross-validation.
POSITIVE_TARGET_PEAK = 0.20
POSITIVE_GAIN_MIN_DB = -30.0
POSITIVE_GAIN_MAX_DB = 6.0

# Minimum distance between the phrase and the window edges.
PLACEMENT_MARGIN_SECONDS = 0.02
# A take must fit into this fraction of the window, otherwise the phrase would
# be cut and the clip labelled as a full wake word anyway.
MAX_PHRASE_FILL = 0.97
STRETCH_MIN_RATE = 0.88
STRETCH_MAX_RATE = 1.12

# Share of negative clips per kind (per split). Shares are skipped when their
# source is missing, and whatever is left goes to the synthetic kinds below.
HARD_NEGATIVE_SHARE = 0.35  # user-provided recordings of other words/phrases
REAL_BACKGROUND_SHARE = 0.30  # user-provided background recordings
PARTIAL_PHRASE_SHARE = 0.10  # an incomplete wake phrase must not trigger
REVERSED_PHRASE_SHARE = 0.05  # same voice and spectrum, different word
PARTIAL_MIN_FRACTION = 0.25
PARTIAL_MAX_FRACTION = 0.60
# Windows cut from downloaded speech corpora are added on top of the balanced
# negatives: hundreds of speakers saying anything but the wake phrase.
DEFAULT_CORPUS_NEGATIVES = 6000
# A device hears its user across a room and in the middle of other speech
# ("so, <phrase>, turn on..."), while takes are recorded close to the mic in
# silence. Some variants of every speech clip (positives and speech negatives
# alike, so neither becomes a cue) get room reverb and other speech around it.
REVERB_PROB = 0.35
REVERB_RT60_SECONDS = (0.15, 0.7)
REVERB_DRR_DB = (-2.0, 12.0)  # direct-to-reverberant energy ratio
SPEECH_CONTEXT_PROB = 0.3
SPEECH_CONTEXT_GAP_SECONDS = (0.04, 0.25)
SPEECH_CONTEXT_BELOW_DB = (2.0, 12.0)

ContextFn = Callable[[int], np.ndarray]


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def emit_progress(stage: str, progress: float, message: str, **extra: object) -> None:
    payload = {
        "stage": stage,
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        **extra,
    }
    print(f"{PROGRESS_PREFIX} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def emit_result(payload: dict) -> None:
    print(f"{RESULT_PREFIX} {json.dumps(payload, ensure_ascii=False)}", flush=True)


def log(message: str) -> None:
    print(f"[augment] {message}", flush=True)


def list_wavs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".ogg"}
    )


def load_mono(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, source_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if source_sr != sr:
        mono = librosa.resample(mono, orig_sr=source_sr, target_sr=sr, res_type="kaiser_best")
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if mono.size:
        mono -= float(mono.mean())
    return mono


def trim_and_normalize(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    if audio.size == 0:
        return audio
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=32)
        if trimmed.size:
            audio = trimmed
    except Exception:
        pass
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-7:
        audio = audio / peak
    return audio.astype(np.float32)


def place_in_clip(
    audio: np.ndarray,
    clip_samples: int,
    rng: random.Random,
    sr: int = SAMPLE_RATE,
    random_offset: bool = True,
) -> np.ndarray:
    """Center-crop long clips, otherwise place the phrase at a random offset.

    The offset is uniform over all positions where the phrase fits completely.
    A sliding window on the device sees the phrase at every alignment, so the
    model must not rely on it always starting early (or ending late).
    """
    return place_with_offset(audio, clip_samples, rng, sr, random_offset)[0]


def place_with_offset(
    audio: np.ndarray,
    clip_samples: int,
    rng: random.Random,
    sr: int = SAMPLE_RATE,
    random_offset: bool = True,
) -> tuple[np.ndarray, int]:
    """``place_in_clip`` that also returns where the phrase starts in the window."""
    out = np.zeros(clip_samples, dtype=np.float32)
    pad = 0
    if audio.size == 0:
        return out, 0
    if audio.size >= clip_samples:
        start = max(0, (audio.size - clip_samples) // 2)
        out[:] = audio[start : start + clip_samples]
    else:
        slack = clip_samples - audio.size
        if random_offset:
            margin = min(int(PLACEMENT_MARGIN_SECONDS * sr), slack // 2)
            pad = rng.randint(margin, slack - margin)
        else:
            pad = slack // 2
        out[pad : pad + audio.size] = audio
    # Tiny edge fades suppress clicks without altering the phrase.
    fade = min(int(0.008 * sr), out.size // 2)
    if fade > 1:
        out[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return out, pad


def room_impulse_response(rng: random.Random, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic room: a direct path plus an exponentially decaying diffuse tail."""
    rt60 = rng.uniform(*REVERB_RT60_SECONDS)
    length = int(min(rt60, 0.8) * sr)
    generator = np.random.default_rng(rng.randrange(2**31))
    t = np.arange(length, dtype=np.float32) / sr
    tail = generator.normal(0.0, 1.0, length).astype(np.float32) * np.exp(-6.91 * t / rt60)
    pre_delay = int(rng.uniform(0.002, 0.012) * sr)
    tail[:pre_delay] = 0.0
    drr = rng.uniform(*REVERB_DRR_DB)
    tail *= math.sqrt(10.0 ** (-drr / 10.0) / max(float(np.sum(tail**2)), 1e-12))
    tail[0] += 1.0
    return tail


def add_reverb(audio: np.ndarray, max_samples: int, rng: random.Random) -> np.ndarray:
    """Convolve with a synthetic room; the tail is kept as long as the window allows."""
    wet = fftconvolve(audio, room_impulse_response(rng))[: max(audio.size, max_samples)]
    peak = float(np.max(np.abs(wet))) if wet.size else 0.0
    source_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-9:
        wet *= source_peak / peak
    return wet.astype(np.float32)


def add_speech_context(
    window: np.ndarray, start: int, length: int, speech_rms: float, context: ContextFn, rng: random.Random
) -> None:
    """Other speech before and after the phrase, a short pause apart, quieter than it."""
    bed = np.asarray(context(window.size), dtype=np.float32)[: window.size]
    if bed.size < window.size:
        return
    gap = int(rng.uniform(*SPEECH_CONTEXT_GAP_SECONDS) * SAMPLE_RATE)
    lo, hi = max(0, start - gap), min(window.size, start + length + gap)
    bed[lo:hi] = 0.0
    fade = min(int(0.02 * SAMPLE_RATE), lo, window.size - hi)
    if fade > 1:
        bed[lo - fade : lo] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        bed[hi : hi + fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
    active = bed[np.abs(bed) > 1e-6]
    if active.size < SAMPLE_RATE // 10:
        return
    level = float(np.sqrt(np.mean(active**2)))
    target = speech_rms / (10.0 ** (rng.uniform(*SPEECH_CONTEXT_BELOW_DB) / 20.0))
    window += bed * (target / max(level, 1e-9))


def peak_normalize(audio: np.ndarray, target_peak: float = 0.89) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-8:
        return audio.astype(np.float32)
    return np.clip(audio * (target_peak / peak), -1.0, 1.0).astype(np.float32)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / np.sqrt(freqs[1:])
    colored = np.fft.irfft(spectrum * scaling, n=n)
    return colored.astype(np.float32)


def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(size=n).astype(np.float64)
    # A leaky integrator avoids the DC drift of a plain cumulative sum.
    brown = np.empty(n, dtype=np.float64)
    value = 0.0
    for index in range(n):
        value = 0.995 * value + white[index]
        brown[index] = value
    return brown.astype(np.float32)


def synthetic_noise(n: int, rng: random.Random) -> np.ndarray:
    """Create noisy, non-speech backgrounds so the model sees negative audio.

    This is deliberately not a replacement for real background recordings. It
    covers room tone, hiss, hum and clicks; the UI lets users provide a folder
    with real background/noise WAVs for a much stronger negative class.
    """
    generator = np.random.default_rng(rng.randrange(2**31))
    choice = rng.choice(["white", "pink", "brown", "hum", "clicks"])
    if choice == "white":
        noise = generator.normal(0, 1, n).astype(np.float32)
    elif choice == "pink":
        noise = pink_noise(n, generator)
    elif choice == "brown":
        noise = brown_noise(n, generator)
    elif choice == "hum":
        t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
        base = rng.choice([50.0, 60.0])
        noise = (
            0.5 * generator.normal(0, 1, n).astype(np.float32)
            + 0.7 * np.sin(2 * np.pi * base * t + rng.random() * 6.28).astype(np.float32)
            + 0.25 * np.sin(2 * np.pi * base * 2 * t).astype(np.float32)
            + 0.12 * np.sin(2 * np.pi * base * 3 * t).astype(np.float32)
        )
    else:
        noise = 0.04 * generator.normal(0, 1, n).astype(np.float32)
        clicks = max(2, int(n / SAMPLE_RATE * rng.uniform(1.5, 7.0)))
        for _ in range(clicks):
            position = rng.randrange(max(1, n - 800))
            decay = np.exp(-np.linspace(0, 12, 800)).astype(np.float32)
            noise[position : position + 800] += decay * rng.uniform(0.15, 0.8)
    if not np.any(np.isfinite(noise)):
        return np.zeros(n, dtype=np.float32)
    return noise.astype(np.float32)


def fit_noise(noise: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    if noise.size == 0:
        return synthetic_noise(n, rng)
    if noise.size >= n:
        start = rng.randrange(0, noise.size - n + 1)
        return noise[start : start + n].copy()
    repeats = int(math.ceil(n / noise.size))
    tiled = np.tile(noise, repeats)
    start = rng.randrange(0, max(1, tiled.size - n + 1))
    return tiled[start : start + n].copy()


class NoiseBank:
    def __init__(self, folder: Path | None, rng: random.Random):
        self._rng = rng
        self._clips: list[np.ndarray] = []
        self.source_names: list[str] = []
        if folder and folder.exists():
            files = list_wavs(folder)
            log(f"Loading {len(files)} background files from {folder}")
            for path in files[:500]:
                try:
                    clip = trim_and_normalize(load_mono(path))
                    if clip.size:
                        self._clips.append(clip)
                        self.source_names.append(path.name)
                except Exception as exc:
                    print(f"[augment] skipped background {path.name}: {exc}", file=sys.stderr, flush=True)

    @property
    def has_real_noise(self) -> bool:
        return bool(self._clips)

    def get(self, n: int) -> np.ndarray:
        if not self._clips:
            return synthetic_noise(n, self._rng)
        return fit_noise(self._rng.choice(self._clips), n, self._rng)


def augment_speech(
    clean: np.ndarray,
    clip_samples: int,
    noise_bank: NoiseBank,
    rng: random.Random,
    variant_index: int,
    vary_voice: bool = True,
    context: ContextFn | None = None,
) -> tuple[np.ndarray, dict]:
    """Generate one speech variation and return audio plus provenance.

    Used for wake-word positives and for speech negatives (other phrases,
    partial or reversed wake phrases), so both classes share the exact same
    level, noise and placement statistics and only the words differ.
    """
    params = {
        "pitch_semitones": 0.0,
        "stretch_rate": 1.0,
        "snr_db": None,
        "gain_db": 0.0,
        "noise": "none",
        "reverb": False,
        "speech_context": False,
    }
    audio = clean.copy()
    noise_snr: float | None = None

    # Keep variant 0 close to the original so the clean condition is represented.
    # Corpus speech already comes from many voices; it only needs the noise.
    if variant_index > 0:
        if vary_voice and rng.random() < 0.8:
            semitones = rng.uniform(-2.5, 2.5)
            if abs(semitones) > 0.15:
                audio = librosa.effects.pitch_shift(
                    y=audio, sr=SAMPLE_RATE, n_steps=semitones, res_type="kaiser_best"
                )
                params["pitch_semitones"] = round(float(semitones), 3)

        if vary_voice and rng.random() < 0.75:
            # Never slow the phrase down so much that it no longer fits the
            # window (time_stretch output length is len / rate).
            min_rate = max(STRETCH_MIN_RATE, audio.size / (clip_samples * MAX_PHRASE_FILL))
            if min_rate < STRETCH_MAX_RATE:
                rate = rng.uniform(min_rate, STRETCH_MAX_RATE)
                audio = librosa.effects.time_stretch(y=audio, rate=rate)
                params["stretch_rate"] = round(float(rate), 3)

        if rng.random() < REVERB_PROB:
            audio = add_reverb(audio, int(clip_samples * MAX_PHRASE_FILL), rng)
            params["reverb"] = True

        if rng.random() < 0.9:
            noise_snr = rng.uniform(5.0, 25.0)
            params["snr_db"] = round(float(noise_snr), 2)
            params["noise"] = "real" if noise_bank.has_real_noise else "synthetic"

    # Limit peaks before placing, but do not flatten the level: a fixed gain
    # applied to every clip makes live microphone input (which is not
    # normalized) land far outside the training distribution.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    speech_rms = rms(audio)

    length = audio.size
    audio, start = place_with_offset(audio, clip_samples, rng)
    if context is not None and variant_index > 0 and rng.random() < SPEECH_CONTEXT_PROB:
        add_speech_context(audio, start, length, speech_rms, context, rng)
        params["speech_context"] = True

    # Mix noise into the complete window, not just the trimmed phrase. Otherwise
    # noisy positives get silent edges while negative clips are noisy edge to
    # edge, which teaches the model a bogus envelope cue.
    if noise_snr is not None:
        noise = noise_bank.get(clip_samples)
        noise_rms = rms(noise)
        if noise_rms > 1e-8:
            target_rms = speech_rms / (10.0 ** (noise_snr / 20.0))
            audio = audio + noise * (target_rms / noise_rms)

    audio = peak_normalize(audio, POSITIVE_TARGET_PEAK)
    gain_db = rng.uniform(POSITIVE_GAIN_MIN_DB, POSITIVE_GAIN_MAX_DB)
    audio = (audio * 10.0 ** (gain_db / 20.0)).astype(np.float32)
    params["gain_db"] = round(float(gain_db), 2)
    return audio, params


SYNTHETIC_NEGATIVE_KINDS = (
    "silence",
    "dither",
    "room_tone",
    "hum",
    "clicks",
    "noise",
    "tone",
    "broadband",
)
SPEECH_NEGATIVE_KINDS = ("hard_negative", "partial_phrase", "reversed_phrase")


def scale_rms(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    level = rms(audio)
    if level < 1e-9:
        return audio.astype(np.float32)
    return (audio * (10.0 ** (target_dbfs / 20.0) / level)).astype(np.float32)


def random_segment(audio: np.ndarray, max_samples: int, rng: random.Random) -> np.ndarray:
    if audio.size <= max_samples:
        return audio
    start = rng.randrange(0, audio.size - max_samples + 1)
    return audio[start : start + max_samples]


def negative_plan(
    count: int,
    rng: random.Random,
    has_hard_negatives: bool,
    has_real_background: bool,
    has_phrase_sources: bool,
) -> list[str]:
    """Return a shuffled list of negative kinds for one split.

    Wake-word clips contain silence around the phrase, so training with only
    broadband noise teaches the model that "quiet = wake word"; the synthetic
    cycle always includes digital silence, dither, room tone and hum. Speech
    negatives (other phrases, partial and reversed wake phrases) stop the
    model from learning "any speech = wake word".
    """
    shares = [
        ("hard_negative", HARD_NEGATIVE_SHARE if has_hard_negatives else 0.0),
        ("real_background", REAL_BACKGROUND_SHARE if has_real_background else 0.0),
        ("partial_phrase", PARTIAL_PHRASE_SHARE if has_phrase_sources else 0.0),
        ("reversed_phrase", REVERSED_PHRASE_SHARE if has_phrase_sources else 0.0),
    ]
    plan: list[str] = []
    for kind, share in shares:
        plan.extend([kind] * int(round(count * share)))
    plan = plan[:count]
    plan.extend(
        SYNTHETIC_NEGATIVE_KINDS[index % len(SYNTHETIC_NEGATIVE_KINDS)]
        for index in range(count - len(plan))
    )
    rng.shuffle(plan)
    return plan


def make_speech_negative(
    kind: str,
    clip_samples: int,
    noise_bank: NoiseBank,
    rng: random.Random,
    index: int,
    phrase_sources: list[tuple[str, np.ndarray]],
    hard_sources: list[tuple[str, np.ndarray]],
    context: ContextFn | None = None,
) -> tuple[np.ndarray, dict]:
    max_samples = int(clip_samples * MAX_PHRASE_FILL)
    if kind == "hard_negative":
        source_name, source = rng.choice(hard_sources)
        speech = random_segment(source, max_samples, rng)
    else:
        source_name, source = rng.choice(phrase_sources)
        if kind == "partial_phrase":
            # Only the beginning or the end of the phrase: on a device the
            # window slides over incomplete phrases before and after the
            # complete one, and those must stay below threshold.
            fraction = rng.uniform(PARTIAL_MIN_FRACTION, PARTIAL_MAX_FRACTION)
            length = max(1, int(source.size * fraction))
            speech = source[:length] if rng.random() < 0.5 else source[-length:]
        else:
            speech = source[::-1]
        speech = speech[:max_samples]
    audio, params = augment_speech(
        np.ascontiguousarray(speech, dtype=np.float32), clip_samples, noise_bank, rng, index, context=context
    )
    params["source"] = source_name
    return audio, params


def make_negative(
    kind: str,
    clip_samples: int,
    noise_bank: NoiseBank,
    rng: random.Random,
    index: int,
    phrase_sources: list[tuple[str, np.ndarray]],
    hard_sources: list[tuple[str, np.ndarray]],
    context: ContextFn | None = None,
) -> tuple[np.ndarray, dict]:
    """Create one negative clip of the given kind (see ``negative_plan``)."""
    if kind in SPEECH_NEGATIVE_KINDS:
        audio, params = make_speech_negative(
            kind, clip_samples, noise_bank, rng, index, phrase_sources, hard_sources, context
        )
        audio = np.clip(audio, -0.98, 0.98).astype(np.float32)
        return audio, {"kind": kind, **params}

    generator = np.random.default_rng(rng.randrange(2**31))

    if kind == "silence":
        audio = np.zeros(clip_samples, dtype=np.float32)
    elif kind == "dither":
        audio = scale_rms(
            generator.normal(0.0, 1.0, clip_samples), rng.uniform(-72.0, -58.0)
        )
    elif kind == "room_tone":
        audio = scale_rms(
            pink_noise(clip_samples, generator), rng.uniform(-64.0, -44.0)
        )
    elif kind == "hum":
        t = np.arange(clip_samples, dtype=np.float32) / SAMPLE_RATE
        base = rng.choice([50.0, 60.0])
        audio = (
            np.sin(2 * np.pi * base * t + rng.random() * 6.28)
            + 0.4 * np.sin(2 * np.pi * base * 2 * t)
            + 0.2 * np.sin(2 * np.pi * base * 3 * t)
            + 0.05 * generator.normal(0.0, 1.0, clip_samples)
        ).astype(np.float32)
        audio = scale_rms(audio, rng.uniform(-52.0, -34.0))
    elif kind == "clicks":
        audio = scale_rms(
            pink_noise(clip_samples, generator), rng.uniform(-66.0, -52.0)
        )
        clicks = max(2, int(clip_samples / SAMPLE_RATE * rng.uniform(2.0, 8.0)))
        for _ in range(clicks):
            position = rng.randrange(max(1, clip_samples - 800))
            decay = np.exp(-np.linspace(0, 14, 800)).astype(np.float32)
            audio[position : position + 800] += decay * rng.uniform(0.02, 0.3)
    elif kind == "tone":
        t = np.arange(clip_samples, dtype=np.float32) / SAMPLE_RATE
        audio = scale_rms(
            pink_noise(clip_samples, generator), rng.uniform(-64.0, -50.0)
        )
        frequency = rng.uniform(300.0, 3000.0)
        audio += (
            rng.uniform(0.01, 0.08) * np.sin(2 * np.pi * frequency * t)
        ).astype(np.float32)
    else:
        # "real_background", "noise" and "broadband": the noise bank returns
        # real recordings when supplied, synthetic noise otherwise.
        audio = noise_bank.get(clip_samples)
        audio = scale_rms(audio, rng.uniform(-40.0, -18.0))
        if kind == "broadband":
            # A second, independent noise bed makes the negative less stationary.
            audio += scale_rms(
                generator.normal(0.0, 1.0, clip_samples), rng.uniform(-48.0, -30.0)
            )

    audio = np.clip(audio, -0.98, 0.98).astype(np.float32)
    return audio, {
        "kind": kind,
        "noise": kind,
        "source": "real" if noise_bank.has_real_noise and kind != "silence" else "synthetic",
    }


def split_sources(
    sources: list, val_ratio: float, test_ratio: float, rng: random.Random
) -> dict[str, list]:
    if len(sources) < 3:
        raise ValueError(
            "At least 3 clean recordings are required for a leakage-free train/val/test split. "
            "Record more examples of your wake phrase (5-10 recommended)."
        )
    shuffled = sources.copy()
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, round(n_total * test_ratio))
    n_val = max(1, round(n_total * val_ratio))
    if n_test + n_val >= n_total:
        n_test = 1
        n_val = 1
    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough recordings to create non-empty train/val/test splits.")
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def fold_split(sources: list, folds: int, fold: int) -> dict[str, list]:
    """Cross-validation split: fold ``fold`` of ``folds`` is held out (half val, half test).

    Takes are assigned to folds by a fixed shuffle of their names, so running
    every fold holds every take out exactly once.
    """
    if not 2 <= folds <= len(sources) or not 0 <= fold < folds:
        raise ValueError(f"--fold must be in 0..{folds - 1} and --folds between 2 and the number of takes.")
    ordered = sorted(sources, key=lambda item: item[0].name)
    random.Random(0).shuffle(ordered)
    held = [item for index, item in enumerate(ordered) if index % folds == fold]
    train = [item for index, item in enumerate(ordered) if index % folds != fold]
    half = max(1, len(held) // 2)
    return {"test": held[:half], "val": held[half:] or held[:1], "train": train}


def load_speech_folder(folder: Path, limit: int = 500) -> list[tuple[str, np.ndarray]]:
    loaded: list[tuple[str, np.ndarray]] = []
    files = list_wavs(folder)
    log(f"Loading {min(len(files), limit)} negative phrase files from {folder}")
    for path in files[:limit]:
        try:
            audio = trim_and_normalize(load_mono(path))
        except Exception as exc:
            print(f"[augment] skipped negative phrase {path.name}: {exc}", file=sys.stderr, flush=True)
            continue
        if audio.size >= int(0.15 * SAMPLE_RATE):
            loaded.append((path.name, audio))
    return loaded


class CorpusPool:
    """Utterances of prepared speech corpora, split train/val/test by utterance.

    Utterances whose transcript contains the wake phrase are dropped: on a
    device they would (rightly) trigger, so they are not negatives.
    """

    def __init__(self, folders: list[Path], phrase: str, val_ratio: float, test_ratio: float, rng: random.Random):
        self.corpora: list[Corpus] = []
        self.pools: dict[str, list[tuple[int, list[int]]]] = {"train": [], "val": [], "test": []}
        self.summary: list[dict] = []
        for folder in folders:
            corpus = Corpus(folder)
            keep = [i for i in range(len(corpus)) if not contains_phrase(corpus.text(i), phrase)]
            parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
            for index in keep:
                parts[utterance_split(corpus.name, index, val_ratio, test_ratio)].append(index)
            position = len(self.corpora)
            self.corpora.append(corpus)
            for split, indices in parts.items():
                if indices:
                    self.pools[split].append((position, indices))
            self.summary.append({
                "name": corpus.name,
                "utterances": len(corpus),
                "hours": round(corpus.hours, 2),
                "excluded_with_phrase": len(corpus) - len(keep),
            })
            log(f"Speech corpus {corpus.name}: {len(keep)} utterances ({corpus.hours:.1f} h), "
                f"{len(corpus) - len(keep)} contain the wake phrase and are skipped")

    def __bool__(self) -> bool:
        return bool(self.corpora)

    def bed(self, split: str, samples: int, rng: random.Random) -> np.ndarray:
        """Back-to-back utterances of this split, at least ``samples`` long."""
        pieces: list[np.ndarray] = []
        total = 0
        while total < samples:
            position, indices = rng.choice(self.pools[split])
            audio = self.corpora[position].audio(rng.choice(indices))
            pieces.append(audio)
            total += audio.size
        bed = np.concatenate(pieces)
        start = rng.randrange(0, bed.size - samples + 1)
        return bed[start : start + samples]

    def draw(self, split: str, max_samples: int, rng: random.Random) -> tuple[str, np.ndarray]:
        position, indices = rng.choice(self.pools[split])
        corpus = self.corpora[position]
        index = rng.choice(indices)
        audio = corpus.audio(index)
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=32)
            if trimmed.size >= SAMPLE_RATE // 5:
                audio = trimmed
        except Exception:
            pass
        audio = random_segment(audio, max_samples, rng)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-7:
            audio = audio / peak
        return f"{corpus.name}:{index}", np.ascontiguousarray(audio, dtype=np.float32)


def load_synthetic(
    folder: Path, max_phrase_samples: int, warnings: list[str]
) -> tuple[list[tuple[str, np.ndarray]], dict[str, list[tuple[str, np.ndarray]]]]:
    """Read TTS clips written by voice.py: positives and negatives grouped by text."""
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        return [], {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError:
        warnings.append("The synthetic voice manifest is unreadable; synthetic clips were skipped.")
        return [], {}
    positives: list[tuple[str, np.ndarray]] = []
    negatives: dict[str, list[tuple[str, np.ndarray]]] = {}
    too_long = 0
    for clip in manifest.get("clips") or []:
        path = folder / clip.get("file", "")
        try:
            audio = trim_and_normalize(load_mono(path))
        except Exception as exc:
            print(f"[augment] skipped synthetic clip {path.name}: {exc}", file=sys.stderr, flush=True)
            continue
        if audio.size == 0:
            continue
        if audio.size > max_phrase_samples:
            too_long += 1
            continue
        source = f"tts:{clip.get('voiceName', clip.get('voice', ''))}:{clip.get('text', '')}"
        if clip.get("label") == "positive":
            positives.append((source, audio))
        else:
            negatives.setdefault(str(clip.get("text", "")), []).append((source, audio))
    if too_long:
        warnings.append(f"Skipped {too_long} synthetic clip(s) longer than the window.")
    log(
        f"Synthetic voices: {len(positives)} wake-phrase clips, "
        f"{sum(len(clips) for clips in negatives.values())} negative clips "
        f"from {len(negatives)} texts"
    )
    return positives, negatives


def write_dataset_info(
    output: Path,
    args: argparse.Namespace,
    counts: dict,
    negative_kinds: dict[str, dict[str, int]],
    source_groups: dict,
    noise_bank: NoiseBank,
    started_at: datetime,
    warnings: list[str],
) -> dict:
    info = {
        "phrase": args.phrase,
        "created_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "sample_rate": SAMPLE_RATE,
        "clip_seconds": args.clip_seconds,
        "variations_per_sample": args.variations,
        "split_ratios": {"val": args.val_ratio, "test": args.test_ratio},
        "counts": counts,
        "negative_kinds": negative_kinds,
        "sources": source_groups,
        "background_source": "real" if noise_bank.has_real_noise else "synthetic",
        "background_files": noise_bank.source_names,
        "negative_phrase_files": source_groups.get("_negative_phrases", []),
        "warnings": warnings,
        "seed": args.seed,
    }
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info


def manifest_row(
    split: str, label: str, filename: str, source: str, variant: int, params: dict
) -> dict:
    snr = params.get("snr_db")
    return {
        "split": split,
        "label": label,
        "file": f"{split}/{label}/{filename}",
        "source": source,
        "kind": params.get("kind", label),
        "variant": variant,
        "noise": params.get("noise", ""),
        "snr_db": snr if snr is not None else "",
        "pitch_semitones": params.get("pitch_semitones", ""),
        "stretch_rate": params.get("stretch_rate", ""),
        "gain_db": params.get("gain_db", ""),
        "reverb": int(bool(params.get("reverb"))),
        "speech_context": int(bool(params.get("speech_context"))),
    }


def run(args: argparse.Namespace) -> dict:
    global POSITIVE_GAIN_MIN_DB
    # The quietest level speech clips are trained at (see the constant).
    POSITIVE_GAIN_MIN_DB = args.min_gain_db
    rng = random.Random(args.seed)
    started_at = datetime.now(timezone.utc)
    input_dir = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    noise_dir = Path(args.background_dir).expanduser().resolve() if args.background_dir else None
    negative_dirs = [Path(item).expanduser().resolve() for item in (args.negative_dir or [])]
    synthetic_dir = Path(args.synthetic_dir).expanduser().resolve() if args.synthetic_dir else None

    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    for negative_dir in negative_dirs:
        if not negative_dir.is_dir():
            raise ValueError(f"Negative phrase directory does not exist: {negative_dir}")
    if args.variations < 10:
        raise ValueError("--variations must be at least 10 (the default is 30).")
    if args.clip_seconds < 1.0 or args.clip_seconds > 4.0:
        raise ValueError("--clip-seconds must be between 1.0 and 4.0 for embedded targets.")

    clips = list_wavs(input_dir)
    if not clips:
        raise ValueError(f"No WAV/FLAC/OGG recordings found in {input_dir}")

    log(f"Found {len(clips)} clean recordings")
    emit_progress("scan", 0.02, f"Found {len(clips)} clean recordings")
    clip_samples = int(round(args.clip_seconds * SAMPLE_RATE))
    max_phrase_samples = int(clip_samples * MAX_PHRASE_FILL)

    # Load every take before splitting so unusable takes cannot leave a split
    # empty. A take longer than the window would be center-cropped and still
    # labelled as a complete wake word, so it is skipped with a warning.
    warnings: list[str] = []
    takes: list[tuple[Path, np.ndarray]] = []
    too_long: list[str] = []
    for path in clips:
        try:
            clean = trim_and_normalize(load_mono(path))
        except Exception as exc:
            print(f"[augment] failed to load {path.name}: {exc}", file=sys.stderr, flush=True)
            continue
        if clean.size == 0:
            continue
        if clean.size > max_phrase_samples:
            too_long.append(f"{path.name} ({clean.size / SAMPLE_RATE:.2f}s)")
            continue
        takes.append((path, clean))
    if too_long:
        message = (
            f"Skipped {len(too_long)} take(s) longer than the {args.clip_seconds:g}s window "
            f"after trimming: {', '.join(too_long)}. Choose a longer window or re-record them."
        )
        log(f"WARNING: {message}")
        warnings.append(message)
    if len(takes) < 3:
        hint = " Choose a longer window." if too_long else ""
        raise ValueError(
            f"Only {len(takes)} usable take(s); at least 3 are required for a leakage-free "
            f"train/val/test split.{hint}"
        )

    if args.folds:
        splits = fold_split(takes, args.folds, args.fold)
        log(f"Cross-validation fold {args.fold + 1}/{args.folds}: holding out "
            f"{', '.join(path.name for path, _ in splits['val'] + splits['test'])}")
    else:
        splits = split_sources(takes, args.val_ratio, args.test_ratio, rng)
    noise_bank = NoiseBank(noise_dir, rng)

    hard_negatives: list[tuple[str, np.ndarray]] = []
    for negative_dir in negative_dirs:
        loaded = load_speech_folder(negative_dir)
        if not loaded:
            warnings.append(f"No usable negative phrase recordings in {negative_dir}.")
        hard_negatives.extend((f"{negative_dir.name}/{name}", audio) for name, audio in loaded)
    synthetic_positives, synthetic_negatives = (
        load_synthetic(synthetic_dir, max_phrase_samples, warnings) if synthetic_dir else ([], {})
    )
    # Split speech negatives by source: a recorded file, or one synthesized
    # text with all its voices. The same word never lands in train and test.
    hard_groups: list[tuple[str, list[tuple[str, np.ndarray]]]] = [
        (name, [(name, audio)]) for name, audio in hard_negatives
    ]
    hard_groups.extend((f"tts:{text}", clips) for text, clips in synthetic_negatives.items())
    if len(hard_groups) >= 3:
        grouped = split_sources(hard_groups, args.val_ratio, args.test_ratio, rng)
    else:
        # Too few sources to keep val/test leak-free: use them for training only.
        grouped = {"train": hard_groups, "val": [], "test": []}
        if hard_groups:
            message = "Fewer than 3 negative phrase sources: they are used in the train split only."
            log(f"WARNING: {message}")
            warnings.append(message)
    hard_splits = {
        split: [clip for _, clips in entries for clip in clips] for split, entries in grouped.items()
    }
    corpus_dirs = [Path(item).expanduser().resolve() for item in (args.speech_corpus or [])]
    for folder in corpus_dirs:
        if not (folder / "index.json").is_file():
            raise ValueError(f"Not a prepared speech corpus: {folder}")
    corpus_pool = CorpusPool(corpus_dirs, args.phrase, args.val_ratio, args.test_ratio, rng)
    mined: list[tuple[str, np.ndarray]] = []
    if args.mined_dir:
        mined_dir = Path(args.mined_dir).expanduser().resolve()
        for path in list_wavs(mined_dir):
            audio = load_mono(path)
            if audio.size:
                mined.append((path.stem, audio))
        log(f"Hard negatives mined by an earlier model: {len(mined)} windows (train split only)")
    corpus_targets = {"train": 0, "val": 0, "test": 0}
    if corpus_pool and args.corpus_negatives > 0:
        corpus_targets["val"] = round(args.corpus_negatives * args.val_ratio)
        corpus_targets["test"] = round(args.corpus_negatives * args.test_ratio)
        corpus_targets["train"] = args.corpus_negatives - corpus_targets["val"] - corpus_targets["test"]

    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise ValueError(
                f"Output directory is not empty: {output}. Pass --overwrite to rebuild the dataset."
            )
        log(f"Removing previous dataset at {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label in (LABEL_WAKE, LABEL_BACKGROUND):
            (output / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {s: {LABEL_WAKE: 0, LABEL_BACKGROUND: 0} for s in splits}
    negative_kinds: dict[str, dict[str, int]] = {s: {} for s in splits}
    synthetic_positive_variants = len(synthetic_positives) * args.synthetic_positive_variations
    # Positives plus the same number of negatives, plus corpus speech.
    total_steps = (
        2 * (len(takes) * args.variations + synthetic_positive_variants)
        + sum(corpus_targets.values())
        + len(mined)
    )
    steps_done = 0

    for split_name in ("train", "val", "test"):
        phrase_sources = [(path.name, clean) for path, clean in splits[split_name]]
        context: ContextFn | None = None
        if corpus_pool and corpus_pool.pools[split_name]:
            context = lambda samples, split=split_name: corpus_pool.bed(split, samples, rng)
        for source_path, clean in splits[split_name]:
            for variant in range(args.variations):
                audio, params = augment_speech(clean, clip_samples, noise_bank, rng, variant, context=context)
                filename = f"{source_path.stem}_{split_name}_p{variant:02d}.wav"
                sf.write(
                    str(output / split_name / LABEL_WAKE / filename),
                    audio,
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
                counts[split_name][LABEL_WAKE] += 1
                manifest_rows.append(
                    manifest_row(split_name, LABEL_WAKE, filename, source_path.name, variant, params)
                )
                steps_done += 1
                emit_progress(
                    "augment",
                    0.05 + 0.85 * (steps_done / max(total_steps, 1)),
                    f"{split_name}: {source_path.stem} · variant {variant + 1}/{args.variations}",
                    positive_clips=counts[split_name][LABEL_WAKE],
                )

        # TTS voices saying the wake phrase add speaker variety, but only to
        # training: val/test stay real recordings so the metrics stay honest.
        if split_name == "train":
            for index, (source, clean) in enumerate(synthetic_positives):
                for variant in range(args.synthetic_positive_variations):
                    # Variant 0 means "keep clean" in augment_speech; TTS audio
                    # is already clean, so always augment it.
                    audio, params = augment_speech(clean, clip_samples, noise_bank, rng, variant + 1, context=context)
                    filename = f"tts_{index:04d}_train_p{variant:02d}.wav"
                    sf.write(
                        str(output / split_name / LABEL_WAKE / filename),
                        audio,
                        SAMPLE_RATE,
                        subtype="PCM_16",
                    )
                    counts[split_name][LABEL_WAKE] += 1
                    params["kind"] = "tts_wake_word"
                    manifest_rows.append(
                        manifest_row(split_name, LABEL_WAKE, filename, source, variant, params)
                    )
                    steps_done += 1
                if index % 4 == 0:
                    emit_progress(
                        "augment",
                        0.05 + 0.85 * (steps_done / max(total_steps, 1)),
                        f"train: synthetic voice {index + 1}/{len(synthetic_positives)}",
                    )

        # Keep classes balanced in every split. Speech negatives come only
        # from sources assigned to this split, so they do not leak either.
        plan = negative_plan(
            counts[split_name][LABEL_WAKE],
            rng,
            has_hard_negatives=bool(hard_splits[split_name]),
            has_real_background=noise_bank.has_real_noise,
            has_phrase_sources=bool(phrase_sources),
        )
        for index, kind in enumerate(plan):
            audio, params = make_negative(
                kind,
                clip_samples,
                noise_bank,
                rng,
                index,
                phrase_sources,
                hard_splits[split_name],
                context,
            )
            filename = f"{split_name}_neg_{index:04d}_{kind}.wav"
            sf.write(
                str(output / split_name / LABEL_BACKGROUND / filename),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            counts[split_name][LABEL_BACKGROUND] += 1
            kinds = negative_kinds[split_name]
            kinds[kind] = kinds.get(kind, 0) + 1
            manifest_rows.append(
                manifest_row(split_name, LABEL_BACKGROUND, filename, params["source"], index, params)
            )
            steps_done += 1
            if index % 8 == 0 or index == len(plan) - 1:
                emit_progress(
                    "negatives",
                    0.05 + 0.85 * (steps_done / max(total_steps, 1)),
                    f"{split_name}: negative {index + 1}/{len(plan)} · {kind.replace('_', ' ')}",
                )
        max_phrase = int(clip_samples * MAX_PHRASE_FILL)
        for index in range(corpus_targets[split_name]):
            source, speech = corpus_pool.draw(split_name, max_phrase, rng)
            audio, params = augment_speech(speech, clip_samples, noise_bank, rng, index + 1, vary_voice=False)
            params["speech_context"] = False
            audio = np.clip(audio, -0.98, 0.98).astype(np.float32)
            params["kind"] = "corpus_speech"
            filename = f"{split_name}_corpus_{index:05d}.wav"
            sf.write(
                str(output / split_name / LABEL_BACKGROUND / filename),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            counts[split_name][LABEL_BACKGROUND] += 1
            kinds = negative_kinds[split_name]
            kinds["corpus_speech"] = kinds.get("corpus_speech", 0) + 1
            manifest_rows.append(
                manifest_row(split_name, LABEL_BACKGROUND, filename, source, index, params)
            )
            steps_done += 1
            if index % 50 == 0 or index == corpus_targets[split_name] - 1:
                emit_progress(
                    "corpus",
                    0.05 + 0.85 * (steps_done / max(total_steps, 1)),
                    f"{split_name}: corpus speech {index + 1}/{corpus_targets[split_name]}",
                )
        # Windows an earlier model fired on. They come from training speech
        # only (see mine.py), so they never leak into validation or test.
        for index, (source, window) in enumerate(mined if split_name == "train" else []):
            audio, params = augment_speech(window, clip_samples, noise_bank, rng, index + 1, vary_voice=False)
            audio = np.clip(audio, -0.98, 0.98).astype(np.float32)
            params["kind"] = "mined"
            params["speech_context"] = False
            filename = f"train_mined_{index:05d}.wav"
            sf.write(str(output / "train" / LABEL_BACKGROUND / filename), audio, SAMPLE_RATE, subtype="PCM_16")
            counts["train"][LABEL_BACKGROUND] += 1
            negative_kinds["train"]["mined"] = negative_kinds["train"].get("mined", 0) + 1
            manifest_rows.append(manifest_row("train", LABEL_BACKGROUND, filename, f"mined:{source}", index, params))
            steps_done += 1
        log(
            f"{split_name}: {counts[split_name][LABEL_WAKE]} wake word / "
            f"{counts[split_name][LABEL_BACKGROUND]} background clips "
            f"({json.dumps(negative_kinds[split_name], sort_keys=True)})"
        )

    emit_progress("manifest", 0.92, "Writing manifest and dataset metadata")

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "label",
                "file",
                "source",
                "kind",
                "variant",
                "noise",
                "snr_db",
                "pitch_semitones",
                "stretch_rate",
                "gain_db",
                "reverb",
                "speech_context",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    source_groups: dict = {
        split: [path.name for path, _ in entries] for split, entries in splits.items()
    }
    source_groups["_background"] = noise_bank.source_names
    source_groups["_negative_phrases"] = [name for name, _ in hard_negatives]
    source_groups["_negative_phrase_splits"] = {
        split: [key for key, _ in entries] for split, entries in grouped.items()
    }
    source_groups["_speech_corpora"] = corpus_pool.summary
    source_groups["_synthetic"] = {
        "positive_clips": len(synthetic_positives),
        "positive_variants": synthetic_positive_variants,
        "negative_texts": len(synthetic_negatives),
    }
    info = write_dataset_info(
        output, args, counts, negative_kinds, source_groups, noise_bank, started_at, warnings
    )

    total = sum(c[LABEL_WAKE] + c[LABEL_BACKGROUND] for c in counts.values())
    log(f"Dataset ready: {total} clips at {output}")
    emit_progress("done", 1.0, f"Dataset ready · {total} clips", total_clips=total)
    return {
        "output": str(output),
        "counts": counts,
        "negative_kinds": negative_kinds,
        "total_clips": total,
        "background_source": info["background_source"],
        "negative_phrase_files": len(hard_negatives),
        "synthetic": source_groups["_synthetic"],
        "speech_corpora": corpus_pool.summary,
        "clip_seconds": args.clip_seconds,
        "variations": args.variations,
        "warnings": warnings,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment wake-word recordings into a dataset")
    parser.add_argument("--input", required=True, help="Folder with clean wake-word recordings")
    parser.add_argument("--output", required=True, help="Destination dataset folder")
    parser.add_argument("--phrase", required=True, help="Wake phrase (stored in metadata)")
    parser.add_argument("--variations", type=int, default=30, help="Variations per recording (>=10)")
    parser.add_argument("--clip-seconds", type=float, default=1.5, help="Fixed model window length")
    parser.add_argument("--background-dir", default=None, help="Optional folder of real background audio")
    parser.add_argument(
        "--negative-dir",
        action="append",
        default=None,
        help="Folder of other words/phrases (ideally similar-sounding) used as negatives; repeatable",
    )
    parser.add_argument(
        "--synthetic-dir",
        default=None,
        help="Optional folder written by voice.py synthesize (manifest.json + clips)",
    )
    parser.add_argument(
        "--speech-corpus",
        action="append",
        default=None,
        help="Prepared speech corpus folder (audio.pcm + index.json); repeatable",
    )
    parser.add_argument(
        "--corpus-negatives",
        type=int,
        default=DEFAULT_CORPUS_NEGATIVES,
        help="Total background clips cut from the speech corpora (all splits)",
    )
    parser.add_argument(
        "--mined-dir",
        default=None,
        help="Folder of hard negative windows written by mine.py (train split only)",
    )
    parser.add_argument(
        "--synthetic-positive-variations",
        type=int,
        default=4,
        help="Augmented variants per synthetic wake-phrase clip (train split only)",
    )
    parser.add_argument(
        "--min-gain-db",
        type=float,
        default=POSITIVE_GAIN_MIN_DB,
        help="Quietest gain for speech clips after normalizing to -14 dBFS peak (-12 close mic ... -30 across a room)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--folds", type=int, default=0, help="Cross-validation: number of take folds (0 = off)")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation: which fold to hold out")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.val_ratio <= 0 or args.test_ratio <= 0:
        parser.error("--val-ratio and --test-ratio must be positive")
    if args.val_ratio + args.test_ratio >= 0.8:
        parser.error("--val-ratio + --test-ratio must be below 0.8")
    return args


def main() -> int:
    configure_stdio()
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        print(f"[augment] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    emit_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
