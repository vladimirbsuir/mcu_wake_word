"""Project-level tasks: read the phrase from takes, write word lists, voice them."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from . import groups as word_groups
from .base import SttEngine, TtsEngine, Voice
from .common import VoiceError, emit_progress, log

SAMPLE_RATE = 16_000
TRIM_TOP_DB = 40
EDGE_PAD_SECONDS = 0.05
POSITIVE_SPEEDS = (0.85, 1.0, 1.15)
NEGATIVE_SPEED_RANGE = (0.9, 1.1)
MAX_POSITIVE_VOICES = 40


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(sample_rate)


def transcribe_takes(stt: SttEngine, recordings: Path) -> dict:
    files = sorted(path for path in recordings.glob("*.wav") if path.is_file())
    if not files:
        raise VoiceError("The project has no recorded takes to transcribe.")
    transcripts = []
    for index, path in enumerate(files):
        emit_progress("transcribe", (index + 0.5) / len(files), f"Transcribing {path.stem}")
        audio, sample_rate = _load_mono(path)
        text = stt.transcribe(audio, sample_rate)
        log(f"{path.name}: {text!r}")
        transcripts.append({"file": path.name, "text": text})
    votes = Counter(word_groups.normalize(item["text"]) for item in transcripts if item["text"].strip())
    if not votes:
        raise VoiceError("Speech recognition returned no text for any take.")
    best, count = votes.most_common(1)[0]
    # Report the most common spelling as the model wrote it (with capitals etc.).
    original = next(item["text"] for item in transcripts if word_groups.normalize(item["text"]) == best)
    return {
        "suggestion": original.strip().rstrip("."),
        "agreement": round(count / len(transcripts), 3),
        "transcripts": transcripts,
    }


def _prepare(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import librosa

    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32))
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    trimmed, (start, end) = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)
    if trimmed.size == 0:
        return trimmed
    pad = int(EDGE_PAD_SECONDS * SAMPLE_RATE)
    audio = audio[max(0, start - pad) : min(audio.size, end + pad)]
    peak = float(np.max(np.abs(audio)))
    return (audio / peak * 0.9).astype(np.float32) if peak > 1e-6 else audio


ENGINES_DIR = "engines"
CACHE_FILE = "cache.json"
CACHE_SAVE_EVERY = 25


def _job_key(job: dict, language: str) -> str:
    return f"{job['label']}|{language}|{job['text']}|{job['voice'].id}|{job['speed']}"


def negative_share(text: str, engine_key: str, engine_keys: list[str], voices_per_word: int) -> int:
    """How many of a word's voices this engine provides.

    ``voices_per_word`` is the total over all selected engines: with three
    engines and 2 voices per word a word is still voiced twice, by two of the
    engines, not six times. Engines take turns in an order seeded by the word,
    so the split is stable and spreads evenly over the word list.
    """
    keys = sorted(set(engine_keys) | {engine_key})
    ranking = sorted(keys, key=lambda key: hashlib.sha1(f"{text}|{key}".encode("utf-8")).hexdigest())
    return sum(1 for slot in range(voices_per_word) if ranking[slot % len(ranking)] == engine_key)


def desired_jobs(
    voices: list,
    engine_key: str,
    voice_data: dict,
    include_positives: bool,
    voices_per_negative: int,
    engine_keys: list[str] | None = None,
) -> list[dict]:
    """Every clip one engine should provide for the current lists and phrase.

    Voices and speeds are drawn from a generator seeded by the engine and the
    text, so the same word always maps to the same voices: a later run with a
    longer list, more voices per word or another engine added only needs the
    clips that are actually new.
    """
    spoken = (voice_data.get("spokenText") or "").strip()
    jobs: list[dict] = []
    for group_id, group in (voice_data.get("groups") or {}).items():
        definition = word_groups.BY_ID.get(group_id)
        label = definition.label if definition else "negative"
        for item in group.get("items") or []:
            rng = random.Random(f"{engine_key}|{item}")
            order = list(voices)
            rng.shuffle(order)
            share = negative_share(item, engine_key, engine_keys or [engine_key], voices_per_negative)
            for voice in order[: min(share, len(order))]:
                jobs.append({
                    "label": label, "group": group_id, "text": item, "voice": voice,
                    "speed": round(rng.uniform(*NEGATIVE_SPEED_RANGE), 3),
                })
    if include_positives:
        if not spoken:
            raise VoiceError("Set the spoken phrase before synthesizing positives.")
        positive_voices = list(voices)
        if len(positive_voices) > MAX_POSITIVE_VOICES:
            random.Random(f"{engine_key}|positives").shuffle(positive_voices)
            positive_voices = positive_voices[:MAX_POSITIVE_VOICES]
        for voice in positive_voices:
            for speed in POSITIVE_SPEEDS:
                jobs.append({"label": "positive", "group": "wake_phrase", "text": spoken, "voice": voice, "speed": speed})
    return jobs


def _save_cache(path: Path, cache: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def _load_cache(output: Path, engine_key: str) -> dict:
    try:
        cache = json.loads((output / ENGINES_DIR / engine_key / CACHE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    cache.setdefault("clips", {})
    return cache


def _engine_result(cache: dict, engine_key: str, engine_label: str, keys: list[str], synthesized: int) -> dict:
    unique = list(dict.fromkeys(keys))
    return {
        "key": engine_key,
        "label": engine_label,
        "wanted": len(unique),
        "synthesized": synthesized,
        "reused": len(unique) - synthesized,
        "failures": sum(1 for key in unique if cache["clips"][key].get("failed")),
        "clips": [cache["clips"][key] for key in unique if not cache["clips"][key].get("failed")],
    }


def cached_engine(
    engine_key: str,
    engine_label: str,
    voice_data: dict,
    output: Path,
    include_positives: bool,
    voices_per_negative: int,
    engine_keys: list[str] | None = None,
) -> dict | None:
    """The engine's result when every wanted clip is cached already, so the model need not load."""
    cache = _load_cache(output, engine_key)
    voices = [Voice(item["id"], item["name"]) for item in cache.get("voices") or []]
    if not voices or cache.get("language") != (voice_data.get("language") or ""):
        return None
    jobs = desired_jobs(voices, engine_key, voice_data, include_positives, voices_per_negative, engine_keys)
    keys = [_job_key(job, voice_data.get("language") or "") for job in jobs]
    if any(key not in cache["clips"] for key in keys):
        return None
    return _engine_result(cache, engine_key, engine_label, keys, 0)


def synthesize_engine(
    tts: TtsEngine,
    engine_key: str,
    engine_label: str,
    voice_data: dict,
    output: Path,
    include_positives: bool,
    voices_per_negative: int,
    progress=None,
    engine_keys: list[str] | None = None,
) -> dict:
    """Voice what this engine does not have yet; return the clips it should contribute.

    Clips live in ``output/engines/<engine_key>/`` with a ``cache.json`` keyed
    by label, language, text, voice and speed. The cache is saved every few
    clips, so a canceled run keeps what it already made.
    """
    voices = tts.voices()
    if not voices:
        raise VoiceError(f"{engine_label} has no voices for this language.")
    language = voice_data.get("language") or ""
    folder = output / ENGINES_DIR / engine_key
    folder.mkdir(parents=True, exist_ok=True)
    cache_path = folder / CACHE_FILE
    cache = _load_cache(output, engine_key)
    cache["engine"] = engine_label
    # The voice list depends on the language for some engines.
    cache["language"] = language
    cache["voices"] = [{"id": voice.id, "name": voice.name} for voice in voices]

    jobs = desired_jobs(voices, engine_key, voice_data, include_positives, voices_per_negative, engine_keys)
    keys = [_job_key(job, language) for job in jobs]
    missing = [(job, key) for job, key in zip(jobs, keys) if key not in cache["clips"]]
    log(f"{engine_label}: {len(jobs)} clips wanted, {len(jobs) - len(missing)} already made, {len(missing)} to synthesize")
    failures = 0
    for index, (job, key) in enumerate(missing):
        voice = job["voice"]
        if progress:
            progress(index / max(1, len(missing)), f"{engine_label} · {job['label']}: “{job['text']}” · {voice.name}")
        name = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".wav"
        try:
            audio, sample_rate = tts.synthesize(job["text"], voice.id, job["speed"])
            audio = _prepare(audio, sample_rate)
        except VoiceError as exc:
            log(f"WARNING: {exc}")
            audio = np.zeros(0, dtype=np.float32)
        if audio.size < int(0.1 * SAMPLE_RATE):
            # Remembered as failed, so it is not retried on every run.
            failures += 1
            cache["clips"][key] = {"failed": True}
            log(f"WARNING: no usable audio for {job['text']!r} ({voice.name})")
        else:
            sf.write(str(folder / name), audio, SAMPLE_RATE, subtype="PCM_16")
            cache["clips"][key] = {
                "file": f"{ENGINES_DIR}/{engine_key}/{name}",
                "label": job["label"],
                "group": job["group"],
                "text": job["text"],
                "voice": f"{engine_key}:{voice.id}",
                "voiceName": voice.name,
                "engine": engine_label,
                "speed": job["speed"],
                "seconds": round(audio.size / SAMPLE_RATE, 3),
            }
        if (index + 1) % CACHE_SAVE_EVERY == 0:
            _save_cache(cache_path, cache)
    _save_cache(cache_path, cache)
    return _engine_result(cache, engine_key, engine_label, keys, len(missing) - failures)


def write_synthetic_manifest(output: Path, voice_data: dict, engines: list[dict]) -> dict:
    """The set ``augment.py`` uses: only clips of the selected engines for the current lists."""
    clips = [clip for engine in engines for clip in engine["clips"]]
    if not clips:
        raise VoiceError("Synthesis produced no usable clips. Check the TTS models and the log.")
    counts = Counter(clip["label"] for clip in clips)
    manifest = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "engine": " + ".join(engine["label"] for engine in engines),
        "engines": [
            {key: engine[key] for key in ("key", "label", "wanted", "synthesized", "reused", "failures")}
            | {"voices": len({clip["voice"] for clip in engine["clips"]})}
            for engine in engines
        ],
        "language": voice_data.get("language") or "",
        "spokenText": (voice_data.get("spokenText") or "").strip(),
        "voices": len({clip["voice"] for clip in clips}),
        "counts": {"positive": counts.get("positive", 0), "negative": counts.get("negative", 0)},
        "failures": sum(engine["failures"] for engine in engines),
        "clips": clips,
    }
    _save_cache(output / "manifest.json", manifest)
    # Files of the earlier single-engine layout are no longer referenced.
    for legacy in ("positive", "negative"):
        shutil.rmtree(output / legacy, ignore_errors=True)
    return {key: manifest[key] for key in ("createdAt", "engine", "engines", "voices", "counts", "failures")}
