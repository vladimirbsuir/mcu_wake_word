#!/usr/bin/env python3
"""Stream long audio through an exported wake-word model like the firmware does.

Accuracy on fixed 1.5 s test clips says little about a detector that runs ten
times per second on live audio. This script measures what matters on a device:

* **False activations per hour**: hours of ordinary speech (prepared corpora
  that were never used for training) are joined into one stream and scored
  window by window. Two detector rules are compared: ``all:k`` fires when
  ``k`` consecutive windows score at or above the threshold, ``mean:k`` when
  the average of the last ``k`` scores does. After a detection the detector
  is muted for a refractory period.
* **Recall**: held-out wake-phrase takes are embedded in several seconds of
  context (silence, noise or other speech) at random levels and the same
  detector has to fire around them.

Both are computed for a grid of thresholds and detector rules, so the firmware
threshold can be chosen for a target false-activation rate instead of 0.5.

Progress is emitted as ``@@PROGRESS@@ {json}`` and the result as
``@@RESULT@@ {json}`` (also written to ``evaluation.json`` in the model folder).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import train as T
from augment import NoiseBank, list_wavs, load_mono
from voicekit.corpora import Corpus
from voicekit.groups import contains_phrase

THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05)) + (0.97, 0.98, 0.99)
# "all:k": k windows in a row at or above the threshold. "mean:k": the average
# of the last k scores at or above it, which tolerates one uncertain window.
DETECTORS = ("all:1", "all:2", "all:3", "mean:2", "mean:3", "mean:4", "mean:5")
DEFAULT_STEP_MS = 100
REFRACTORY_SECONDS = 1.5
PAUSE_RANGE_SECONDS = (0.2, 1.2)
CONTEXT_SECONDS = (1.0, 2.5)  # audio before and after an embedded take
CONTEXT_KINDS = ("quiet", "noise", "speech")
TARGET_FA_PER_HOUR = (0.5, 1.0, 2.0)
TOP_FALSE_ACTIVATIONS = 25
INFERENCE_BATCH = 256


def log(message: str) -> None:
    print(f"[evaluate] {message}", flush=True)


class Scorer:
    """Log-Mel frames for a whole stream, then the int8 model on every window."""

    def __init__(self, models_dir: Path, step_ms: int, batch: int = INFERENCE_BATCH):
        import tensorflow as tf

        self.metadata = json.loads((models_dir / "metadata.json").read_text(encoding="utf-8"))
        pre = self.metadata["preprocessing"]
        self.mean = float(pre["logmel_mean"])
        self.std = float(pre["logmel_std"])
        self.frames = int(pre["frame_count"])
        self.step = max(1, int(round(step_ms / 1000 * T.SAMPLE_RATE / T.HOP_LENGTH)))
        model_path = models_dir / "wake_word_int8.tflite"
        if not model_path.is_file():
            raise ValueError(f"{model_path} is missing. Train a model first.")
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        input_index = self.interpreter.get_input_details()[0]["index"]
        try:
            self.interpreter.resize_tensor_input(input_index, [batch, self.frames, T.N_MELS, 1])
            self.interpreter.allocate_tensors()
            self.batch = batch
        except Exception:
            self.interpreter.resize_tensor_input(input_index, [1, self.frames, T.N_MELS, 1])
            self.interpreter.allocate_tensors()
            self.batch = 1
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]

    @property
    def step_seconds(self) -> float:
        return self.step * T.HOP_LENGTH / T.SAMPLE_RATE

    @property
    def window_seconds(self) -> float:
        return ((self.frames - 1) * T.HOP_LENGTH + T.N_FFT) / T.SAMPLE_RATE

    def logmel(self, audio: np.ndarray, chunk_frames: int = 30_000) -> np.ndarray:
        """Frames of the whole stream, identical to ``train.extract_logmel`` per window."""
        import librosa

        total = 1 + (audio.size - T.N_FFT) // T.HOP_LENGTH if audio.size >= T.N_FFT else 0
        parts = []
        for first in range(0, total, chunk_frames):
            count = min(chunk_frames, total - first)
            start = first * T.HOP_LENGTH
            piece = audio[start : start + (count - 1) * T.HOP_LENGTH + T.N_FFT]
            mel = librosa.feature.melspectrogram(
                y=piece,
                sr=T.SAMPLE_RATE,
                n_fft=T.N_FFT,
                win_length=T.WIN_LENGTH,
                hop_length=T.HOP_LENGTH,
                n_mels=T.N_MELS,
                fmin=T.FMIN,
                fmax=T.FMAX,
                window="hann",
                center=False,
                power=2.0,
            )
            parts.append(np.log(mel + 1e-6).T.astype(np.float32))
        if not parts:
            return np.zeros((0, T.N_MELS), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    def scores(self, audio: np.ndarray) -> np.ndarray:
        """Wake score of every window, one per ``step`` frames."""
        features = (self.logmel(audio) - self.mean) / self.std
        starts = np.arange(0, features.shape[0] - self.frames + 1, self.step)
        out = np.zeros(starts.size, dtype=np.float32)
        scale, zero_point = self.output["quantization"]
        for first in range(0, starts.size, self.batch):
            chunk = starts[first : first + self.batch]
            windows = np.stack([features[s : s + self.frames] for s in chunk])[..., None]
            if windows.shape[0] < self.batch:
                pad = np.zeros((self.batch - windows.shape[0], *windows.shape[1:]), dtype=np.float32)
                windows = np.concatenate([windows, pad], axis=0)
            self.interpreter.set_tensor(self.input["index"], T.quantize_input(windows, self.input))
            self.interpreter.invoke()
            raw = self.interpreter.get_tensor(self.output["index"])[: chunk.size]
            probs = (raw.astype(np.float32) - zero_point) * scale
            out[first : first + chunk.size] = probs[:, T.WAKE_CLASS]
        return out


def parse_detector(key: str) -> tuple[str, int]:
    mode, windows = key.split(":")
    return mode, int(windows)


def detector_series(scores: np.ndarray, key: str) -> tuple[np.ndarray, int]:
    """Scores the rule thresholds and how many of them in a row must pass."""
    mode, windows = parse_detector(key)
    if mode == "all":
        return scores, windows
    # Causal moving average, exactly what firmware can keep in a ring buffer.
    cumulative = np.concatenate([[0.0], np.cumsum(scores, dtype=np.float64)])
    index = np.arange(1, scores.size + 1)
    start = np.maximum(0, index - windows)
    return ((cumulative[index] - cumulative[start]) / (index - start)).astype(np.float32), 1


def detections(scores: np.ndarray, threshold: float, consecutive: int, refractory_steps: int) -> list[int]:
    """Window indices where the firmware-style detector fires."""
    fired: list[int] = []
    run = 0
    muted_until = -1
    for index, value in enumerate(scores):
        if index <= muted_until:
            run = 0
            continue
        run = run + 1 if value >= threshold else 0
        if run >= consecutive:
            fired.append(index)
            muted_until = index + refractory_steps
            run = 0
    return fired


def build_stream(corpus: Corpus, phrase: str, max_hours: float, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Join utterances with short quiet pauses; returns audio and (start, end, utterance)."""
    order = [i for i in range(len(corpus)) if not contains_phrase(corpus.text(i), phrase)]
    rng.shuffle(order)
    limit = int(max_hours * 3600 * T.SAMPLE_RATE) if max_hours > 0 else None
    pieces: list[np.ndarray] = []
    spans: list[tuple[int, int, int]] = []
    position = 0
    for index in order:
        pause = np.zeros(int(rng.uniform(*PAUSE_RANGE_SECONDS) * T.SAMPLE_RATE), dtype=np.float32)
        pause += np.random.default_rng(index).normal(0.0, 1e-4, pause.size).astype(np.float32)
        utterance = corpus.audio(index)
        pieces.extend([pause, utterance])
        start = position + pause.size
        spans.append((start, start + utterance.size, index))
        position = start + utterance.size
        if limit is not None and position >= limit:
            break
    return np.concatenate(pieces).astype(np.float32), spans


def false_activations(scorer: Scorer, corpus: Corpus, phrase: str, max_hours: float, rng: random.Random) -> dict:
    audio, spans = build_stream(corpus, phrase, max_hours, rng)
    hours = audio.size / T.SAMPLE_RATE / 3600
    log(f"{corpus.name}: scoring {hours:.2f} h of speech")
    scores = scorer.scores(audio)
    refractory = int(round(REFRACTORY_SECONDS / scorer.step_seconds))
    counts = {}
    for key in DETECTORS:
        series, run = detector_series(scores, key)
        counts[key] = {f"{t:.2f}": len(detections(series, t, run, refractory)) for t in THRESHOLDS}
    # The loudest offenders, to see what the model confuses with the phrase.
    starts = np.array([s for s, _, _ in spans])
    offenders: dict[int, float] = {}
    for window in np.argsort(scores)[::-1][: TOP_FALSE_ACTIVATIONS * 8]:
        center = int(window * scorer.step * T.HOP_LENGTH + scorer.window_seconds * T.SAMPLE_RATE / 2)
        span = int(np.searchsorted(starts, center, side="right")) - 1
        if span < 0:
            continue
        utterance = spans[span][2]
        offenders[utterance] = max(offenders.get(utterance, 0.0), float(scores[window]))
        if len(offenders) >= TOP_FALSE_ACTIVATIONS:
            break
    top = [
        {"utterance": index, "score": round(score, 3), "text": corpus.text(index)}
        for index, score in sorted(offenders.items(), key=lambda item: -item[1])
    ]
    return {"name": corpus.name, "hours": round(hours, 3), "counts": counts, "top": top}


def recall(
    scorer: Scorer,
    takes: list[tuple[str, np.ndarray]],
    contexts: list[Corpus],
    noise_bank: NoiseBank,
    placements: int,
    phrase: str,
    rng: random.Random,
) -> dict:
    refractory = int(round(REFRACTORY_SECONDS / scorer.step_seconds))
    hits = {key: {f"{t:.2f}": 0 for t in THRESHOLDS} for key in DETECTORS}
    misses_at: dict[str, list[dict]] = {}
    trial_log: list[dict] = []
    trials = 0
    for name, take in takes:
        for placement in range(placements):
            kind = CONTEXT_KINDS[placement % len(CONTEXT_KINDS)]
            if kind == "speech" and not contexts:
                kind = "noise"
            before = int(rng.uniform(*CONTEXT_SECONDS) * T.SAMPLE_RATE)
            after = int(rng.uniform(*CONTEXT_SECONDS) * T.SAMPLE_RATE)
            gain_db = rng.uniform(-9.0, 6.0)
            gain = 10.0 ** (gain_db / 20.0)
            speech = take * gain
            level = float(np.sqrt(np.mean(speech**2))) + 1e-9
            total = before + speech.size + after
            if kind == "quiet":
                bed = np.random.default_rng(rng.randrange(2**31)).normal(0.0, 1e-4, total).astype(np.float32)
            elif kind == "noise":
                bed = noise_bank.get(total)
                bed *= level / (10.0 ** (rng.uniform(5.0, 20.0) / 20.0)) / (float(np.sqrt(np.mean(bed**2))) + 1e-9)
            else:
                corpus = rng.choice(contexts)
                pieces = []
                while sum(p.size for p in pieces) < total:
                    index = rng.randrange(len(corpus))
                    if not contains_phrase(corpus.text(index), phrase):
                        pieces.append(corpus.audio(index))
                bed = np.concatenate(pieces)[:total]
                # Other speech before and after, quieter than the wake phrase.
                bed[before : before + speech.size] *= 0.0
                bed *= level / (10.0 ** (rng.uniform(3.0, 12.0) / 20.0)) / (float(np.sqrt(np.mean(bed**2))) + 1e-9)
            audio = bed.astype(np.float32)
            audio[before : before + speech.size] += speech.astype(np.float32)
            scores = scorer.scores(np.clip(audio, -1.0, 1.0))
            # A detection counts when it fires while the window still overlaps the take.
            first = max(0, int((before - scorer.window_seconds * T.SAMPLE_RATE) / (scorer.step * T.HOP_LENGTH)))
            last = int((before + speech.size + 0.5 * T.SAMPLE_RATE) / (scorer.step * T.HOP_LENGTH))
            trials += 1
            # Highest threshold at which each rule still caught this take.
            caught: dict[str, float] = {}
            for key in DETECTORS:
                series, run = detector_series(scores, key)
                caught[key] = 0.0
                for t in THRESHOLDS:
                    fired = detections(series, t, run, refractory)
                    if any(first <= f <= last for f in fired):
                        hits[key][f"{t:.2f}"] += 1
                        caught[key] = max(caught[key], t)
            best = float(scores[first : last + 1].max()) if last >= first and scores.size else 0.0
            trial_log.append({
                "take": name,
                "context": kind,
                "gainDb": round(gain_db, 1),
                "peak": round(best, 3),
                "caught": caught,
            })
            if best < 0.5:
                misses_at.setdefault(name, []).append({"context": kind, "best": round(best, 3)})
    rates = {k: {t: round(v / max(1, trials), 4) for t, v in row.items()} for k, row in hits.items()}
    return {
        "takes": [name for name, _ in takes],
        "trials": trials,
        "recall": rates,
        "weak": misses_at,
        "trialLog": trial_log,
    }


def per_take(rec: dict, point: dict | None) -> list[dict]:
    """Recall of every held-out take at the chosen detector setting, weakest first."""
    if not rec or not point:
        return []
    rows: dict[str, dict] = {}
    for trial in rec.get("trialLog", []):
        row = rows.setdefault(trial["take"], {"take": trial["take"], "trials": 0, "hits": 0, "peaks": [], "contexts": {}})
        hit = trial["caught"][point["detector"]] >= point["threshold"]
        row["trials"] += 1
        row["hits"] += int(hit)
        row["peaks"].append(trial["peak"])
        context = row["contexts"].setdefault(trial["context"], [0, 0])
        context[0] += int(hit)
        context[1] += 1
    result = []
    for row in rows.values():
        result.append({
            "take": row["take"],
            "trials": row["trials"],
            "recall": round(row["hits"] / max(1, row["trials"]), 3),
            "medianPeak": round(float(np.median(row["peaks"])), 3),
            "byContext": {key: f"{hit}/{total}" for key, (hit, total) in row["contexts"].items()},
        })
    return sorted(result, key=lambda item: (item["recall"], item["medianPeak"]))


def operating_points(fa: list[dict], rec: dict) -> list[dict]:
    hours = sum(item["hours"] for item in fa)
    points = []
    for target in TARGET_FA_PER_HOUR:
        best = None
        for detector in DETECTORS:
            mode, windows = parse_detector(detector)
            for t in THRESHOLDS:
                key = f"{t:.2f}"
                rate = sum(item["counts"][detector][key] for item in fa) / max(hours, 1e-9)
                value = rec["recall"][detector][key] if rec else None
                # Ties go to fewer false activations.
                better = best is None or (value or 0) > (best["recall"] or 0) or (
                    (value or 0) == (best["recall"] or 0) and rate < best["faPerHour"]
                )
                if rate <= target and better:
                    best = {
                        "targetFaPerHour": target,
                        "detector": detector,
                        "mode": mode,
                        "windows": windows,
                        "threshold": t,
                        "faPerHour": round(rate, 3),
                        "recall": value,
                    }
        if best:
            points.append(best)
    return points


def write_detector_header(path: Path, result: dict) -> dict | None:
    """Firmware defaults from the operating point closest to 1 false activation per hour."""
    points = result["operatingPoints"]
    if not points:
        return None
    chosen = next((p for p in points if p["targetFaPerHour"] == 1.0), points[-1])
    lines = [
        "/* Auto-generated by WakeForge Studio from the streaming evaluation. */",
        "#ifndef WAKE_WORD_DETECTOR_H",
        "#define WAKE_WORD_DETECTOR_H",
        "",
        f"/* Measured: {chosen['faPerHour']} false activations/hour on {result['speechHours']} h of speech, "
        f"recall {chosen['recall']}. */",
        "/* WW_DETECT_AVERAGE 1: fire when the mean of the last WW_DETECT_WINDOWS wake scores",
        " * reaches WW_DETECT_THRESHOLD. 0: fire when each of the last WW_DETECT_WINDOWS scores does. */",
        f"#define WW_DETECT_AVERAGE {1 if chosen['mode'] == 'mean' else 0}",
        f"#define WW_DETECT_WINDOWS {chosen['windows']}",
        f"#define WW_DETECT_THRESHOLD {chosen['threshold']:.2f}f",
        f"#define WW_DETECT_STEP_MS {result['stepMs']}",
        f"#define WW_DETECT_REFRACTORY_MS {int(result['refractorySeconds'] * 1000)}",
        "",
        "#endif /* WAKE_WORD_DETECTOR_H */",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return chosen


def held_out_takes(dataset: Path, recordings: Path) -> list[Path]:
    info = json.loads((dataset / "dataset_info.json").read_text(encoding="utf-8"))
    names = [*info["sources"].get("test", []), *info["sources"].get("val", [])]
    return [recordings / name for name in names if (recordings / name).is_file()]


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    models_dir = Path(args.models).resolve()
    scorer = Scorer(models_dir, args.step_ms)
    phrase = args.phrase or scorer.metadata.get("phrase", "")
    log(f"Window {scorer.window_seconds:.2f} s, step {scorer.step_seconds * 1000:.0f} ms, batch {scorer.batch}")

    eval_corpora = [Corpus(Path(folder)) for folder in args.corpus or []]
    if not eval_corpora:
        raise ValueError("Pass at least one evaluation corpus (--corpus).")
    fa = []
    for position, corpus in enumerate(eval_corpora):
        T.emit_progress("false_activations", 0.05 + 0.6 * position / len(eval_corpora), f"Streaming {corpus.name}")
        fa.append(false_activations(scorer, corpus, phrase, args.max_hours, rng))

    take_paths = [Path(p) for p in args.takes or []]
    if args.dataset and args.recordings:
        take_paths.extend(held_out_takes(Path(args.dataset), Path(args.recordings)))
    takes = []
    for path in dict.fromkeys(take_paths):
        audio = load_mono(path)
        if audio.size:
            takes.append((path.name, audio))
    rec: dict = {}
    if takes:
        T.emit_progress("recall", 0.7, f"Embedding {len(takes)} held-out take(s) in context")
        noise_bank = NoiseBank(Path(args.background_dir) if args.background_dir else None, rng)
        rec = recall(scorer, takes, eval_corpora, noise_bank, args.placements, phrase, rng)
    else:
        log("WARNING: no wake-phrase takes given; only false activations are measured.")

    hours = sum(item["hours"] for item in fa)
    summary = {
        key: {
            f"{t:.2f}": {
                "faPerHour": round(sum(item["counts"][key][f"{t:.2f}"] for item in fa) / max(hours, 1e-9), 3),
                "recall": rec["recall"][key][f"{t:.2f}"] if rec else None,
            }
            for t in THRESHOLDS
        }
        for key in DETECTORS
    }
    result = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "model": str(models_dir),
        "phrase": phrase,
        "stepMs": round(scorer.step_seconds * 1000),
        "refractorySeconds": REFRACTORY_SECONDS,
        "speechHours": round(hours, 3),
        "falseActivations": fa,
        "recall": rec,
        "curve": summary,
        "operatingPoints": operating_points(fa, rec),
    }
    chosen = next((p for p in result["operatingPoints"] if p["targetFaPerHour"] == 1.0), None)
    result["perTake"] = per_take(rec, chosen or (result["operatingPoints"] or [None])[-1])
    if not args.no_save:
        result["detector"] = write_detector_header(models_dir / "wake_word_detector.h", result)
        (models_dir / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    T.emit_progress("done", 1.0, "Evaluation finished")
    return result


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming false-activation and recall benchmark")
    parser.add_argument("--models", required=True, help="Folder with wake_word_int8.tflite and metadata.json")
    parser.add_argument("--corpus", action="append", help="Prepared evaluation corpus folder; repeatable")
    parser.add_argument("--takes", nargs="*", help="Wake-phrase recordings for the recall test")
    parser.add_argument("--dataset", default=None, help="Dataset folder: use its val/test takes for recall")
    parser.add_argument("--recordings", default=None, help="Recordings folder that --dataset refers to")
    parser.add_argument("--background-dir", default=None)
    parser.add_argument("--phrase", default=None, help="Wake phrase (defaults to the model metadata)")
    parser.add_argument("--placements", type=int, default=12, help="Contexts per take for the recall test")
    parser.add_argument("--step-ms", type=int, default=DEFAULT_STEP_MS)
    parser.add_argument("--max-hours", type=float, default=0.0, help="Limit each corpus stream (0 = all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    T.configure_stdio()
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[evaluate] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    T.emit_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
