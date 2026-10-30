#!/usr/bin/env python3
"""Hard negative mining: find the speech the current model mistakes for the phrase.

The trained INT8 model is streamed over the *training* part of every speech
corpus (the same hash split ``augment.py`` uses, so validation and test speech
is never touched). Windows that score at or above ``--min-score`` are saved,
loudest first, at most one per utterance second, into ``<project>/mined/``.
The next dataset build adds them to the training negatives; retraining and
mining again usually lowers false activations several times over.

Progress is emitted as ``@@PROGRESS@@ {json}`` and the result as
``@@RESULT@@ {json}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

import train as T
from evaluate import Scorer
from voicekit.corpora import Corpus, utterance_split
from voicekit.groups import contains_phrase

DEFAULT_MIN_SCORE = 0.3
DEFAULT_MAX_WINDOWS = 1500
# Matches the default dataset split ratios of augment.py.
VAL_RATIO = 0.15
TEST_RATIO = 0.15
CHUNK_UTTERANCES = 400  # utterances scored per stream
SEPARATION_SECONDS = 1.0  # keep at most one window per second of one utterance


def log(message: str) -> None:
    print(f"[mine] {message}", flush=True)


def mine_corpus(scorer: Scorer, corpus: Corpus, phrase: str, min_score: float, progress) -> list[dict]:
    """Score training utterances in chunks and return candidate windows."""
    indices = [
        i
        for i in range(len(corpus))
        if utterance_split(corpus.name, i, VAL_RATIO, TEST_RATIO) == "train"
        and not contains_phrase(corpus.text(i), phrase)
    ]
    window = int(round(scorer.window_seconds * T.SAMPLE_RATE))
    hop = scorer.step * T.HOP_LENGTH
    found: list[dict] = []
    for first in range(0, len(indices), CHUNK_UTTERANCES):
        chunk = indices[first : first + CHUNK_UTTERANCES]
        # Pad every utterance with half a window of silence so the phrase-like
        # part can sit anywhere in the window, as on the device.
        pad = np.zeros(window // 2, dtype=np.float32)
        pieces, spans, position = [], [], 0
        for index in chunk:
            audio = corpus.audio(index)
            pieces.extend([pad, audio])
            spans.append((position + pad.size, position + pad.size + audio.size, index))
            position += pad.size + audio.size
        pieces.append(pad)
        stream = np.concatenate(pieces)
        scores = scorer.scores(stream)
        starts = np.array([s for s, _, _ in spans])
        for w in np.flatnonzero(scores >= min_score):
            begin = int(w * hop)
            center = begin + window // 2
            span = int(np.searchsorted(starts, center, side="right")) - 1
            if span < 0 or center >= spans[span][1]:
                continue  # window centred on padding
            found.append({
                "corpus": corpus.name,
                "utterance": spans[span][2],
                "offset": begin - spans[span][0],
                "score": float(scores[w]),
                "audio": stream[begin : begin + window].copy(),
            })
        progress(min(1.0, (first + len(chunk)) / max(1, len(indices))))
    return found


def select(candidates: list[dict], max_windows: int, existing: set[str]) -> list[dict]:
    """Loudest first, one window per utterance second, skipping already mined ones."""
    chosen: list[dict] = []
    taken: dict[tuple[str, int], list[int]] = {}
    separation = int(SEPARATION_SECONDS * T.SAMPLE_RATE)
    for item in sorted(candidates, key=lambda c: -c["score"]):
        key = (item["corpus"], item["utterance"])
        if any(abs(item["offset"] - other) < separation for other in taken.get(key, [])):
            continue
        name = f"{item['corpus']}_{item['utterance']}_{item['offset'] // separation}"
        if name in existing:
            continue
        taken.setdefault(key, []).append(item["offset"])
        item["name"] = name
        chosen.append(item)
        if len(chosen) >= max_windows:
            break
    return chosen


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"windows": []}
    existing = {item["name"] for item in manifest["windows"]}

    scorer = Scorer(Path(args.models).resolve(), args.step_ms)
    corpora = [Corpus(Path(folder)) for folder in args.corpus or []]
    if not corpora:
        raise ValueError("Pass at least one training speech corpus (--corpus).")
    candidates: list[dict] = []
    for position, corpus in enumerate(corpora):
        log(f"Scanning {corpus.name} ({corpus.hours:.1f} h, training part)")

        def progress(fraction: float, position=position, name=corpus.name) -> None:
            T.emit_progress("mine", 0.02 + 0.9 * (position + fraction) / len(corpora), f"Scanning {name} · {fraction:.0%}")

        candidates.extend(mine_corpus(scorer, corpus, args.phrase, args.min_score, progress))
    chosen = select(candidates, args.max_windows, existing)
    for item in chosen:
        sf.write(str(output / f"{item['name']}.wav"), item.pop("audio"), T.SAMPLE_RATE, subtype="PCM_16")
        item["score"] = round(item["score"], 3)
        item["text"] = next(c for c in corpora if c.name == item["corpus"]).text(item["utterance"])
    manifest["windows"].extend(chosen)
    manifest["updatedAt"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"{len(candidates)} windows scored >= {args.min_score}; saved {len(chosen)} new, {len(manifest['windows'])} in total")
    T.emit_progress("done", 1.0, f"Saved {len(chosen)} hard negatives")
    return {
        "output": str(output),
        "candidates": len(candidates),
        "added": len(chosen),
        "total": len(manifest["windows"]),
        "examples": [{"score": c["score"], "text": c["text"]} for c in chosen[:10]],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard negatives with the current model")
    parser.add_argument("--models", required=True)
    parser.add_argument("--corpus", action="append", help="Prepared training speech corpus; repeatable")
    parser.add_argument("--output", required=True, help="Folder for mined windows (appended to)")
    parser.add_argument("--phrase", required=True)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    parser.add_argument("--max-windows", type=int, default=DEFAULT_MAX_WINDOWS)
    parser.add_argument("--step-ms", type=int, default=100)
    return parser.parse_args(argv)


def main() -> int:
    T.configure_stdio()
    args = parse_args()
    try:
        result = run(args)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        print(f"[mine] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    T.emit_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
