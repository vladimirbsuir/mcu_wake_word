#!/usr/bin/env python3
"""Live test: run the exported model on the PC microphone in real time.

Every 100 ms the last window of audio goes through the same INT8 model and
the same detector rules as the streaming test and the ESP32 firmware. Each
detection saves the audio around it, and a "missed" request (the app creates
``<clips>/.snap``) saves the last few seconds, so both can be listened to and
added to the training data.

Progress lines (``@@PROGRESS@@ {json}``) carry the current score, the detector
value, the input level and the recent events; the app draws them live. The
job runs until it is canceled. ``--input-file`` streams a WAV instead of the
microphone (for testing).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

import train as T
from evaluate import Scorer

STEP_MS = 100
RING_SECONDS = 8
CLIP_BEFORE_SECONDS = 2.0
CLIP_AFTER_SECONDS = 0.6
SNAPSHOT_SECONDS = 4.0
MAX_EVENTS = 30


class Detector:
    """Same rules as evaluate.py and the firmware: "all:k" or "mean:k"."""

    def __init__(self, average: bool, windows: int, threshold: float, refractory_steps: int):
        self.average = average
        self.windows = max(1, windows)
        self.threshold = threshold
        self.refractory = refractory_steps
        self.history: deque[float] = deque(maxlen=self.windows)
        self.run = 0
        self.muted = 0

    def push(self, score: float) -> tuple[bool, float]:
        self.history.append(score)
        if self.average:
            value = float(np.mean(self.history))
            fire = value >= self.threshold
        else:
            value = score
            self.run = self.run + 1 if score >= self.threshold else 0
            fire = self.run >= self.windows
        if self.muted > 0:
            self.muted -= 1
            self.run = 0
            return False, value
        if fire:
            self.muted = self.refractory
            self.run = 0
            return True, value
        return False, value


def db(value: float) -> float:
    return round(20 * float(np.log10(value)), 1) if value > 1e-9 else -120.0


class Source:
    """16 kHz float32 blocks from the microphone (resampled) or a WAV file."""

    def __init__(self, args: argparse.Namespace):
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()
        self.file = Path(args.input_file) if args.input_file else None
        self.name = "file"
        if self.file:
            audio, rate = sf.read(str(self.file), dtype="float32", always_2d=True)
            if rate != T.SAMPLE_RATE:
                import librosa

                audio = librosa.resample(audio[:, 0], orig_sr=rate, target_sr=T.SAMPLE_RATE)[:, None]
            self._file_audio = audio[:, 0]
            self.name = self.file.name
            return
        try:
            import sounddevice as sd
        except ImportError:
            # First run: the package was added after the environment was set up.
            import subprocess

            T.emit_progress("live", 0.0, "Installing sounddevice")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "sounddevice"])
            import sounddevice as sd
        import soxr
        device = sd.query_devices(args.device, kind="input") if args.device is not None else sd.query_devices(kind="input")
        self.name = device["name"]
        self.rate = int(device["default_samplerate"])
        self.resampler = soxr.ResampleStream(self.rate, T.SAMPLE_RATE, 1, dtype="float32", quality="HQ")

        def callback(indata, frames, time_info, status):  # audio thread
            self.blocks.put(indata[:, 0].copy())

        self.stream = sd.InputStream(
            device=args.device,
            samplerate=self.rate,
            channels=1,
            dtype="float32",
            blocksize=self.rate // 50,
            callback=callback,
        )
        self.stream.start()

    def read(self) -> np.ndarray | None:
        """Next block of 16 kHz audio, None at the end of a file."""
        if self.file:
            if not hasattr(self, "_position"):
                self._position = 0
            if self._position >= self._file_audio.size:
                return None
            block = self._file_audio[self._position : self._position + 320]
            self._position += 320
            return block
        raw = self.blocks.get()
        return self.resampler.resample_chunk(raw)


def run(args: argparse.Namespace) -> dict:
    models = Path(args.models).resolve()
    clips = Path(args.clips_dir).resolve()
    clips.mkdir(parents=True, exist_ok=True)
    for old in clips.glob("*.wav"):
        old.unlink()
    snap_flag = clips / ".snap"
    snap_flag.unlink(missing_ok=True)

    scorer = Scorer(models, STEP_MS, batch=1)
    window = int(round(scorer.window_seconds * T.SAMPLE_RATE))
    step = STEP_MS * T.SAMPLE_RATE // 1000
    detector = Detector(args.rule == "mean", args.windows, args.threshold, args.refractory_ms // STEP_MS)
    gain = 10 ** (args.gain_db / 20)
    source = Source(args)
    T.emit_progress("live", 0.0, f"Listening on {source.name}", device=source.name)

    ring = np.zeros(RING_SECONDS * T.SAMPLE_RATE, dtype=np.float32)
    filled = 0                     # samples written since start
    pending = np.zeros(0, dtype=np.float32)
    events: deque[dict] = deque(maxlen=MAX_EVENTS)
    scheduled: list[tuple[int, dict]] = []  # (save at sample, event)
    detections = 0
    counter = 0
    started = time.monotonic()

    def last(seconds_before: float, end: int) -> np.ndarray:
        count = min(int(seconds_before * T.SAMPLE_RATE), filled, ring.size)
        indices = np.arange(end - count, end) % ring.size
        return ring[indices]

    def save(event: dict, audio: np.ndarray) -> None:
        sf.write(str(clips / event["file"]), np.clip(audio, -1, 1), T.SAMPLE_RATE, subtype="PCM_16")
        event["saved"] = True

    while True:
        block = source.read()
        if block is None:
            break
        if block.size == 0:
            continue
        pending = np.concatenate([pending, block * gain])
        while pending.size >= step:
            chunk, pending = pending[:step], pending[step:]
            indices = (filled + np.arange(step)) % ring.size
            ring[indices] = chunk
            filled += step
            if filled < window:
                continue
            audio = last(window / T.SAMPLE_RATE + 1e-9, filled)[-window:]
            score = float(scorer.scores(np.clip(audio, -1.0, 1.0))[-1])
            fired, value = detector.push(score)
            ms = int(filled * 1000 / T.SAMPLE_RATE)
            if fired:
                detections += 1
                counter += 1
                event = {
                    "id": counter,
                    "kind": "detection",
                    "ms": ms,
                    "at": datetime.now().strftime("%H:%M:%S"),
                    "score": round(score, 3),
                    "rule": round(value, 3),
                    "peakDb": db(float(np.abs(audio).max())),
                    "file": f"detection_{counter:03d}.wav",
                    "saved": False,
                }
                events.appendleft(event)
                scheduled.append((filled + int(CLIP_AFTER_SECONDS * T.SAMPLE_RATE), event))
                print(f"[live] detection {detections}: score {score:.2f}", flush=True)
            if snap_flag.exists():
                snap_flag.unlink(missing_ok=True)
                counter += 1
                event = {
                    "id": counter,
                    "kind": "missed",
                    "ms": ms,
                    "at": datetime.now().strftime("%H:%M:%S"),
                    "file": f"missed_{counter:03d}.wav",
                    "saved": False,
                }
                save(event, last(SNAPSHOT_SECONDS, filled))
                events.appendleft(event)
            for item in [entry for entry in scheduled if entry[0] <= filled]:
                scheduled.remove(item)
                save(item[1], last(CLIP_BEFORE_SECONDS + CLIP_AFTER_SECONDS, filled))
            T.emit_progress(
                "live",
                0.5,
                f"score {score:.2f}",
                score=round(score, 3),
                rule=round(value, 3),
                rmsDb=db(float(np.sqrt(np.mean(chunk**2)))),
                peakDb=db(float(np.abs(chunk).max())),
                ms=ms,
                detections=detections,
                events=list(events),
                device=source.name,
            )
        if args.duration and time.monotonic() - started > args.duration:
            break
    for at, event in scheduled:
        save(event, last(CLIP_BEFORE_SECONDS + CLIP_AFTER_SECONDS, filled))
    return {"detections": detections, "seconds": round(filled / T.SAMPLE_RATE, 1), "events": list(events)}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live wake-word test on the PC microphone")
    parser.add_argument("--models", required=True)
    parser.add_argument("--clips-dir", required=True, help="Where detection and missed clips are written")
    parser.add_argument("--rule", choices=("all", "mean"), default="all")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--refractory-ms", type=int, default=1500)
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="Input device index or name (default: system default)")
    parser.add_argument("--input-file", default=None, help="Stream a WAV file instead of the microphone")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds (0 = until canceled)")
    args = parser.parse_args(argv)
    if args.device is not None and args.device.isdigit():
        args.device = int(args.device)
    return args


def main() -> int:
    T.configure_stdio()
    args = parse_args()
    try:
        result = run(args)
    except KeyboardInterrupt:
        result = {"stopped": True}
    except Exception as exc:
        print(f"[live] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    T.emit_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
