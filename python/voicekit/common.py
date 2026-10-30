"""Job protocol, logging and download helpers shared by all voicekit commands."""

from __future__ import annotations

import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

PROGRESS_PREFIX = "@@PROGRESS@@"
RESULT_PREFIX = "@@RESULT@@"
USER_AGENT = "WakeForgeStudio/0.1 (+local model installer)"


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def log(message: str) -> None:
    print(f"[voice] {message}", flush=True)


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


class VoiceError(Exception):
    """An expected failure with a message that is safe to show to the user."""


ProgressFn = Callable[[float, str], None]


def download(url: str, destination: Path, on_progress: ProgressFn | None = None) -> None:
    """Stream ``url`` to ``destination`` via a ``.part`` file.

    An existing ``.part`` file from an interrupted or canceled run is resumed
    with an HTTP Range request, so multi-gigabyte models never restart from zero.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    resume_from = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            resumed = resume_from > 0 and response.status == 206
            if resume_from and not resumed:
                log(f"Server ignored the resume request for {destination.name}; restarting")
            done = resume_from if resumed else 0
            total = int(response.headers.get("Content-Length") or 0) + done
            last_report = 0.0
            with partial.open("ab" if resumed else "wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if on_progress and (now - last_report > 0.4):
                        last_report = now
                        fraction = done / total if total else 0.0
                        on_progress(fraction, f"{done / 1048576:.0f}/{total / 1048576:.0f} MB")
            if total and done < total:
                raise VoiceError(f"connection closed after {done} of {total} bytes")
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and resume_from:
            # The partial file already holds the whole download.
            partial.replace(destination)
            return
        raise VoiceError(f"Download failed for {url}: HTTP {exc.code}") from exc
    except Exception as exc:
        # Keep the .part file: the next attempt resumes from it.
        raise VoiceError(f"Download failed for {url}: {exc}. Retry to resume.") from exc
    partial.replace(destination)


def extract(archive: Path, destination: Path, strip_top_level: bool = True) -> None:
    """Extract a .zip or .tar.* archive, optionally dropping its single top folder."""
    staging = destination.with_name(destination.name + ".extract")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
    elif ".tar" in name:
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
    else:
        raise VoiceError(f"Unsupported archive type: {archive.name}")
    root = staging
    entries = list(staging.iterdir())
    if strip_top_level and len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
    destination.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        target = destination / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(staging, ignore_errors=True)


def read_json(path: Path, default: object) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
