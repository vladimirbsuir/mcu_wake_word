"""Downloadable speech corpora: real speech that must *not* wake the device.

Two roles:

* ``negatives`` corpora are cut into windows by ``augment.py`` and become
  background clips in every split, so the model learns that ordinary speech
  from hundreds of speakers is not the wake phrase.
* ``eval`` corpora are never trained on. ``evaluate.py`` streams them through
  the exported model like the firmware would and counts false activations
  per hour.

A prepared corpus is one folder with ``audio.pcm`` (16 kHz mono int16, all
utterances back to back) and ``index.json`` (offset, length and transcript of
every utterance), so readers can memory-map hours of speech cheaply.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .common import VoiceError, download, log

SAMPLE_RATE = 16_000
INSTALLED_MARKER = ".installed.json"
HF = "https://huggingface.co/datasets"
GOLOS_CROWD = f"{HF}/bond005/sberdevices_golos_10h_crowd/resolve/main/data"
GOLOS_FARFIELD = f"{HF}/bond005/sberdevices_golos_100h_farfield/resolve/main/data"
FLEURS = f"{HF}/google/fleurs/resolve/main/data"


@dataclass(frozen=True)
class CorpusFile:
    url: str
    name: str
    size_mb: float


@dataclass(frozen=True)
class CorpusSpec:
    id: str
    role: str  # "negatives" | "eval"
    title: str
    description: str
    language: str
    format: str  # "parquet" (audio struct + transcription) | "fleurs" (tar.gz + tsv)
    files: tuple[CorpusFile, ...]
    hours: float
    license: str
    homepage: str

    @property
    def size_mb(self) -> float:
        return sum(item.size_mb for item in self.files)


CATALOG: tuple[CorpusSpec, ...] = (
    CorpusSpec(
        id="golos-crowd",
        role="negatives",
        title="Golos crowd (SberDevices)",
        description="Voice-assistant requests read by crowd workers on their phones: "
        "short commands, many speakers, close microphone.",
        language="ru",
        format="parquet",
        files=(
            CorpusFile(f"{GOLOS_CROWD}/train-00000-of-00003-85be0fb4b3de87ab.parquet", "train-0.parquet", 330.4),
            CorpusFile(f"{GOLOS_CROWD}/train-00001-of-00003-fcf1367484d8316d.parquet", "train-1.parquet", 324.5),
            CorpusFile(f"{GOLOS_CROWD}/train-00002-of-00003-8e44cef5e8ee83d1.parquet", "train-2.parquet", 326.1),
        ),
        hours=9.0,
        license="SberDevices Golos license (non-commercial use)",
        homepage="https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd",
    ),
    CorpusSpec(
        id="golos-farfield",
        role="negatives",
        title="Golos far-field (SberDevices)",
        description="Smart-speaker requests recorded across a room: reverberant, "
        "distant speech like a real device hears.",
        language="ru",
        format="parquet",
        files=(
            CorpusFile(f"{GOLOS_FARFIELD}/train-00000-of-00003-066abd799f2fa68d.parquet", "train-0.parquet", 286.5),
        ),
        hours=3.4,
        license="SberDevices Golos license (non-commercial use)",
        homepage="https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield",
    ),
    CorpusSpec(
        id="fleurs-ru-test",
        role="eval",
        title="FLEURS Russian (test)",
        description="Read Wikipedia sentences, different speakers and microphones than "
        "the training corpora: an honest false-activation benchmark.",
        language="ru",
        format="fleurs",
        files=(
            CorpusFile(f"{FLEURS}/ru_ru/audio/test.tar.gz", "test.tar.gz", 433.1),
            CorpusFile(f"{FLEURS}/ru_ru/test.tsv", "test.tsv", 0.7),
        ),
        hours=2.5,
        license="CC BY 4.0",
        homepage="https://huggingface.co/datasets/google/fleurs",
    ),
    CorpusSpec(
        id="golos-farfield-test",
        role="eval",
        title="Golos far-field (test)",
        description="Held-out far-field smart-speaker requests for the false-activation benchmark.",
        language="ru",
        format="parquet",
        files=(
            CorpusFile(f"{GOLOS_FARFIELD}/test-00000-of-00001-7fb58d4cda8ff9fe.parquet", "test.parquet", 129.5),
        ),
        hours=1.4,
        license="SberDevices Golos license (non-commercial use)",
        homepage="https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield",
    ),
)
BY_ID = {spec.id: spec for spec in CATALOG}


def get(corpus_id: str) -> CorpusSpec:
    try:
        return BY_ID[corpus_id]
    except KeyError as exc:
        raise KeyError(f"Unknown speech corpus '{corpus_id}'.") from exc


def corpus_dir(root: Path, corpus_id: str) -> Path:
    return root / corpus_id


def is_installed(root: Path, corpus_id: str) -> bool:
    return (corpus_dir(root, corpus_id) / INSTALLED_MARKER).is_file()


def installed_info(root: Path, corpus_id: str) -> dict | None:
    try:
        return json.loads((corpus_dir(root, corpus_id) / INSTALLED_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def remove(root: Path, corpus_id: str) -> None:
    folder = corpus_dir(root, corpus_id)
    if folder.exists():
        shutil.rmtree(folder)


def _to_16k_mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sample_rate != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE, res_type="soxr_hq")
    return audio


def _decode(data: bytes) -> np.ndarray:
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return _to_16k_mono(audio, int(sample_rate))


def _iter_parquet(path: Path) -> Iterator[tuple[np.ndarray, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise VoiceError("Reading this corpus needs the 'pyarrow' package.") from exc
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=64, columns=["audio", "transcription"]):
        for audio, text in zip(batch.column("audio").to_pylist(), batch.column("transcription").to_pylist()):
            if not audio or not audio.get("bytes"):
                continue
            yield _decode(audio["bytes"]), str(text or "")


def _iter_fleurs(folder: Path) -> Iterator[tuple[np.ndarray, str]]:
    tsv = next(folder.glob("*.tsv"))
    texts: dict[str, str] = {}
    with tsv.open(encoding="utf-8", newline="") as handle:
        # id, file name, raw transcription, normalized transcription, ...
        for row in csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 4:
                texts[row[1]] = row[3]
    archive = next(folder.glob("*.tar.gz"))
    with tarfile.open(archive) as bundle:
        for member in bundle:
            if not member.isfile() or not member.name.endswith(".wav"):
                continue
            handle = bundle.extractfile(member)
            if handle is None:
                continue
            yield _decode(handle.read()), texts.get(Path(member.name).name, "")


def _utterances(spec: CorpusSpec, raw: Path) -> Iterator[tuple[np.ndarray, str]]:
    if spec.format == "parquet":
        for item in spec.files:
            yield from _iter_parquet(raw / item.name)
    elif spec.format == "fleurs":
        yield from _iter_fleurs(raw)
    else:
        raise VoiceError(f"Unknown corpus format '{spec.format}'.")


ProgressFn = Callable[[str, float, str], None]


def install(root: Path, corpus_id: str, on_progress: ProgressFn) -> dict:
    """Download, convert to 16 kHz int16 and index one corpus.

    Raw downloads live in ``.raw`` until the conversion succeeds, so a
    canceled or failed run resumes instead of downloading again.
    """
    spec = get(corpus_id)
    if is_installed(root, spec.id):
        return installed_info(root, spec.id) or {"id": spec.id}
    folder = corpus_dir(root, spec.id)
    raw = folder / ".raw"
    total_mb = max(1.0, spec.size_mb)
    done_mb = 0.0
    for item in spec.files:
        def report(fraction: float, detail: str, start=done_mb, item=item) -> None:
            on_progress("download", 0.8 * (start + item.size_mb * fraction) / total_mb, f"{spec.title}: {item.name} · {detail}")

        target = raw / item.name
        if not target.is_file():
            download(item.url, target, report)
        done_mb += item.size_mb

    info = prepare(spec, folder, raw, on_progress)
    shutil.rmtree(raw, ignore_errors=True)
    return info


def prepare(spec: CorpusSpec, folder: Path, raw: Path, on_progress: ProgressFn) -> dict:
    """Convert downloaded files in ``raw`` into ``audio.pcm`` + ``index.json``."""
    folder.mkdir(parents=True, exist_ok=True)
    on_progress("prepare", 0.8, f"{spec.title}: converting audio")
    pcm_part = folder / "audio.pcm.part"
    index: list[list] = []
    offset = 0
    with pcm_part.open("wb") as handle:
        for count, (audio, text) in enumerate(_utterances(spec, raw)):
            if audio.size < SAMPLE_RATE // 4:
                continue
            pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype("<i2")
            handle.write(pcm.tobytes())
            index.append([offset, int(pcm.size), text])
            offset += int(pcm.size)
            if count % 200 == 0:
                hours = offset / SAMPLE_RATE / 3600
                on_progress("prepare", min(0.98, 0.8 + 0.18 * hours / max(spec.hours, 0.1)), f"{spec.title}: {len(index)} utterances · {hours:.1f} h")
    if not index:
        raise VoiceError(f"{spec.title}: no usable audio was found in the download.")
    pcm_part.replace(folder / "audio.pcm")
    hours = offset / SAMPLE_RATE / 3600
    (folder / "index.json").write_text(
        json.dumps({"sampleRate": SAMPLE_RATE, "utterances": index}, ensure_ascii=False), encoding="utf-8"
    )
    info = {"id": spec.id, "role": spec.role, "utterances": len(index), "hours": round(hours, 2)}
    (folder / INSTALLED_MARKER).write_text(json.dumps(info), encoding="utf-8")
    log(f"Prepared {spec.title}: {len(index)} utterances, {hours:.2f} h")
    return info


def utterance_split(corpus_name: str, index: int, val_ratio: float, test_ratio: float) -> str:
    """Stable train/val/test assignment of one utterance.

    A hash instead of a shuffle keeps every tool (dataset builder, hard
    negative mining) agreeing on the split, so mined windows can never come
    from validation or test speech.
    """
    bucket = zlib.crc32(f"{corpus_name}:{index}".encode("utf-8")) % 10_000 / 10_000
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


class Corpus:
    """Read-only, memory-mapped view of a prepared corpus folder."""

    def __init__(self, folder: Path):
        index_path = folder / "index.json"
        if not index_path.is_file():
            raise VoiceError(f"{folder} is not a prepared speech corpus (index.json missing).")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.folder = folder
        self.name = folder.name
        self.utterances: list[tuple[int, int, str]] = [(int(o), int(n), str(t)) for o, n, t in index["utterances"]]
        self._pcm = np.memmap(folder / "audio.pcm", dtype="<i2", mode="r")

    def __len__(self) -> int:
        return len(self.utterances)

    @property
    def hours(self) -> float:
        return sum(n for _, n, _ in self.utterances) / SAMPLE_RATE / 3600

    def text(self, index: int) -> str:
        return self.utterances[index][2]

    def audio(self, index: int) -> np.ndarray:
        offset, length, _ = self.utterances[index]
        return np.asarray(self._pcm[offset : offset + length], dtype=np.float32) / 32768.0
