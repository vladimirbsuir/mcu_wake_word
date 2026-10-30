"""Built-in catalog of downloadable local models.

Every entry says what it is for (category), which engine runs it, which
languages it can handle (``None`` means effectively any language), what has to
be downloaded and which Python packages it needs. Adding a model is a matter
of adding an entry here; adding a new *kind* of model means adding an engine
in ``providers/`` and mapping it in ``registry.py``.
"""

from __future__ import annotations

import json
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

LLAMA_CPP_BUILD = "b12002"
SHERPA_ASR = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
SHERPA_TTS = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
PIPER_VOICES = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
INSTALLED_MARKER = ".installed.json"

PARAKEET_V3_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
)
SUPERTONIC_3_LANGUAGES = (
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr",
    "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk",
    "sl", "sv", "tr", "uk", "vi",
)


@dataclass(frozen=True)
class Artifact:
    url: str
    # "file" keeps the download as ``target``; archives are extracted into ``target``.
    kind: str
    target: str = "."
    size_mb: float = 0.0


@dataclass(frozen=True)
class Requirement:
    """A Python package an engine imports, plus how to install it."""

    module: str
    pip: tuple[str, ...]
    size_mb: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    category: str  # "stt" | "tts" | "llm" | "runtime"
    engine: str
    title: str
    description: str
    languages: tuple[str, ...] | None
    artifacts: tuple[Artifact, ...]
    license: str
    ram_gb: float
    requirements: tuple[Requirement, ...] = ()
    depends_on: tuple[str, ...] = ()
    voices: tuple[dict, ...] = ()
    params: dict = field(default_factory=dict)
    recommended_for: tuple[str, ...] = ()

    @property
    def size_mb(self) -> float:
        return sum(artifact.size_mb for artifact in self.artifacts)


SHERPA = Requirement("sherpa_onnx", ("sherpa-onnx==2.0.1",), 60)
PIPER = Requirement("piper", ("piper-tts==1.8.0",), 25)
TORCH_CPU = Requirement(
    "torch",
    ("torch", "--index-url", "https://download.pytorch.org/whl/cpu"),
    250,
)


def _llama_cpp_artifact() -> Artifact:
    system = platform.system()
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        name = f"llama-{LLAMA_CPP_BUILD}-bin-win-vulkan-{'arm64' if arm else 'x64'}.zip"
        size = 24 if arm else 35
    elif system == "Darwin":
        name = f"llama-{LLAMA_CPP_BUILD}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"
        size = 15
    else:
        name = f"llama-{LLAMA_CPP_BUILD}-bin-ubuntu-vulkan-{'arm64' if arm else 'x64'}.tar.gz"
        size = 33
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}/{name}"
    return Artifact(url, "archive", ".", size)


def _piper_voice(language: str, locale: str, name: str, quality: str, size_mb: float) -> tuple[Artifact, ...]:
    stem = f"{locale}-{name}-{quality}"
    base = f"{PIPER_VOICES}/{language}/{locale}/{name}/{quality}/{stem}"
    return (
        Artifact(f"{base}.onnx", "file", f"{stem}.onnx", size_mb),
        Artifact(f"{base}.onnx.json", "file", f"{stem}.onnx.json", 0.02),
    )


def _gguf(repo: str, filename: str, size_mb: float) -> tuple[Artifact, ...]:
    return (Artifact(f"https://huggingface.co/{repo}/resolve/main/{filename}", "file", filename, size_mb),)


CATALOG: tuple[ModelSpec, ...] = (
    # ----------------------------------------------------------------- STT
    ModelSpec(
        id="stt-gigaam-v3-ru",
        category="stt",
        engine="sherpa-transducer",
        title="GigaAM v3 (Russian)",
        description="Sber's Russian ASR, RNN-T. Best accuracy for Russian, fast on CPU.",
        languages=("ru",),
        artifacts=(
            Artifact(f"{SHERPA_ASR}/sherpa-onnx-nemo-transducer-giga-am-v3-russian-2025-12-16.tar.bz2", "archive", ".", 159),
        ),
        license="MIT",
        ram_gb=1.0,
        requirements=(SHERPA,),
        params={"model_type": "nemo_transducer"},
        recommended_for=("ru",),
    ),
    ModelSpec(
        id="stt-parakeet-v3",
        category="stt",
        engine="sherpa-transducer",
        title="Parakeet TDT 0.6B v3",
        description="NVIDIA multilingual ASR for 25 European languages, int8 on CPU.",
        languages=PARAKEET_V3_LANGUAGES,
        artifacts=(
            Artifact(f"{SHERPA_ASR}/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2", "archive", ".", 464),
        ),
        license="CC-BY-4.0",
        ram_gb=1.5,
        requirements=(SHERPA,),
        params={"model_type": "nemo_transducer"},
        recommended_for=tuple(code for code in PARAKEET_V3_LANGUAGES if code != "ru"),
    ),
    ModelSpec(
        id="stt-whisper-turbo",
        category="stt",
        engine="sherpa-whisper",
        title="Whisper large-v3 turbo",
        description="OpenAI Whisper turbo, ~99 languages. Universal fallback, slower on CPU.",
        languages=None,
        artifacts=(Artifact(f"{SHERPA_ASR}/sherpa-onnx-whisper-turbo.tar.bz2", "archive", ".", 537),),
        license="MIT",
        ram_gb=2.0,
        requirements=(SHERPA,),
    ),
    # ----------------------------------------------------------------- TTS
    ModelSpec(
        id="tts-supertonic3",
        category="tts",
        engine="sherpa-supertonic",
        title="Supertonic 3",
        description="31 languages, 10 voices, very fast on CPU (44.1 kHz).",
        languages=SUPERTONIC_3_LANGUAGES,
        artifacts=(Artifact(f"{SHERPA_TTS}/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2", "archive", ".", 122),),
        license="OpenRAIL-M (weights), MIT (code)",
        ram_gb=1.0,
        requirements=(SHERPA,),
        voices=tuple(
            {"id": str(index), "name": f"{'M' if index < 5 else 'F'}{index % 5 + 1}"}
            for index in range(10)
        ),
        recommended_for=SUPERTONIC_3_LANGUAGES,
    ),
    ModelSpec(
        id="tts-silero-ru",
        category="tts",
        engine="silero",
        title="Silero v5 (Russian)",
        description="High-quality Russian voices with stress handling. Needs PyTorch (CPU).",
        languages=("ru",),
        artifacts=(Artifact("https://models.silero.ai/models/tts/ru/v5_ru.pt", "file", "v5_ru.pt", 139),),
        license="CC BY-NC-SA 4.0 (non-commercial)",
        ram_gb=1.5,
        requirements=(TORCH_CPU,),
        voices=tuple({"id": name, "name": name} for name in ("aidar", "baya", "kseniya", "xenia", "eugene")),
        params={"sample_rate": 48000},
        recommended_for=("ru",),
    ),
    ModelSpec(
        id="tts-piper-ru",
        category="tts",
        engine="piper",
        title="Piper voices (Russian)",
        description="Four classic Russian Piper voices: Denis, Dmitri, Irina, Ruslan.",
        languages=("ru",),
        artifacts=sum(
            (_piper_voice("ru", "ru_RU", name, "medium", 60) for name in ("denis", "dmitri", "irina", "ruslan")),
            (),
        ),
        license="Voice-specific (see model cards)",
        ram_gb=0.5,
        requirements=(PIPER,),
        voices=tuple(
            {"id": f"ru_RU-{name}-medium", "name": name.capitalize()}
            for name in ("denis", "dmitri", "irina", "ruslan")
        ),
    ),
    ModelSpec(
        id="tts-piper-en",
        category="tts",
        engine="piper",
        title="Piper LibriTTS-R (English)",
        description="One Piper model with 900+ English speakers: maximum speaker variety.",
        languages=("en",),
        artifacts=_piper_voice("en", "en_US", "libritts_r", "medium", 75),
        license="CC-BY-4.0",
        ram_gb=0.5,
        requirements=(PIPER,),
        voices=({"id": "en_US-libritts_r-medium", "name": "LibriTTS-R", "speakers": 904},),
        recommended_for=("en",),
    ),
    # ----------------------------------------------------------------- LLM
    ModelSpec(
        id="runtime-llamacpp",
        category="runtime",
        engine="llamacpp-runtime",
        title=f"llama.cpp server ({LLAMA_CPP_BUILD})",
        description="Runs local GGUF language models. Vulkan build: uses the GPU when available.",
        languages=None,
        artifacts=(_llama_cpp_artifact(),),
        license="MIT",
        ram_gb=0.0,
    ),
    ModelSpec(
        id="llm-qwen35-4b",
        category="llm",
        engine="llamacpp",
        title="Qwen3.5 4B (Q4_K_M)",
        description="Best quality for its size, 200+ languages. ~4 GB RAM while running.",
        languages=None,
        artifacts=_gguf("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2614),
        license="Apache-2.0",
        ram_gb=4.0,
        depends_on=("runtime-llamacpp",),
        params={"gguf": "Qwen3.5-4B-Q4_K_M.gguf", "disable_thinking": True},
        recommended_for=("*",),
    ),
    ModelSpec(
        id="llm-qwen35-2b",
        category="llm",
        engine="llamacpp",
        title="Qwen3.5 2B (Q4_K_M)",
        description="Lighter and faster, noticeably weaker word lists. ~2 GB RAM.",
        languages=None,
        artifacts=_gguf("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1222),
        license="Apache-2.0",
        ram_gb=2.0,
        depends_on=("runtime-llamacpp",),
        params={"gguf": "Qwen3.5-2B-Q4_K_M.gguf", "disable_thinking": True},
    ),
    ModelSpec(
        id="llm-gemma4-e4b",
        category="llm",
        engine="llamacpp",
        title="Gemma 4 E4B (Q4_K_M)",
        description="Google's 140+ language model. Larger download, ~6 GB RAM.",
        languages=None,
        artifacts=_gguf("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf", 4747),
        license="Apache-2.0",
        ram_gb=6.0,
        depends_on=("runtime-llamacpp",),
        params={"gguf": "gemma-4-E4B-it-Q4_K_M.gguf"},
    ),
)

BY_ID = {spec.id: spec for spec in CATALOG}


def get(model_id: str) -> ModelSpec:
    try:
        return BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model '{model_id}'.") from exc


def model_dir(models_root: Path, model_id: str) -> Path:
    return models_root / model_id


def is_installed(models_root: Path, model_id: str) -> bool:
    return (model_dir(models_root, model_id) / INSTALLED_MARKER).is_file()


def mark_installed(models_root: Path, spec: ModelSpec) -> None:
    payload = {"id": spec.id, "artifacts": [artifact.url for artifact in spec.artifacts]}
    (model_dir(models_root, spec.id) / INSTALLED_MARKER).write_text(json.dumps(payload), encoding="utf-8")


def remove(models_root: Path, model_id: str) -> None:
    folder = model_dir(models_root, model_id)
    if folder.exists():
        shutil.rmtree(folder)
