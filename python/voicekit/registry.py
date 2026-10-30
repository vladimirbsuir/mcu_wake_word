"""Turns the user's settings into engines.

Settings (written by the desktop app) look like::

    {
      "active": {
        "stt": {"provider": "local", "model": "stt-gigaam-v3-ru"},
        "tts": {"provider": "local", "model": "tts-supertonic3"},
        "llm": {"provider": "p_openrouter"}
      },
      "profiles": [
        {"id": "p_openrouter", "type": "openai", "name": "OpenRouter",
         "baseUrl": "https://openrouter.ai/api/v1",
         "models": {"llm": "qwen/qwen3.5-35b", "tts": "", "stt": ""},
         "voices": [], "languages": []}
      ]
    }

API keys are never stored in this file; the app passes them to the job as
``WAKEFORGE_API_KEY_<PROFILE ID>`` environment variables.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from . import catalog
from .base import LlmEngine, SttEngine, TtsEngine
from .common import VoiceError

CATEGORIES = ("stt", "llm", "tts")
LOCAL = "local"

# Profile types for user-configured endpoints. Each type lists the categories
# it can serve; the profile itself decides which ones by setting a model.
PROFILE_TYPES = {
    "openai": {
        "title": "OpenAI-compatible API",
        "description": (
            "Any server that speaks the OpenAI API: OpenAI, OpenRouter, Groq, Together, "
            "or local Ollama / LM Studio / llama.cpp servers."
        ),
        "categories": ["llm", "tts", "stt"],
    },
}


@dataclass(frozen=True)
class Selection:
    category: str
    provider: str  # "local" or a profile id
    model: catalog.ModelSpec | None = None
    profile: dict | None = None

    @property
    def label(self) -> str:
        if self.model:
            return self.model.title
        assert self.profile is not None
        model = (self.profile.get("models") or {}).get(self.category, "")
        return f"{self.profile.get('name', 'Endpoint')} · {model}"


def api_key_env(profile_id: str) -> str:
    return "WAKEFORGE_API_KEY_" + re.sub(r"[^A-Z0-9]", "_", profile_id.upper())


def resolve(category: str, settings: dict, choice: dict | None = None) -> Selection:
    """The active provider of a category, or the explicit ``choice`` when given."""
    if choice is None:
        choice = (settings.get("active") or {}).get(category) or {}
    provider = choice.get("provider")
    if not provider:
        raise VoiceError(f"No {category.upper()} provider is selected. Choose one in AI models.")
    if provider == LOCAL:
        model_id = choice.get("model") or ""
        try:
            spec = catalog.get(model_id)
        except KeyError as exc:
            raise VoiceError(str(exc)) from exc
        if spec.category != category:
            raise VoiceError(f"{spec.title} is not a {category.upper()} model.")
        return Selection(category, LOCAL, model=spec)
    for profile in settings.get("profiles") or []:
        if profile.get("id") == provider:
            if not (profile.get("models") or {}).get(category):
                raise VoiceError(
                    f"Endpoint '{profile.get('name')}' has no {category.upper()} model configured."
                )
            return Selection(category, provider, profile=profile)
    raise VoiceError(f"The selected {category.upper()} endpoint no longer exists.")


def supports_language(selection: Selection, language: str) -> bool:
    if selection.model is not None:
        languages = selection.model.languages
        return languages is None or language in languages
    declared = (selection.profile or {}).get("languages") or []
    return not declared or language in declared


def _ensure_local_ready(spec: catalog.ModelSpec, models_root: Path) -> None:
    for model_id in (*spec.depends_on, spec.id):
        if not catalog.is_installed(models_root, model_id):
            raise VoiceError(f"{catalog.get(model_id).title} is not downloaded yet. Install it in AI models.")
    import importlib.util

    for requirement in spec.requirements:
        if importlib.util.find_spec(requirement.module) is None:
            raise VoiceError(
                f"{spec.title} needs the Python package '{requirement.pip[0]}'. "
                "Install it in AI models."
            )


def _profile_client(selection: Selection):
    from .providers.openai_compat import OpenAICompatClient

    profile = selection.profile or {}
    return OpenAICompatClient(
        base_url=profile.get("baseUrl", ""),
        api_key=os.environ.get(api_key_env(profile.get("id", ""))),
        name=profile.get("name", "endpoint"),
    )


def create_stt(settings: dict, models_root: Path, language: str) -> SttEngine:
    selection = resolve("stt", settings)
    if selection.model is None:
        from .providers.openai_compat import OpenAICompatStt

        model = selection.profile["models"]["stt"]
        return OpenAICompatStt(_profile_client(selection), model, language)
    spec = selection.model
    _ensure_local_ready(spec, models_root)
    folder = catalog.model_dir(models_root, spec.id)
    from .providers import sherpa

    if spec.engine == "sherpa-transducer":
        return sherpa.TransducerStt(folder, spec.params.get("model_type", "transducer"))
    if spec.engine == "sherpa-whisper":
        return sherpa.WhisperStt(folder, language)
    raise VoiceError(f"Unsupported STT engine '{spec.engine}'.")


def engine_key(selection: Selection) -> str:
    """Folder-safe id of a provider and its model, e.g. ``local-tts-supertonic3``.

    Endpoint keys include the model name, so switching an endpoint to another
    model never reuses clips of the previous one.
    """
    if selection.model is not None:
        raw = f"local-{selection.model.id}"
    else:
        raw = f"{selection.provider}-{(selection.profile or {}).get('models', {}).get(selection.category, '')}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:80]


def create_tts(settings: dict, models_root: Path, language: str, choice: dict | None = None) -> TtsEngine:
    selection = resolve("tts", settings, choice)
    if selection.model is None:
        from .providers.openai_compat import OpenAICompatTts

        profile = selection.profile
        return OpenAICompatTts(
            _profile_client(selection), profile["models"]["tts"], profile.get("voices") or []
        )
    spec = selection.model
    _ensure_local_ready(spec, models_root)
    folder = catalog.model_dir(models_root, spec.id)
    if spec.engine == "sherpa-supertonic":
        from .providers import sherpa

        return sherpa.SupertonicTts(folder, language, spec.voices)
    if spec.engine == "piper":
        from .providers.piper_engine import PiperTts

        return PiperTts(folder, spec.voices)
    if spec.engine == "silero":
        from .providers.silero import SileroTts

        return SileroTts(folder, spec.params.get("sample_rate", 48000))
    raise VoiceError(f"Unsupported TTS engine '{spec.engine}'.")


def create_llm(settings: dict, models_root: Path) -> LlmEngine:
    selection = resolve("llm", settings)
    if selection.model is None:
        from .providers.openai_compat import OpenAICompatLlm

        return OpenAICompatLlm(_profile_client(selection), selection.profile["models"]["llm"])
    spec = selection.model
    _ensure_local_ready(spec, models_root)
    if spec.engine == "llamacpp":
        from .providers.llamacpp import LlamaCppLlm

        return LlamaCppLlm(
            runtime_dir=catalog.model_dir(models_root, "runtime-llamacpp"),
            gguf=catalog.model_dir(models_root, spec.id) / spec.params["gguf"],
            disable_thinking=bool(spec.params.get("disable_thinking")),
        )
    raise VoiceError(f"Unsupported LLM engine '{spec.engine}'.")
