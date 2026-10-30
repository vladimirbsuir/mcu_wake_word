#!/usr/bin/env python3
"""Speech/LLM helper jobs for WakeForge Studio.

Subcommands (all print ``@@PROGRESS@@``/``@@RESULT@@`` lines like the other
pipeline scripts):

  status      catalog, install state and Python packages (fast, no model loads)
  install     download a catalog model and its dependencies
  deps        pip-install the Python packages a model needs
  transcribe  read the spoken phrase back from a project's takes (STT)
  words       write word lists for the wake phrase (LLM)
  synthesize  voice the word lists and the wake phrase (TTS)
  check       quick end-to-end test of the selected provider of one category
  corpus      download and prepare a speech corpus (negatives or benchmark)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from voicekit import catalog, corpora, groups, registry, tasks
from voicekit.common import (
    VoiceError,
    configure_stdio,
    download,
    emit_progress,
    emit_result,
    extract,
    log,
    read_json,
)
from voicekit.languages import languages_payload


def requirement_status(spec: catalog.ModelSpec) -> list[dict]:
    return [
        {
            "package": requirement.pip[0],
            "module": requirement.module,
            "installed": importlib.util.find_spec(requirement.module) is not None,
            "sizeMb": requirement.size_mb,
        }
        for requirement in spec.requirements
    ]


def cmd_status(args: argparse.Namespace) -> dict:
    root = Path(args.models_dir)
    models = []
    for spec in catalog.CATALOG:
        models.append({
            "id": spec.id,
            "category": spec.category,
            "engine": spec.engine,
            "title": spec.title,
            "description": spec.description,
            "languages": list(spec.languages) if spec.languages is not None else None,
            "recommendedFor": list(spec.recommended_for),
            "sizeMb": round(spec.size_mb),
            "ramGb": spec.ram_gb,
            "license": spec.license,
            "voices": len(spec.voices),
            "speakers": sum(int(voice.get("speakers", 1)) for voice in spec.voices),
            "dependsOn": list(spec.depends_on),
            "installed": catalog.is_installed(root, spec.id),
            "requirements": requirement_status(spec),
        })
    return {
        "models": models,
        "profileTypes": registry.PROFILE_TYPES,
        "groups": groups.groups_payload(),
        "languages": languages_payload(),
        "corpora": corpus_status(Path(args.datasets_dir)) if args.datasets_dir else [],
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "modelsDir": str(root),
    }


def corpus_status(root: Path) -> list[dict]:
    items = []
    for spec in corpora.CATALOG:
        info = corpora.installed_info(root, spec.id)
        items.append({
            "id": spec.id,
            "role": spec.role,
            "title": spec.title,
            "description": spec.description,
            "language": spec.language,
            "hours": spec.hours,
            "sizeMb": round(spec.size_mb),
            "license": spec.license,
            "homepage": spec.homepage,
            "installed": info is not None,
            "utterances": (info or {}).get("utterances"),
            "installedHours": (info or {}).get("hours"),
        })
    return items


def cmd_corpus(args: argparse.Namespace) -> dict:
    if importlib.util.find_spec("pyarrow") is None and corpora.get(args.corpus).format == "parquet":
        emit_progress("deps", 0.0, "Installing pyarrow")
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "pyarrow"]
        if subprocess.call(command) != 0:
            raise VoiceError("pip could not install pyarrow, which reads this corpus.")
    return corpora.install(Path(args.datasets_dir), args.corpus, emit_progress)


def _install_one(root: Path, spec: catalog.ModelSpec, base: float, span: float) -> None:
    folder = catalog.model_dir(root, spec.id)
    folder.mkdir(parents=True, exist_ok=True)
    total = max(1.0, spec.size_mb)
    done = 0.0
    for artifact in spec.artifacts:
        share = max(artifact.size_mb, 0.01) / total
        name = artifact.url.rsplit("/", 1)[-1]

        def report(fraction: float, detail: str, start=done, share=share, name=name) -> None:
            emit_progress("download", base + span * (start + share * fraction), f"{spec.title}: {name} · {detail}")

        if artifact.kind == "file":
            download(artifact.url, folder / artifact.target, report)
        else:
            archive = folder / f".download-{name}"
            download(artifact.url, archive, report)
            emit_progress("extract", base + span * (done + share), f"{spec.title}: extracting {name}")
            extract(archive, folder / artifact.target)
            archive.unlink(missing_ok=True)
        done += share
    catalog.mark_installed(root, spec)
    log(f"Installed {spec.title} into {folder}")


def cmd_install(args: argparse.Namespace) -> dict:
    root = Path(args.models_dir)
    spec = catalog.get(args.model)
    chain = [catalog.get(dep) for dep in spec.depends_on if not catalog.is_installed(root, dep)]
    chain.append(spec)
    total = sum(max(item.size_mb, 1) for item in chain)
    base = 0.0
    for item in chain:
        span = max(item.size_mb, 1) / total
        _install_one(root, item, base, span)
        base += span
    missing = [req for req in requirement_status(spec) if not req["installed"]]
    return {"model": spec.id, "installed": True, "missingPackages": [req["package"] for req in missing]}


def cmd_deps(args: argparse.Namespace) -> dict:
    spec = catalog.get(args.model)
    installed = []
    for index, requirement in enumerate(spec.requirements):
        if importlib.util.find_spec(requirement.module) is not None:
            continue
        emit_progress("deps", index / max(1, len(spec.requirements)), f"Installing {requirement.pip[0]}")
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirement.pip]
        log(" ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"[pip] {line}", flush=True)
        if process.wait() != 0:
            raise VoiceError(f"pip could not install {requirement.pip[0]}. See the log above.")
        installed.append(requirement.pip[0])
    return {"model": spec.id, "installedPackages": installed}


def _settings(args: argparse.Namespace) -> dict:
    settings = read_json(Path(args.settings), {}) if args.settings else {}
    return settings if isinstance(settings, dict) else {}


def _require_language(selection: registry.Selection, language: str) -> None:
    if not registry.supports_language(selection, language):
        raise VoiceError(f"{selection.label} does not support this language ({language}).")


def cmd_transcribe(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    _require_language(registry.resolve("stt", settings), args.language)
    emit_progress("load", 0.02, "Loading speech recognition model")
    with registry.create_stt(settings, Path(args.models_dir), args.language) as stt:
        return tasks.transcribe_takes(stt, Path(args.recordings))


def cmd_words(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    voice = read_json(Path(args.voice_file), {})
    phrase = (voice.get("spokenText") or "").strip()
    language = voice.get("language") or ""
    if not phrase or not language:
        raise VoiceError("Set the spoken phrase and its language first.")
    requests = {key: int(value) for key, value in json.loads(args.groups).items() if int(value) > 0}
    if not requests:
        raise VoiceError("Choose at least one word group.")
    selection = registry.resolve("llm", settings)
    emit_progress("load", 0.03, f"Starting {selection.label}")
    existing = {key: list(value.get("items") or []) for key, value in (voice.get("groups") or {}).items()}

    def on_group(index: int, group: groups.WordGroup, done: int, total: int) -> None:
        fraction = (index + min(1.0, done / max(1, total))) / len(requests)
        emit_progress("words", 0.1 + 0.85 * fraction, f"Writing “{group.title}” · {done}/{total}")

    with registry.create_llm(settings, Path(args.models_dir)) as llm:
        result = groups.generate(llm, phrase, language, requests, existing, on_group)
    emit_progress("words", 1.0, "Word lists ready")
    return {"groups": result, "provider": selection.label}


def tts_choices(voice: dict, settings: dict) -> list[dict]:
    """TTS engines chosen for this project, or the active TTS when none were chosen."""
    chosen = [item for item in ((voice.get("synthesis") or {}).get("engines") or []) if item.get("provider")]
    if chosen:
        return chosen
    active = (settings.get("active") or {}).get("tts")
    if not active:
        raise VoiceError("No TTS is selected. Choose one or more in the Phrases step or in AI models.")
    return [active]


def cmd_synthesize(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    voice = read_json(Path(args.voice_file), {})
    language = voice.get("language") or ""
    selections = [registry.resolve("tts", settings, choice) for choice in tts_choices(voice, settings)]
    for selection in selections:
        _require_language(selection, language)
    output = Path(args.output)
    keys = [registry.engine_key(selection) for selection in selections]
    results = []
    for position, selection in enumerate(selections):
        share = 1.0 / len(selections)

        def progress(fraction: float, message: str, position=position) -> None:
            emit_progress("synthesize", 0.02 + 0.96 * share * (position + fraction), message)

        key = registry.engine_key(selection)
        cached = tasks.cached_engine(
            key, selection.label, voice, output, args.positives, args.voices_per_negative, keys
        )
        if cached is not None:
            log(f"{selection.label}: all {cached['wanted']} clips already made")
            results.append(cached)
            continue
        progress(0.0, f"Loading {selection.label}")
        with registry.create_tts(settings, Path(args.models_dir), language, {
            "provider": selection.provider,
            "model": selection.model.id if selection.model else None,
        }) as tts:
            results.append(tasks.synthesize_engine(
                tts,
                key,
                selection.label,
                voice,
                output,
                include_positives=args.positives,
                voices_per_negative=args.voices_per_negative,
                progress=progress,
                engine_keys=keys,
            ))
    summary = tasks.write_synthetic_manifest(output, voice, results)
    made = sum(item["synthesized"] for item in results)
    reused = sum(item["reused"] for item in results)
    emit_progress("synthesize", 1.0, f"{made} new clips, {reused} reused")
    return summary


def cmd_check(args: argparse.Namespace) -> dict:
    settings = _settings(args)
    root = Path(args.models_dir)
    selection = registry.resolve(args.category, settings)
    emit_progress("check", 0.1, f"Testing {selection.label}")
    if args.category == "llm":
        with registry.create_llm(settings, root) as llm:
            answer = llm.chat(
                [{"role": "user", "content": 'Reply with JSON {"items": ["ok"]}.'}],
                schema=groups.SCHEMA,
                temperature=0.0,
            )
        return {"category": "llm", "provider": selection.label, "sample": answer[:200]}
    language = args.language or "en"
    if args.category == "tts":
        with registry.create_tts(settings, root, language) as tts:
            voices = tts.voices()
            audio, sample_rate = tts.synthesize("test" if language == "en" else "тест" if language == "ru" else "ok", voices[0].id)
        return {"category": "tts", "provider": selection.label, "voices": len(voices), "seconds": round(len(audio) / sample_rate, 2)}
    with registry.create_stt(settings, root, language):
        pass
    return {"category": "stt", "provider": selection.label, "loaded": True}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WakeForge speech and language helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser, settings: bool = True) -> None:
        command.add_argument("--models-dir", required=True)
        if settings:
            command.add_argument("--settings", default=None)

    status = sub.add_parser("status")
    common(status, settings=False)
    status.add_argument("--datasets-dir", default=None)
    corpus = sub.add_parser("corpus")
    common(corpus, settings=False)
    corpus.add_argument("--datasets-dir", required=True)
    corpus.add_argument("--corpus", required=True)
    install = sub.add_parser("install")
    common(install, settings=False)
    install.add_argument("--model", required=True)
    deps = sub.add_parser("deps")
    common(deps, settings=False)
    deps.add_argument("--model", required=True)
    transcribe = sub.add_parser("transcribe")
    common(transcribe)
    transcribe.add_argument("--recordings", required=True)
    transcribe.add_argument("--language", required=True)
    words = sub.add_parser("words")
    common(words)
    words.add_argument("--voice-file", required=True)
    words.add_argument("--groups", required=True, help='JSON object, e.g. {"confusable": 30}')
    synth = sub.add_parser("synthesize")
    common(synth)
    synth.add_argument("--voice-file", required=True)
    synth.add_argument("--output", required=True)
    synth.add_argument("--voices-per-negative", type=int, default=2)
    synth.add_argument("--positives", action=argparse.BooleanOptionalAction, default=True)
    check = sub.add_parser("check")
    common(check)
    check.add_argument("--category", choices=registry.CATEGORIES, required=True)
    check.add_argument("--language", default=None)
    return parser.parse_args(argv)


COMMANDS = {
    "status": cmd_status,
    "install": cmd_install,
    "deps": cmd_deps,
    "transcribe": cmd_transcribe,
    "words": cmd_words,
    "synthesize": cmd_synthesize,
    "check": cmd_check,
    "corpus": cmd_corpus,
}


def main() -> int:
    configure_stdio()
    args = parse_args()
    try:
        result = COMMANDS[args.command](args)
    except (VoiceError, KeyError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        print(f"[voice] ERROR: {message}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # pragma: no cover - unexpected crash, keep the traceback
        import traceback

        print(f"[voice] ERROR: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    emit_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
