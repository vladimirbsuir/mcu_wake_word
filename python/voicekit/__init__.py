"""Pluggable speech and language providers for WakeForge Studio.

Three categories are supported, each with exactly one active choice:

* ``stt`` - speech-to-text, used to read the spoken phrase back from takes
* ``llm`` - a language model that writes word lists (negatives and so on)
* ``tts`` - text-to-speech, used to voice those word lists and the wake phrase

A choice is either a model from the local catalog (``catalog.py``) or a
user-configured endpoint profile (``providers/openai_compat.py``). New
providers implement the small engine interfaces in ``base.py`` and register
themselves in ``registry.py``.
"""
