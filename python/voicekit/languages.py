"""Language codes, display names and simple script checks."""

from __future__ import annotations

import unicodedata

# ISO 639-1 code -> (English name, native name, script)
LANGUAGES: dict[str, tuple[str, str, str]] = {
    "ar": ("Arabic", "العربية", "ARABIC"),
    "bg": ("Bulgarian", "Български", "CYRILLIC"),
    "cs": ("Czech", "Čeština", "LATIN"),
    "da": ("Danish", "Dansk", "LATIN"),
    "de": ("German", "Deutsch", "LATIN"),
    "el": ("Greek", "Ελληνικά", "GREEK"),
    "en": ("English", "English", "LATIN"),
    "es": ("Spanish", "Español", "LATIN"),
    "et": ("Estonian", "Eesti", "LATIN"),
    "fi": ("Finnish", "Suomi", "LATIN"),
    "fr": ("French", "Français", "LATIN"),
    "hi": ("Hindi", "हिन्दी", "DEVANAGARI"),
    "hr": ("Croatian", "Hrvatski", "LATIN"),
    "hu": ("Hungarian", "Magyar", "LATIN"),
    "id": ("Indonesian", "Bahasa Indonesia", "LATIN"),
    "it": ("Italian", "Italiano", "LATIN"),
    "ja": ("Japanese", "日本語", ""),
    "kk": ("Kazakh", "Қазақша", "CYRILLIC"),
    "ko": ("Korean", "한국어", "HANGUL"),
    "lt": ("Lithuanian", "Lietuvių", "LATIN"),
    "lv": ("Latvian", "Latviešu", "LATIN"),
    "mt": ("Maltese", "Malti", "LATIN"),
    "nl": ("Dutch", "Nederlands", "LATIN"),
    "pl": ("Polish", "Polski", "LATIN"),
    "pt": ("Portuguese", "Português", "LATIN"),
    "ro": ("Romanian", "Română", "LATIN"),
    "ru": ("Russian", "Русский", "CYRILLIC"),
    "sk": ("Slovak", "Slovenčina", "LATIN"),
    "sl": ("Slovenian", "Slovenščina", "LATIN"),
    "sv": ("Swedish", "Svenska", "LATIN"),
    "tr": ("Turkish", "Türkçe", "LATIN"),
    "uk": ("Ukrainian", "Українська", "CYRILLIC"),
    "vi": ("Vietnamese", "Tiếng Việt", "LATIN"),
    "zh": ("Chinese", "中文", ""),
}


def language_name(code: str) -> str:
    entry = LANGUAGES.get(code)
    return entry[0] if entry else code


def languages_payload() -> list[dict]:
    return [
        {"code": code, "name": name, "native": native}
        for code, (name, native, _) in sorted(LANGUAGES.items(), key=lambda item: item[1][0])
    ]


def matches_script(text: str, language: str) -> bool:
    """True when most letters of ``text`` use the script expected for ``language``.

    Used to drop LLM answers that come back transliterated or in another
    language. Languages without a single expected script are not checked.
    """
    entry = LANGUAGES.get(language)
    if not entry or not entry[2]:
        return True
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    expected = entry[2]
    hits = sum(1 for ch in letters if unicodedata.name(ch, "").startswith(expected))
    return hits / len(letters) >= 0.8
