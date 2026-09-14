"""Validation shared by release and runtime registry metadata paths."""

from __future__ import annotations

import re
from typing import Any

VOICE_GENDERS = {"female", "male", "neutral", "unknown"}
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")
_LOCALE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{2,4})?$")


def validate_voice_metadata(
    voices: list[str], metadata: Any
) -> dict[str, dict[str, Any]]:
    """Validate complete per-voice metadata and return it unchanged."""
    if not isinstance(metadata, dict):
        raise ValueError("voice_metadata must be an object")
    if set(metadata) != set(voices):
        raise ValueError("voice_metadata must exactly cover the runtime voice roster")

    for voice, detail in metadata.items():
        if not isinstance(detail, dict):
            raise ValueError(f"{voice}: voice metadata must be an object")
        if detail.get("gender") not in VOICE_GENDERS:
            raise ValueError(f"{voice}: invalid voice gender")
        language = detail.get("language")
        if not isinstance(language, str) or _LANGUAGE_RE.fullmatch(language) is None:
            raise ValueError(f"{voice}: invalid voice language")
        locale = detail.get("locale")
        if not isinstance(locale, str) or _LOCALE_RE.fullmatch(locale) is None:
            raise ValueError(f"{voice}: invalid voice locale")
        label = detail.get("language_label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{voice}: missing voice language label")
    return metadata
