from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_test"))
from frontend import thorsten_cleanup

kokorog2p = pytest.importorskip("kokorog2p")


FIXTURES = Path(__file__).parent / "fixtures"
RUSSIAN_GOLDEN_FIXTURES = json.loads(
    (FIXTURES / "russian_zaakirio_frontend.json").read_text(encoding="utf-8")
)
KAZAKH_GOLDEN_FIXTURES = json.loads(
    (FIXTURES / "kazakh_anuarsv_frontend.json").read_text(encoding="utf-8")
)


def _phonemize_or_skip(text: str, language: str) -> str:
    try:
        result = kokorog2p.phonemize(text, language=language, return_ids=False)
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"kokorog2p has no native {language} frontend: {exc}")
    return result.phonemes


SWEDISH_SENTENCE = "Hej. Det här är ett lokalt rösttest före publiceringen av modellen."


def test_swedish_frontend_symbols_fit_stock_kokoro_vocabulary() -> None:
    result = kokorog2p.phonemize(SWEDISH_SENTENCE, language="sv", return_ids=False)
    phonemes = result.phonemes
    vocabulary = kokorog2p.get_kokoro_vocab()
    assert not sorted(set(phonemes) - set(vocabulary))
    assert any(symbol in phonemes for symbol in ("ɕ", "ø", "œ", "ɛ"))


def test_thorsten_short_u_cleanup_removes_unsupported_symbol() -> None:
    result = kokorog2p.phonemize("Brücke", language="de", return_ids=False)
    cleaned = thorsten_cleanup(result.phonemes)
    assert "ʏ" not in cleaned
    assert thorsten_cleanup("bʏkə") == "bykə"


@pytest.mark.parametrize("fixture", RUSSIAN_GOLDEN_FIXTURES)
def test_russian_frontend_matches_zaakirio_reference(fixture: dict[str, str]) -> None:
    phonemes = _phonemize_or_skip(fixture["text"], "ru")
    assert phonemes == fixture["expected"]


@pytest.mark.parametrize("fixture", KAZAKH_GOLDEN_FIXTURES)
def test_kazakh_frontend_matches_misaki_reference(fixture: dict[str, str]) -> None:
    phonemes = _phonemize_or_skip(fixture["text"], "kk")
    assert phonemes == fixture["expected"]


def test_kazakh_frontend_symbols_fit_model_vocabulary() -> None:
    for fixture in KAZAKH_GOLDEN_FIXTURES:
        phonemes = _phonemize_or_skip(fixture["text"], "kk")
        vocabulary = kokorog2p.get_kokoro_vocab()
        assert not sorted(set(phonemes) - set(vocabulary))
