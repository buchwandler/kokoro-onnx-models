from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "local_test"))
from frontend import thorsten_cleanup

kokorog2p = pytest.importorskip("kokorog2p")


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
