"""Gemeinsame Test-Vorbereitung."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def _camoufox_aus(monkeypatch):
    """Camoufox-Stufe (stealth.py) startet in Tests nur, wenn ein Test sie einschaltet."""
    monkeypatch.setenv("READER_STEALTH", "0")


@pytest.fixture(autouse=True)
def _bibliothek_aus(monkeypatch):
    """Themen-Bibliothek (app/library.py) nur in Tests, die sie selbst einschalten."""
    monkeypatch.setenv("LIBRARY_K", "0")


@pytest.fixture(autouse=True)
def _reranker_paar_format(monkeypatch):
    """Die bestehenden Reranker-Tests pruefen den Paar-Weg (mMiniLM). Nemotron-Tests
    in test_nemotron_format.py setzen RERANKER_FORMAT selbst."""
    monkeypatch.setenv("RERANKER_FORMAT", "pair")
