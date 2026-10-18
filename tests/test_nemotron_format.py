"""Nemotron-Eingabeformat und Lade-Schutz, ohne Modell."""

import pytest

from app import config, rerank


def test_prompt_folgt_nvidias_modellkarte():
    assert rerank.nemotron_prompt("wer?", "Text") == "question:wer? \n \n passage:Text"


def test_standard_in_diesem_repo_ist_nemotron(monkeypatch):
    monkeypatch.delenv("RERANKER_FORMAT", raising=False)
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    s = config.get_settings()
    assert s.reranker_format == "nemotron"
    assert s.reranker_model == rerank.NEMOTRON_BASE


def test_fremdes_hub_modell_mit_fremdem_code_wird_abgelehnt(monkeypatch):
    monkeypatch.setenv("RERANKER_FORMAT", "nemotron")
    monkeypatch.setenv("RERANKER_MODEL", "jemand/boeses-modell")
    pytest.importorskip("transformers")
    with pytest.raises(ValueError, match="nur nvidia/llama-nemotron-rerank-1b-v2"):
        rerank._load_nemotron(config.get_settings(), "cpu")


def test_nemotron_tokenisiert_nur_unter_der_sperre_und_mit_prompt(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("RERANKER_FORMAT", "nemotron")
    seen = []

    class Tok:
        def __call__(self, texts, **k):
            assert rerank._INFER_LOCK.locked(), "Tokenizer ausserhalb der Sperre"
            assert k["max_length"] == 256 + rerank.NEMOTRON_EXTRA_TOKENS
            seen.extend(texts)

            class Enc(dict):
                def to(self, _d):
                    return self

            return Enc(n=len(texts))

    class Model:
        def __call__(self, n):
            return type("O", (), {"logits": torch.arange(n, dtype=torch.float32).view(n, 1)})()

    monkeypatch.setattr(rerank, "_get_reranker", lambda: (Tok(), Model(), "cpu"))
    assert rerank._cross_scores("frage", ["a", "b"], 256) == [0.0, 1.0]
    assert seen == [rerank.nemotron_prompt("frage", "a"), rerank.nemotron_prompt("frage", "b")]
