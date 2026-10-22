"""Pivot: private Quelle (Intranet-Ordner) plus Tavily, ohne Netz und ohne Reranker-Modell."""

from __future__ import annotations

import httpx

from app import private, tavily_search
from app import research as r

DEMO = private.DEFAULT_DIR


def test_demo_intranet_ist_fiktiv_markiert_und_lesbar():
    docs = private.load(DEMO)
    assert 6 <= len(docs) <= 11
    assert all("FIKTIV" in d["full_text"] for d in docs)
    html = next(d for d in docs if d["url"].endswith(".html"))
    assert "<p>" not in html["full_text"] and "Auftragsverarbeitungsvertrag" in html["full_text"]
    assert all(d["url"].startswith("privat://") and d["origin"] == "privat" for d in docs)


def test_private_suche_findet_passende_dokumente():
    got = private.search(["KI-Richtlinie Schulung", "Passwort 90 Tage"], limit=3, root=DEMO)
    urls = [d["url"] for d in got]
    assert "privat://ki-richtlinie.md" in urls and "privat://passwort-und-zugang.md" in urls


def test_tavily_ohne_key_liefert_nichts(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert r.tavily_all(["egal"]) == []


def test_tavily_antwort_wird_zu_seiten(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-platzhalter")
    seen = {}

    def fake_post(url, headers, json, timeout):
        seen.update(url=url, auth=headers["Authorization"], body=json)
        return httpx.Response(200, json={"results": [
            {"url": "https://eur-lex.europa.eu/x", "title": "AI Act", "content": "kurz", "raw_content": "lang"},
            {"url": "/relativ", "title": "kaputt"}]})

    monkeypatch.setattr(tavily_search.httpx, "post", fake_post)
    hits = tavily_search.search("AI Act", 5)
    assert seen["url"] == tavily_search.URL and seen["auth"] == "Bearer test-platzhalter"
    assert seen["body"]["include_raw_content"] == "markdown"
    assert hits == [{"url": "https://eur-lex.europa.eu/x", "title": "AI Act", "content": "kurz",
                     "full_text": "lang", "origin": "tavily"}]


def test_mix_haelt_beide_herkuenfte_im_bild():
    pages = [{"origin": "tavily"}] * 8 + [{"origin": "privat"}] * 4
    order = list(range(12))  # alle oeffentlichen vorne
    got = r._mix(order, pages, 6)
    assert len(got) == 6 and sum(pages[i]["origin"] == "privat" for i in got) == 3
    assert got == sorted(got)  # Rang-Reihenfolge bleibt
    assert r._mix(list(range(8)), pages[:8], 6) == list(range(6))


def test_research_mischt_privat_und_oeffentlich_mit_labels(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    public = [{"url": "https://a.org/ai-act", "title": "AI Act", "content": "", "origin": "tavily",
               "full_text": "Article 4 of the AI Act requires providers and deployers to ensure AI literacy of their staff."},
              {"url": "https://b.eu/ai-act", "title": "AI Act FAQ", "content": "", "origin": "tavily",
               "full_text": "Under the AI Act, deployers must ensure a sufficient level of AI literacy of their staff."}]
    monkeypatch.setattr(r, "tavily_all", lambda v: public)
    steps = {}
    res = r.research("Erfüllt unsere interne KI-Richtlinie den EU AI Act zur KI-Schulung?",
                     on_step=lambda s, i: steps.setdefault(s, i), rerank_mode="bm25",
                     quellen=["privat", "tavily"])
    assert steps["search"]["per_origin"]["tavily"] == 2 and steps["search"]["per_origin"]["privat"] >= 1
    assert {s.label for s in res.sources} == {"privat", "öffentlich"}
    assert all(s.stage == "bm25" for s in res.sources)
    for c in res.claims:  # 2-Quellen-Regel gilt weiter
        assert len({next(s.site for s in res.sources if s.n == cit["source"]) for cit in c.citations}) >= 2
    labels = {s["label"] for s in res.as_dict()["sources"]}
    assert labels <= {"privat", "öffentlich"}


def test_unbekannte_quelle_wird_abgelehnt():
    import pytest

    with pytest.raises(ValueError):
        r.research("x", quellen=["intranet"])


def test_regale_aus_dem_einlesen_und_erkennung_ohne_llm():
    regale = private.shelves(DEMO)
    assert "KI" in regale and "IT-Sicherheit" in regale
    assert private.detect_shelf(["Erfüllt unsere interne KI-Richtlinie die Pflichten aus dem EU AI Act?"], DEMO) == "KI"
    assert private.detect_shelf(["Entspricht unsere Passwortrichtlinie den NIST-Empfehlungen?"], DEMO) == "IT-Sicherheit"
    assert private.detect_shelf(["zzz qqq"], DEMO) is None


def test_suche_nur_im_erkannten_regal():
    got = private.search(["Passwort Schulung"], limit=5, root=DEMO, shelf="IT-Sicherheit")
    names = {d["url"].split("://")[1] for d in got}
    assert names and names <= {"passwort-und-zugang.md", "it-schulungsplan.txt", "lieferanten-checkliste.md"}


def test_ohne_regal_datei_alle_dokumente(tmp_path):
    (tmp_path / "a.md").write_text("# Eins\nPasswort Regeln für alle", encoding="utf-8")
    assert private.shelves(tmp_path) == {} and private.detect_shelf(["Passwort"], tmp_path) is None
    assert [d["url"] for d in private.search(["Passwort"], root=tmp_path)] == ["privat://a.md"]
