"""research.finalize, nemotron.parse_json und der MCP-Server, alles ohne Netz."""

from __future__ import annotations

import pytest

from app import mcp_server, nemotron
from app.research import Rejected, Source, finalize, heuristic_variants, plan

SRC = [
    Source(1, "https://a.org/x", "A", "a.org", "", "The bridge opened in 1937 and spans 2,737 metres across the strait."),
    Source(2, "https://b.com/y", "B", "b.com", "", "Opened to traffic in 1937, the bridge spans 2,737 metres in total."),
    Source(3, "https://a.org/z", "A2", "a.org", "", "It was painted international orange in 1937 by the builders."),
]


def claim(*cits):
    return {"claims": [{"text": "The bridge opened in 1937.", "citations": [
        {"source": n, "quote": q} for n, q in cits]}]}


def test_zwei_echte_zitate_von_zwei_seiten_gehen_durch():
    ok, problems = finalize(claim((1, "The bridge opened in 1937"), (2, "Opened to traffic in 1937")), SRC)
    assert len(ok) == 1 and problems == []


def test_erfundenes_zitat_wird_abgelehnt():
    with pytest.raises(Rejected) as exc:
        finalize(claim((1, "The bridge opened in 1937"), (2, "The bridge was finished in 1937")), SRC)
    assert any("steht nicht in Quelle [2]" in p for p in exc.value.problems)


def test_zwei_zitate_derselben_seite_reichen_nicht():
    with pytest.raises(Rejected):
        finalize(claim((1, "The bridge opened in 1937"), (3, "painted international orange in 1937")), SRC)


def test_unbekannte_quelle_und_zu_kurzes_zitat():
    with pytest.raises(Rejected) as exc:
        finalize(claim((9, "The bridge opened in 1937"), (2, "in 1937")), SRC)
    text = " ".join(exc.value.problems)
    assert "[9] gibt es nicht" in text and "zu kurz" in text


def test_typografische_zeichen_und_leerraum_zaehlen_nicht():
    src = [Source(1, "https://a.org", "", "a.org", "", "Er sagte: „Das  Werk ist fertig“ – sofort."),
           Source(2, "https://b.de", "", "b.de", "", 'Er sagte: "Das Werk ist fertig" - sofort.')]
    ok, _ = finalize(claim((1, 'Er sagte: "Das Werk ist fertig"'), (2, "Er sagte: „Das Werk ist fertig“")), src)
    assert len(ok) == 1


def test_gute_aussage_bleibt_trotz_schlechter_daneben():
    draft = claim((1, "The bridge opened in 1937"), (2, "Opened to traffic in 1937"))
    draft["claims"].append({"text": "Made up.", "citations": [{"source": 1, "quote": "nothing like this here"}]})
    ok, problems = finalize(draft, SRC)
    assert len(ok) == 1 and len(problems) == 2


def test_plan_ohne_llm_hat_mindestens_vier_varianten():
    assert len(plan("Wie hoch ist der Mount Everest?", use_llm=False)) >= 4
    assert heuristic_variants("Who won the 2022 FIFA World Cup?")[0] == "Who won the 2022 FIFA World Cup?"


@pytest.mark.parametrize("raw", [
    '{"claims": []}',
    '<think>hm {"x":1}</think>\n```json\n{"claims": []}\n```',
    'Sure! {"claims": []} done',
])
def test_parse_json_robust(raw):
    assert nemotron.parse_json(raw) == {"claims": []}


def test_kein_key_kein_llm(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    assert not nemotron.configured()


def test_mcp_handshake_und_tool_liste():
    init = mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert init["result"]["serverInfo"]["name"] == "searchify-nemotron"
    assert mcp_server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
    assert names == {"search", "read", "research"}
    bad = mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "nope"}})
    assert bad["error"]["code"] == -32602


def test_korrekturrunde_mit_nemotron(monkeypatch):
    """Erster Entwurf hat ein erfundenes Zitat, die Korrektur bekommt die Gruende und besteht."""
    from app import research as r

    monkeypatch.setenv("NEBIUS_API_KEY", "test-platzhalter")
    monkeypatch.setattr(r, "_warm_reranker", lambda mode=None: None)
    monkeypatch.setattr(r, "search_all", lambda v: [{"url": "https://a.org/x"}])
    monkeypatch.setattr(r, "read", lambda hits: [{**h, "full_text": ""} for h in hits])
    monkeypatch.setattr(r, "rank", lambda q, pages, mode=None: SRC)
    seen = []

    def fake_chat(messages, max_tokens=0):
        seen.append(messages[-1]["content"])
        if "plan web searches" in messages[0]["content"]:
            return '["bridge opening year", "Brücke Eröffnung Jahr", "bridge 1937", "bridge history official"]'
        if "rejected" in messages[-1]["content"]:
            return '{"claims":[{"text":"It opened in 1937.","citations":[{"source":1,"quote":"The bridge opened in 1937"},{"source":2,"quote":"Opened to traffic in 1937"}]}]}'
        return '{"claims":[{"text":"It opened in 1937.","citations":[{"source":1,"quote":"The bridge opened in 1937"},{"source":2,"quote":"opened on a sunny day in 1937"}]}]}'

    monkeypatch.setattr(nemotron, "chat", fake_chat)
    steps = []
    res = r.research("When did the bridge open?", on_step=lambda s, i: steps.append(s))
    assert res.mode == "nemotron" and len(res.claims) == 1 and res.rejected == []
    assert len(res.variants) >= 4 and steps.count("draft") == 2
    assert "steht nicht in Quelle [2]" in seen[-1]


def test_suche_verwirft_links_ohne_http(monkeypatch):
    from app import research as r
    from app import search as search_mod

    hits = [{"url": "/clev?event=Sp_x9&sc=abc", "title": "Anzeige"},
            {"url": "https://de.wikipedia.org/wiki/Berliner_Fernsehturm", "title": "Fernsehturm"}]
    monkeypatch.setattr(search_mod, "web_search", lambda q, n, **kw: hits)
    got = r.search_all(["a", "b"])
    assert [h["url"] for h in got] == ["https://de.wikipedia.org/wiki/Berliner_Fernsehturm"]
    assert got[0]["votes"] == 2
