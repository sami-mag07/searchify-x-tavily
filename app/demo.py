"""Live-Demo: eine echte Frage laeuft durch research() und jeder Schritt erscheint sofort.

    python -m app.demo "Wann wurde die Nebius Token Factory vorgestellt?"
    python -m app.demo --json "..."      # Endergebnis zusaetzlich als JSON
    python -m app.demo --nur-nemotron "..."  # Nemotron-Reranker auf allen Abschnitten (langsamer)
    python -m app.demo --nur-l "..."         # nur mMiniLM (Searchify L), zum Vergleich
    python -m app.demo --bm25 "..."          # ohne Reranker-Modell (leichte Rechner)
    python -m app.demo --privat-tavily "..." # Intranet-Ordner (PRIVATE_DIR) plus Tavily
    python -m app.demo --privat-tavily --frage 1   # Demo-Frage 1 bzw. 2 aus demo/fragen.txt
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

from . import nemotron

FRAGEN = Path(__file__).resolve().parent.parent / "demo" / "fragen.txt"
FLAGS = ("--json", "--nur-nemotron", "--nur-l", "--bm25", "--privat-tavily", "--frage")

_T0 = time.perf_counter()


def _say(label: str, text: str) -> None:
    print(f"\033[2m{time.perf_counter() - _T0:5.1f}s\033[0m  \033[1m{label:<9}\033[0m {text}", flush=True)


def show(step: str, info: dict) -> None:
    if step == "start":
        _say("frage", info["question"])
        _say("modell", info["model"])
        _say("quellen", " + ".join(info["quellen"]))
    elif step == "plan":
        _say("plan", f"{len(info['variants'])} Suchvarianten")
        for v in info["variants"]:
            print(f"{'':17}- {v}")
    elif step == "regal":
        _say("regal", info["name"] or "keins erkannt, alle privaten Dokumente")
    elif step == "search":
        parts = ", ".join(f"{n} {name}" for name, n in info["per_origin"].items())
        _say("suche", f"{info['hits']} Dokumente und Seiten ({parts})")
    elif step == "read":
        _say("lesen", f"{info['pages']} Seiten geladen, {info['with_text']} mit Volltext")
    elif step == "rerank":
        how = {"l+nemotron": "mMiniLM sortiert alles, Nemotron die Top 5",
               "nemotron": "Nemotron-Reranker auf allen Abschnitten",
               "l": "nur mMiniLM", "bm25": "BM25 ohne Modell"}[info["mode"]]
        _say("rerank", f"{how}, beste Quellen:")
        for n, site, score, stage in info["sources"][:6]:
            print(f"{'':17}[{n}] {site}  {score:.3f}  ({stage})")
    elif step == "draft":
        _say("entwurf", f"Runde {info['attempt']}: {info['claims']} Aussagen mit Zitaten")
    elif step == "draft_error":
        _say("entwurf", f"Nemotron nicht erreichbar ({info['error']}), extraktiver Ersatz")
    elif step == "finalize":
        _say("prüfung", f"{info['accepted']} Aussagen belegt, {len(info['rejected'])} abgelehnt")
        for why in info["rejected"][:5]:
            print(f"{'':17}x {why}")



def _label(source) -> str:
    return "\033[1;35mprivat\033[0m" if source.origin == "privat" else "\033[1;36möffentlich\033[0m"


def main(argv: list) -> int:
    as_json = "--json" in argv
    mode = ("nemotron" if "--nur-nemotron" in argv else "l" if "--nur-l" in argv
            else "bm25" if "--bm25" in argv else "l+nemotron")
    quellen = ("privat", "tavily") if "--privat-tavily" in argv else ("web",)
    words = [a for a in argv if a not in FLAGS]
    if "--frage" in argv and words and words[0].isdigit():
        fragen = [z.strip() for z in FRAGEN.read_text(encoding="utf-8").splitlines() if z.strip()]
        words = [fragen[int(words[0]) - 1]]
    if not words:
        print(__doc__)
        return 2
    nemotron.load_env_file()
    from .research import research  # noqa: PLC0415 (nach .env laden)

    result = research(" ".join(words), on_step=show, rerank_mode=mode, quellen=quellen)
    by_n = {s.n: s for s in result.sources}
    print()
    for c in result.claims:
        refs = "".join(f"[{cit['source']}]" for cit in c.citations)
        print(textwrap.fill(f"{c.text} {refs}", 100))
        for cit in c.citations:
            src = by_n[cit["source"]]
            print(f"    [{cit['source']}] {_label(src)}  {src.site if src.origin != 'privat' else src.url}")
            print(textwrap.fill(f'„{cit["quote"]}"', 100, initial_indent="        ",
                                subsequent_indent="        "))
    if not result.claims:
        print(result.answer)
    used = {cit["source"] for c in result.claims for cit in c.citations}
    print("\nQuellen:")
    for s in result.sources:
        if s.n in used:
            print(f"  [{s.n}] {_label(s)}  {s.url}")
    if as_json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
