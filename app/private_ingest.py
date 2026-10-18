"""Einlesen (einmalig, offline): ein lokales Sprachmodell stellt jedes private Dokument in ein
Regal (Thema) und schreibt einen Satz Zusammenfassung. Ergebnis als JSON neben dem Ordner
(<ordner>.regale.json); im heissen Pfad liest app/private.py nur noch diese Datei, ohne LLM.

    python -m app.private_ingest [ordner] [--url http://127.0.0.1:8080/v1] [--modell qwen3-4b]

Der Endpunkt ist OpenAI-kompatibel (z. B. llama-server). Alle Dokumente gehen in EINER Anfrage
an das Modell, damit es Regalnamen ueber die Dokumente hinweg einheitlich vergibt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import httpx

from . import private
from .nemotron import parse_json

DEFAULT_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL = "qwen3-4b"
CHARS_PER_DOC = 900
MAX_SHELVES = 5

PROMPT = (
    "You organise a company intranet into a library. Below are {n} internal documents. "
    "Put every document on exactly one shelf. Use AT MOST {hi} different shelves in total, so "
    "several documents MUST share a shelf; group by broad topic (for example AI and compliance, "
    "IT security, HR and organisation), not one shelf per document. Shelf names are short German "
    "topics (1 to 3 words). For every "
    "document also write one German sentence that summarises it. Return ONLY JSON: "
    '{{"dokumente": [{{"datei": "<file>", "regal": "<shelf>", "zusammenfassung": "<sentence>"}}]}} /no_think'
)


def _files(docs: List[Dict]) -> List[str]:
    return [d["url"].split("://", 1)[1] for d in docs]


def build(root: Path, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL) -> Dict:
    docs = private.load(root)
    files = _files(docs)
    body = "\n\n".join(f"### {f}\n{d['full_text'][:CHARS_PER_DOC]}" for f, d in zip(files, docs))
    t0 = time.perf_counter()
    resp = httpx.post(f"{url.rstrip('/')}/chat/completions", timeout=300.0, json={
        "model": model, "temperature": 0, "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": PROMPT.format(n=len(docs), hi=MAX_SHELVES)},
                     {"role": "user", "content": body}],
    })
    resp.raise_for_status()
    got = parse_json(resp.json()["choices"][0]["message"]["content"] or "")
    rows = {str(r.get("datei", "")).strip(): r for r in (got.get("dokumente") or [])}
    missing = [f for f in files if f not in rows or not str(rows[f].get("regal", "")).strip()]
    if missing:
        raise ValueError(f"Modell hat diese Dokumente keinem Regal zugeordnet: {missing}")
    names = {str(rows[f]["regal"]).strip() for f in files}
    if len(names) > MAX_SHELVES or len(names) == len(files):
        raise ValueError(f"Modell hat {len(names)} Regale fuer {len(files)} Dokumente vergeben: {sorted(names)}")
    out = {"modell": model, "erstellt": time.strftime("%Y-%m-%d %H:%M"),
           "dauer_s": round(time.perf_counter() - t0, 1), "dokumente": {}}
    for f in files:
        out["dokumente"][f] = {"regal": str(rows[f]["regal"]).strip(),
                               "zusammenfassung": str(rows[f].get("zusammenfassung", "")).strip()}
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ordner", nargs="?", default=None)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--modell", default=DEFAULT_MODEL)
    a = ap.parse_args(argv)
    root = Path(a.ordner) if a.ordner else private.folder()
    data = build(root, a.url, a.modell)
    path = private.shelves_path(root)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for f, row in data["dokumente"].items():
        print(f"{row['regal']:<24} {f}")
    print(f"-> {path} ({data['dauer_s']} s, {data['modell']})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
