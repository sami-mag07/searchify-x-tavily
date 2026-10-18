"""Private Quelle: ein lokaler Ordner (md/txt/html/pdf) als Firmen-Intranet.

Dateien werden beim ersten Zugriff gelesen (HTML ueber dieselbe Extraktion wie Webseiten,
PDF ueber pypdf, falls installiert) und im Speicher gehalten. search() waehlt per BM25 die
passenden Dokumente; die eigentliche Sortierung macht danach research.rank() gemeinsam mit
den oeffentlichen Treffern (mMiniLM, optional Nemotron auf den Top 5).

Bibliothek: liegt neben dem Ordner eine <ordner>.regale.json (Einlesen mit einem lokalen LLM,
app/private_ingest.py), erkennt detect_shelf() das Regal der Frage ohne LLM, und search() sucht
nur in diesem Regal.

Ordner: Argument oder PRIVATE_DIR, Standard demo/intranet im Repo.
Quellen-URLs haben die Form privat://<relativer Pfad>, damit sie nie mit Web-URLs verwechselt werden.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

SUFFIXES = (".md", ".txt", ".html", ".htm", ".pdf")
SKIP = {"liesmich", "readme"}  # Hinweisdateien zum Ordner, keine Firmendokumente
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "demo" / "intranet"
MAX_BYTES = 5_000_000

_CACHE: Dict[str, List[Dict]] = {}
_LOCK = threading.Lock()


def folder() -> Path:
    return Path(os.environ.get("PRIVATE_DIR", "").strip() or DEFAULT_DIR)


def _text_of(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # noqa: PLC0415 (optional)
        except ImportError:
            return ""
        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in (".html", ".htm"):
        from .extract import extract_main_text  # noqa: PLC0415

        return extract_main_text(raw) or ""
    return raw


def _title(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:120]
    return path.stem


def load(root: Optional[Path] = None) -> List[Dict]:
    """Alle lesbaren Dokumente des Ordners als Seiten {url, title, content, full_text, origin}."""
    root = Path(root or folder()).resolve()
    key = str(root)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        docs = []
        for path in sorted(root.rglob("*")):
            if (path.suffix.lower() not in SUFFIXES or path.stem.lower() in SKIP or not path.is_file()
                    or path.stat().st_size > MAX_BYTES):
                continue
            text = _text_of(path).strip()
            if not text:
                continue
            rel = path.relative_to(root).as_posix()
            docs.append({"url": f"privat://{rel}", "title": _title(path, text),
                         "content": text[:300], "full_text": text, "origin": "privat"})
        _CACHE[key] = docs
        return docs


def shelves_path(root: Optional[Path] = None) -> Path:
    root = Path(root or folder()).resolve()
    return root.parent / f"{root.name}.regale.json"


def shelves(root: Optional[Path] = None) -> Dict[str, List[Dict]]:
    """Regal -> Dokumente laut <ordner>.regale.json (von app/private_ingest.py beim Einlesen
    geschrieben). Ohne Datei: leer, dann wird ueber alle Dokumente gesucht."""
    path = shelves_path(root)
    if not path.is_file():
        return {}
    import json  # noqa: PLC0415

    rows = json.loads(path.read_text(encoding="utf-8")).get("dokumente") or {}
    out: Dict[str, List[Dict]] = {}
    for d in load(root):
        row = rows.get(d["url"].split("://", 1)[1])
        if row:
            out.setdefault(row["regal"], []).append({**d, "summary": row.get("zusammenfassung", "")})
    return out


def detect_shelf(queries: List[str], root: Optional[Path] = None) -> Optional[str]:
    """Regal der Frage ohne LLM: BM25 der Wortstaemme (wie die Themen-Bibliothek) gegen ein
    Profil je Regal aus Regalname, Zusammenfassungen und Dokumenttexten. None = kein Treffer."""
    from .library import stems  # noqa: PLC0415
    from .rerank import lexical_scores  # noqa: PLC0415

    shelf_docs = shelves(root)
    if not shelf_docs:
        return None
    names = sorted(shelf_docs)
    profiles = [" ".join(stems(" ".join([n, n] + [f"{d['title']} {d['summary']} {d['full_text']}"
                                                  for d in shelf_docs[n]]))) for n in names]
    total = [0.0] * len(names)
    for q in queries:
        for i, sc in enumerate(lexical_scores(" ".join(stems(q)), profiles)):
            total[i] += sc
    best = max(range(len(names)), key=lambda i: total[i])
    return names[best] if total[best] > 0 else None


def search(queries: List[str], limit: int = 4, root: Optional[Path] = None,
           shelf: Optional[str] = None) -> List[Dict]:
    """Die `limit` passendsten Dokumente fuer alle Suchvarianten zusammen (BM25 je Abschnitt,
    je Dokument zaehlt der beste Abschnitt). Mit shelf nur die Dokumente dieses Regals.
    Dokumente ohne Treffer bleiben draussen."""
    from .extract import chunk_text  # noqa: PLC0415
    from .rerank import lexical_scores  # noqa: PLC0415

    docs = (shelves(root).get(shelf) or load(root)) if shelf else load(root)
    pieces = [(i, piece) for i, d in enumerate(docs) for piece in chunk_text(d["full_text"], 120, 30)]
    if not pieces:
        return []
    best: Dict[int, float] = {}
    for q in queries:
        for (i, _), s in zip(pieces, lexical_scores(q, [p for _, p in pieces])):
            best[i] = max(best.get(i, 0.0), s)
    order = sorted((i for i in best if best[i] > 0), key=lambda i: -best[i])
    return [docs[i] for i in order[:limit]]
