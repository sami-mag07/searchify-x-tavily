"""Oeffentliches Web ueber die Tavily-Search-API (Sponsor): Tavily liefert das offene Web,
Searchify ergaenzt es um private Quellen (app/private.py).

Key AUSSCHLIESSLICH aus der Umgebung bzw. .env (TAVILY_API_KEY), nie im Code und nie im Log.
Doku: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""

from __future__ import annotations

import os
from typing import Dict, List

import httpx

URL = "https://api.tavily.com/search"
TIMEOUT_S = 15.0


def configured() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def search(query: str, max_results: int = 5) -> List[Dict]:
    """Treffer als {url, title, content, full_text}; full_text ist Tavilys raw_content
    (Seitentext als Markdown), damit die Seiten nicht noch einmal geladen werden muessen."""
    resp = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY'].strip()}"},
        json={"query": query, "max_results": max_results, "search_depth": "basic",
              "include_raw_content": "markdown"},
        timeout=TIMEOUT_S,
    )
    if resp.status_code != 200:
        # Antworttext nicht ausgeben: er koennte Anfrage-Details enthalten
        raise RuntimeError(f"Tavily antwortet mit HTTP {resp.status_code}")
    hits = []
    for r in resp.json().get("results") or []:
        url = r.get("url") or ""
        if not url.startswith(("http://", "https://")):
            continue
        hits.append({"url": url, "title": r.get("title") or "", "content": r.get("content") or "",
                     "full_text": r.get("raw_content") or r.get("content") or "",
                     "origin": "tavily"})
    return hits
