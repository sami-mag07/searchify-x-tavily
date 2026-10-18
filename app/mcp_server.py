"""Schlanker MCP-Server (stdio, JSON-RPC 2.0, ohne Zusatzpaket) mit search, read, research.

Einbinden, z. B. in Claude Code:
    claude mcp add searchify -- /pfad/zur/.venv/bin/python -m app.mcp_server
(Arbeitsverzeichnis = Repo-Wurzel, NEBIUS_API_KEY in .env oder Umgebung.)
"""

from __future__ import annotations

import json
import sys
from typing import Dict

from . import nemotron

PROTOCOL = "2024-11-05"

TOOLS = [
    {
        "name": "search",
        "description": "Web search. Returns title, url and snippet per hit.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 8}},
            "required": ["query"],
        },
    },
    {
        "name": "read",
        "description": "Read web pages in full and return their main text.",
        "inputSchema": {
            "type": "object",
            "properties": {"urls": {"type": "array", "items": {"type": "string"}, "maxItems": 10}},
            "required": ["urls"],
        },
    },
    {
        "name": "research",
        "description": (
            "Answer a question end to end with NVIDIA Nemotron: several search variants, reading, "
            "Nemotron rerank, answer in which every claim carries verbatim quotes from 2 different "
            "sources, checked in code. sources: web (own search backend), tavily (public web via "
            "Tavily), private (local company folder); each cited source is labelled private or public."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"},
                           "sources": {"type": "array", "items": {"enum": ["web", "tavily", "private"]},
                                       "default": ["web"]}},
            "required": ["question"],
        },
    },
]


def call_tool(name: str, args: Dict) -> Dict:
    if name == "search":
        from . import lang as lang_mod  # noqa: PLC0415
        from . import search as search_mod  # noqa: PLC0415

        query = str(args["query"])
        n = max(1, min(int(args.get("max_results") or 8), 20))
        hits = search_mod.web_search(query, n, search_depth="basic", language=lang_mod.detect(query))
        return {"results": [{"title": h.get("title", ""), "url": h.get("url", ""),
                             "snippet": h.get("content", "")} for h in hits[:n]]}
    if name == "read":
        from .pages import read_pages  # noqa: PLC0415

        urls = [str(u) for u in args["urls"]][:10]
        docs, failed = read_pages(urls, deadline_s=15)
        return {"results": [{"url": d.get("url"), "title": d.get("title", ""),
                             "content": d.get("content", "")} for d in docs], "failed": failed}
    if name == "research":
        from .research import research  # noqa: PLC0415

        names = {"web": "web", "tavily": "tavily", "private": "privat"}
        quellen = [names[str(q)] for q in args.get("sources") or ["web"]]
        return research(str(args["question"]), quellen=quellen).as_dict()
    raise KeyError(name)


def handle(msg: Dict):
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL, "capabilities": {"tools": {}},
                  "serverInfo": {"name": "searchify-nemotron", "version": "0.1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            out = call_tool(params.get("name", ""), params.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}]}
        except KeyError as exc:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": f"unknown tool or argument {exc}"}}
        except Exception as exc:  # noqa: BLE001 - Fehler als Tool-Ergebnis, Server laeuft weiter
            result = {"content": [{"type": "text", "text": f"error: {type(exc).__name__}"}], "isError": True}
    elif method == "ping":
        result = {}
    elif mid is None:
        return None  # Notification (z. B. notifications/initialized)
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> None:
    nemotron.load_env_file()
    # stdout gehoert dem Protokoll: Bibliotheksausgaben nach stderr umleiten
    out, sys.stdout = sys.stdout, sys.stderr
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line))
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
