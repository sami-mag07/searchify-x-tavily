"""Nemotron ueber die Nebius Token Factory (OpenAI-kompatibel).

Key AUSSCHLIESSLICH aus der Umgebung bzw. .env (NEBIUS_API_KEY), nie im Code.
Base-URL und Modell laut offizieller Doku (docs.tokenfactory.nebius.com,
api-reference/introduction; Modell-ID aus dem Nebius-Blog zu Nemotron 3 Super):
  https://api.tokenfactory.nebius.com/v1/  und  nvidia/nemotron-3-super-120b-a12b
Beides laesst sich ueber NEMOTRON_BASE_URL / NEMOTRON_MODEL umstellen.

Ohne Key ist configured() False; research.py faellt dann auf einen Stub zurueck.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
TIMEOUT_S = 90.0

_THINK = re.compile(r"<think>.*?</think>", re.S)


def load_env_file(path: Optional[Path] = None) -> None:
    """Liest eine .env (KEY=WERT) in os.environ, ohne Gesetztes zu ueberschreiben.
    Kein python-dotenv noetig; Werte werden nie ausgegeben."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def configured() -> bool:
    return bool(os.environ.get("NEBIUS_API_KEY", "").strip())


def model_name() -> str:
    return os.environ.get("NEMOTRON_MODEL", "").strip() or DEFAULT_MODEL


def _client():
    from openai import OpenAI  # noqa: PLC0415 (lazy, wie app/answer.py)

    return OpenAI(
        base_url=os.environ.get("NEMOTRON_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        api_key=os.environ["NEBIUS_API_KEY"].strip(),
        timeout=TIMEOUT_S,
        max_retries=1,
    )


def thinking() -> bool:
    """Denkmodus von Nemotron 3 (NEMOTRON_THINKING=1). Standard aus: gemessen 18.10.2026 mit
    echten Quellen gleich viele gepruefte Aussagen (3/3), 2 s statt 11 s, und mit Denken
    reichte das Token-Budget nicht immer (abgeschnittenes JSON)."""
    return os.environ.get("NEMOTRON_THINKING", "0").strip() in ("1", "true", "yes")


def chat(messages: List[Dict], max_tokens: int = 2000) -> str:
    """Eine Antwort von Nemotron, ohne Denk-Block."""
    completion = _client().chat.completions.create(
        model=model_name(),
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens * 3 if thinking() else max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking()}},
    )
    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("Nemotron-Antwort abgeschnitten (Token-Budget)")
    return _THINK.sub("", choice.message.content or "").strip()


def parse_json(text: str):
    """Erstes JSON-Objekt bzw. -Array aus einer Modellantwort (auch in ```json-Bloecken)."""
    text = _THINK.sub("", text)
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("keine JSON-Struktur in der Modellantwort")
    obj, _ = json.JSONDecoder().raw_decode(text[min(starts):])
    return obj


def chat_json(messages: List[Dict], max_tokens: int = 2000):
    return parse_json(chat(messages, max_tokens))
