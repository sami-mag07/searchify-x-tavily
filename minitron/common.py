"""Gemeinsames fuer Lehrer, Pruning und Destillation von Nemotron-Rerank."""
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import NEMOTRON_REVISION  # noqa: E402

BASE = "nvidia/llama-nemotron-rerank-1b-v2"
MAX_LEN = 288  # 256 Text-Tokens wie mMiniLM (RERANK_MAX_LENGTH) plus Prompt und Frage

# Unter WSL weicht CUDA bei vollem Grafikspeicher still in den Arbeitsspeicher aus (sehr
# langsam, gemessen 17.10.2026: 5 GB ausgelagert). Lieber laut mit OOM abbrechen.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)


def fmt(q, p):
    return f"question:{q} \n \n passage:{p}"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE, revision=NEMOTRON_REVISION, trust_remote_code=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, dtype=torch.bfloat16, device="cuda"):
    """Vom Hub nur BASE in der festen Revision, sonst ein lokaler Pfad aus diesem Ablauf."""
    revision = NEMOTRON_REVISION if path == BASE else None
    m = AutoModelForSequenceClassification.from_pretrained(path, revision=revision, trust_remote_code=True,
                                                           torch_dtype=dtype)
    if m.config.pad_token_id is None:
        m.config.pad_token_id = m.config.eos_token_id if isinstance(m.config.eos_token_id, int) else 128001
    return m.to(device)


def encode(tok, qs, ps, device="cuda", max_length=MAX_LEN):
    b = tok([fmt(q, p) for q, p in zip(qs, ps)], padding=True, truncation=True,
            max_length=max_length, return_tensors="pt")
    return {k: v.to(device) for k, v in b.items()}


def n_params(m, skip_embed=False):
    return sum(p.numel() for n, p in m.named_parameters() if not (skip_embed and "embed_tokens" in n))


def log(event, **kw):
    print(json.dumps({"t": time.strftime("%H:%M:%S"), "event": event, **kw}, ensure_ascii=False), flush=True)
