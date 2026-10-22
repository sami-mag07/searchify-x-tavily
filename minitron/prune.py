#!/usr/bin/env python3
"""Minitron-artiges Pruning ohne Training:
- Tiefe: Block Importance je Schicht = 1 - mittlere Kosinus-Aehnlichkeit von Ein- und
  Ausgang (nur echte Tokens). Die unwichtigsten Schichten fliegen, erste und letzte bleiben.
- Breite: MLP-Neuronen nach mittlerem Betrag ihrer Aktivierung (Eingang von down_proj),
  je Schicht die wichtigsten behalten.
Kalibrierung auf Lehrer-Daten (train-Teil)."""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from common import BASE, NEMOTRON_REVISION, encode, load_model, load_tokenizer, log, n_params

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=8)
ap.add_argument("--ffn", type=int, default=3072)
ap.add_argument("--calib", type=int, default=512)
a = ap.parse_args()

calib = []
for line in open(a.teacher, encoding="utf-8"):
    d = json.loads(line)
    if d["split"] == "train":
        calib += [(d["query"], p) for p in d["passages"][:2]]
    if len(calib) >= a.calib:
        break

tok = load_tokenizer()
model = load_model(BASE, dtype=torch.float32).eval()
cls = model
core = model.model
layers = core.layers
L = len(layers)
bi_sum = torch.zeros(L, dtype=torch.float64)
bi_n = 0
act_sum = [torch.zeros(layers[0].mlp.down_proj.in_features, dtype=torch.float64, device="cuda") for _ in range(L)]
state = {"mask": None}


def pre_hook(i):
    def h(mod, args, kwargs):
        mod._in = args[0] if args else kwargs["hidden_states"]
    return h


def post_hook(i):
    def h(mod, args, kwargs, out):
        hs = out[0] if isinstance(out, tuple) else out
        m = state["mask"]
        cos = torch.nn.functional.cosine_similarity(mod._in.float(), hs.float(), dim=-1)
        bi_sum[i] += ((1 - cos) * m).sum().item()
        del mod._in
    return h


def mlp_hook(i):
    def h(mod, args):
        x = args[0].float().abs() * state["mask"][..., None]
        act_sum[i] += x.sum(dim=(0, 1)).double()
    return h


hooks = []
for i, layer in enumerate(layers):
    hooks.append(layer.register_forward_pre_hook(pre_hook(i), with_kwargs=True))
    hooks.append(layer.register_forward_hook(post_hook(i), with_kwargs=True))
    hooks.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_hook(i)))

with torch.inference_mode():
    for k in range(0, len(calib), 16):
        chunk = calib[k:k + 16]
        enc = encode(tok, [c[0] for c in chunk], [c[1] for c in chunk])
        state["mask"] = enc["attention_mask"].float()
        bi_n += int(enc["attention_mask"].sum())
        model(**enc)
for h in hooks:
    h.remove()

bi = (bi_sum / bi_n).tolist()
log("block_importance", bi=[round(x, 4) for x in bi])
candidates = sorted(range(1, L - 1), key=lambda i: bi[i])
drop = set(candidates[: L - a.layers])
keep = [i for i in range(L) if i not in drop]
log("layers", keep=keep, drop=sorted(drop))

new_layers = torch.nn.ModuleList()
for new_idx, i in enumerate(keep):
    layer = layers[i]
    idx = torch.topk(act_sum[i], a.ffn).indices.sort().values.to(layer.mlp.gate_proj.weight.device)
    for name in ("gate_proj", "up_proj"):
        lin = getattr(layer.mlp, name)
        new = torch.nn.Linear(lin.in_features, a.ffn, bias=False)
        new.weight.data = lin.weight.data[idx].clone()
        setattr(layer.mlp, name, new)
    lin = layer.mlp.down_proj
    new = torch.nn.Linear(a.ffn, lin.out_features, bias=False)
    new.weight.data = lin.weight.data[:, idx].clone()
    layer.mlp.down_proj = new
    layer.mlp.intermediate_size = a.ffn
    layer.self_attn.layer_idx = new_idx
    new_layers.append(layer)
core.layers = new_layers
cls.config.num_hidden_layers = a.layers
cls.config.intermediate_size = a.ffn
core.config.num_hidden_layers = a.layers
core.config.intermediate_size = a.ffn

out = Path(a.out)
cls.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
src = Path(snapshot_download(BASE, revision=NEMOTRON_REVISION, allow_patterns=["*.py"]))
for f in src.glob("*.py"):
    shutil.copy(f, out / f.name)
log("gespeichert", out=str(out), params=n_params(cls), params_ohne_embedding=n_params(cls, skip_embed=True))
