#!/usr/bin/env python3
"""Destillation: gekuerztes Modell lernt die Lehrer-Scores des vollen Nemotron.
Verlust je Anfrage-Gruppe: KL der Rangverteilungen (Temperatur TAU) plus kleiner
MSE-Anteil auf den rohen Logits. Embedding eingefroren, Schichten mit Checkpointing."""
import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common import encode, load_model, load_tokenizer, log, n_params

ap = argparse.ArgumentParser()
ap.add_argument("--teacher", required=True)
ap.add_argument("--student", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--groups-per-step", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--tau", type=float, default=2.0)
ap.add_argument("--mse", type=float, default=0.01)
ap.add_argument("--eval-every", type=int, default=750)
a = ap.parse_args()

rows = [json.loads(x) for x in open(a.teacher, encoding="utf-8")]
train = [r for r in rows if r["split"] == "train"]
val = [r for r in rows if r["split"] == "val"]
log("daten", train=len(train), val=len(val))

tok = load_tokenizer()
model = load_model(a.student, dtype=torch.float32)
model.model.embed_tokens.weight.requires_grad_(False)


class Ckpt(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states, **kw):
        return checkpoint(lambda h: self.inner(h, **kw), hidden_states, use_reentrant=False)


model.model.layers = torch.nn.ModuleList(Ckpt(l) for l in model.model.layers)
log("student", params=n_params(model), trainierbar=sum(p.numel() for p in model.parameters() if p.requires_grad))

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
steps = int(len(train) * a.epochs / a.groups_per_step)
warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))


def scores_for(groups, train_mode):
    qs = [g["query"] for g in groups for _ in g["passages"]]
    ps = [p for g in groups for p in g["passages"]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**encode(tok, qs, ps)).logits.view(-1).float()
    res, k = [], 0
    for g in groups:
        n = len(g["passages"])
        res.append(out[k:k + n])
        k += n
    return res


def evaluate():
    model.eval()
    top1, spear, n = 0, 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(val), 8):
            groups = val[i:i + 8]
            for g, s in zip(groups, scores_for(groups, False)):
                t = torch.tensor(g["scores"], device=s.device)
                top1 += int(s.argmax() == t.argmax())
                rs, rt = s.argsort().argsort().float(), t.argsort().argsort().float()
                spear += torch.corrcoef(torch.stack([rs, rt]))[0, 1].nan_to_num(0).item()
                n += 1
    model.train()
    return {"top1_gleich_lehrer": round(top1 / n, 4), "spearman": round(spear / n, 4)}


def save(out):
    """bf16-Kopie speichern, ohne die fp32-Gewichte im Training anzutasten."""
    wrapped = model.model.layers
    model.model.layers = torch.nn.ModuleList(l.inner for l in wrapped)
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(out, state_dict=sd, safe_serialization=True)
    model.model.layers = wrapped
    for f in Path(a.student).glob("*.py"):
        shutil.copy(f, Path(out) / f.name)


log("eval_vorher", **evaluate())
best = -1.0
rng = random.Random(11)
model.train()
step = 0
order = []
while step < steps:
    if not order:
        order = list(range(len(train)))
        rng.shuffle(order)
    batch = [train[order.pop()] for _ in range(min(a.groups_per_step, len(order)))]
    loss = 0.0
    for g, s in zip(batch, scores_for(batch, True)):
        t = torch.tensor(g["scores"], device=s.device)
        kl = F.kl_div(F.log_softmax(s / a.tau, -1), F.softmax(t / a.tau, -1), reduction="sum") * a.tau ** 2
        loss = loss + kl + a.mse * F.mse_loss(s, t)
    loss = loss / len(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)
    step += 1
    if step % 100 == 0:
        log("train", step=step, steps=steps, loss=round(loss.item(), 4), lr=sched.get_last_lr()[0])
    if step % a.eval_every == 0 or step == steps:
        m = evaluate()
        log("eval", step=step, **m)
        if m["spearman"] > best:
            best = m["spearman"]
            save(a.out)
            log("gespeichert", step=step, spearman=best)
log("fertig", best_spearman=best)
