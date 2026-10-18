#!/usr/bin/env python3
"""Lehrer-Scores: volles Nemotron bewertet Anfrage-Gruppen aus den oeffentlichen
Trainingsdaten (train/data/distill.jsonl, Eval-Fragen dort schon herausgefiltert).
Ausgabe teacher.jsonl: {query, lang, passages[], scores[], split}."""
import argparse
import collections
import hashlib
import json
import random

import torch

from common import BASE, encode, load_model, load_tokenizer, log

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="train/data/distill.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--per-lang", type=int, default=6000)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()

groups = collections.defaultdict(list)
lang_of = {}
with open(a.src, encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        q = d["query"]
        if len(groups[q]) < a.group:
            groups[q].append(d["passage"])
            lang_of[q] = d["lang"]
by_lang = collections.defaultdict(list)
for q, ps in groups.items():
    if len(ps) >= 4:
        by_lang[lang_of[q]].append(q)
log("groups", **{k: len(v) for k, v in by_lang.items()})

rng = random.Random(7)
picked = []
for lang, qs in sorted(by_lang.items()):
    qs.sort()
    rng.shuffle(qs)
    picked += [(q, lang) for q in qs[: a.per_lang]]
del by_lang

tok = load_tokenizer()
model = load_model(BASE).eval()
pairs = [(i, j, q, p) for i, (q, _) in enumerate(picked) for j, p in enumerate(groups[q])]
pairs.sort(key=lambda x: len(x[3]))
scores = collections.defaultdict(dict)
with torch.inference_mode():
    for k in range(0, len(pairs), a.batch):
        chunk = pairs[k:k + a.batch]
        logits = model(**encode(tok, [c[2] for c in chunk], [c[3] for c in chunk])).logits.view(-1).float().tolist()
        for c, s in zip(chunk, logits):
            scores[c[0]][c[1]] = s
        if k // a.batch % 250 == 0:
            log("teacher", done=k, total=len(pairs))

with open(a.out, "w", encoding="utf-8") as fh:
    for i, (q, lang) in enumerate(picked):
        ps = groups[q]
        split = "val" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 100 < 3 else "train"
        fh.write(json.dumps({"query": q, "lang": lang, "passages": ps,
                             "scores": [scores[i][j] for j in range(len(ps))], "split": split},
                            ensure_ascii=False) + "\n")
log("fertig", queries=len(picked), pairs=len(pairs))
