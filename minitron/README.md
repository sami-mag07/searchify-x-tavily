# Minitron-style pruning of Llama-Nemotron-Rerank

Searchify reranks 20 to 30 search results per query. This folder shrinks NVIDIA's
`nvidia/llama-nemotron-rerank-1b-v2` so it can do that job fast enough, following the
idea of NVIDIA's Minitron work: prune by measured importance, then distill from the
full model.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `teacher.py` | The full model scores 24,000 public queries (6,000 each in de, en, es, fr) with 4 to 8 passages each, 124,434 pairs. 3 % of queries are held out for validation. |
| 2 | `prune.py` | Depth: block importance per layer (1 minus the mean cosine similarity of layer input and output over real tokens), the 8 least important of 16 layers are removed, first and last always stay. Width: MLP neurons ranked by mean activation magnitude, 3,072 of 8,192 kept per layer. |
| 3 | `distill.py` | The pruned student learns the teacher's scores: KL divergence of the per-query ranking distribution (temperature 2) plus a small MSE term on raw logits. Embeddings frozen, gradient checkpointing, one epoch. |

Pruned model: 498 M parameters, of which 235 M do per-token compute (the full model has
968 M besides its 262 M embedding table). Layers kept: 0, 1, 2, 3, 4, 5, 8, 15.

## Results

Measured on 17 October 2026 on an RTX 3070 Ti laptop GPU, replaying recorded Searchify
searches (`eval/objective.py`: dev = 50 fact questions plus research cassette, test = 75
unseen questions plus niche research). All models in neutral mode, i.e. without the
score thresholds that were tuned on mMiniLM logits.

| Reranker | dev | test | ms per rerank call |
|---|---|---|---|
| mMiniLM (Searchify default, 118 M) | 0.923 | 0.851 | ~20 |
| Nemotron 1B, full | 0.933 | 0.825 | ~270 |
| Nemotron pruned, before distillation | 0.609 | 0.616 | ~49 |
| Nemotron pruned + distilled | 0.721 | 0.639 | ~49 |

Distillation ran one epoch (5,823 steps, 1.7 h). On held-out queries the student picks the
teacher's top passage 41 % of the time (28 % before) and reaches a Spearman of 0.28 against
the teacher's scores (0.06 before); gains flattened after about half the epoch. The pruned
student is therefore not good enough to replace mMiniLM. Halving the depth removed too much
for 24,000 queries of distillation data to recover.

## Reproduce

Needs a CUDA GPU with 8 GB, the public training pairs from `train/prepare_data.py`
(`train/data/distill.jsonl`) and, under WSL, a C compiler for Triton (`build-essential`).

```
python minitron/teacher.py --out run/teacher.jsonl
python minitron/prune.py --teacher run/teacher.jsonl --out run/pruned
python minitron/distill.py --teacher run/teacher.jsonl --student run/pruned --out run/student
RERANKER_FORMAT=nemotron RERANKER_MODEL=$PWD/run/student python eval/objective.py --split both
```

`common.py` caps the CUDA allocator at 90 % of GPU memory. Under WSL, CUDA otherwise
spills silently into system RAM when the card is full, which slowed step 1 to a crawl.

## Licenses

- Model: OpenMDW-1.1 plus the Llama 3.2 Community License. Derived models must say
  "Built with Llama" and their name must start with "Llama", for example
  `Llama-Nemotron-Rerank-Searchify-Minitron`.
- Training pairs: partly MS MARCO and mMARCO, which allow non-commercial research only.
  A commercial version needs different training data.
