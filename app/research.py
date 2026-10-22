"""research(frage): agentische Recherche von Anfang bis Ende mit Nemotron.

Ablauf (jeder Schritt meldet sich ueber on_step, die Demo zeigt das live):
  1. plan      Nemotron bildet mindestens MIN_VARIANTS Suchvarianten (Synonyme, andere Sprache)
  2. search    alle Varianten parallel, Treffer nach Seite zusammengefuehrt
  3. read      die besten Seiten im Volltext laden
  4. rerank    Standard: mMiniLM sortiert alle Abschnitte, Nemotron-Reranker die Top 5 neu
  5. draft     Nemotron (Token Factory) schreibt Aussagen, jede mit woertlichen Zitaten
  6. finalize  Code prueft: jedes Zitat steht woertlich im Quelltext, jede Aussage hat
               MIN_SOURCES Quellen von verschiedenen Seiten. Sonst Ablehnung und eine
               Korrekturrunde mit den Gruenden an Nemotron.

Quellen (Argument quellen): "web" (eigenes Such-Backend, Standard), "tavily" (oeffentliches Web
ueber die Tavily-API) und "privat" (lokaler Ordner als Intranet, app/private.py). Mehrere Quellen
werden gemeinsam sortiert; jede Quelle traegt das Label privat oder oeffentlich, die 2-Quellen-Regel
gilt unveraendert.

Ohne NEBIUS_API_KEY laufen 1 und 5 als Stub (Heuristik bzw. extraktiv), der Rest echt.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import nemotron

MIN_VARIANTS = 4
MAX_VARIANTS = 6
MIN_SOURCES = 2
HITS_PER_VARIANT = 8
PAGES_TO_READ = 10
READ_DEADLINE_S = 12.0
SOURCES_FOR_LLM = 10
CHUNKS_PER_PAGE = 3  # BM25-Vorauswahl je Seite, bevor der Reranker rechnet
# Nemotron-Rerank kostet auf dem M3 unter Last ~0,5 s je Abschnitt: Pool klein halten
RERANK_POOL = int(os.environ.get("RESEARCH_RERANK_POOL", "20"))
CHARS_PER_SOURCE = 1800
REPAIR_ROUNDS = 1
L_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # Searchify L, ~20 ms je Anfrage
L_POOL = 40  # Abschnitte, die mMiniLM bewertet (billig)
NEMOTRON_TOP = 5  # nur diese Quellen sortiert Nemotron im Modus "l+nemotron" neu
# bm25: ohne Modell (reine Python-Funktion), fuer Rechner, auf denen kein Reranker laufen soll
RERANK_MODES = ("l+nemotron", "nemotron", "l", "bm25")
DEFAULT_RERANK = os.environ.get("RESEARCH_RERANK", "l+nemotron")
QUELLEN = ("web", "tavily", "privat")
DEFAULT_QUELLEN = ("web",)
TAVILY_QUERIES = 2  # Suchvarianten, die an Tavily gehen (jede Anfrage kostet Guthaben)
TAVILY_RESULTS = 5
PRIVATE_DOCS = 5
MIX_MIN = 3  # bei mehreren Quellen: so viele Plaetze je Herkunft unter den SOURCES_FOR_LLM

Step = Callable[[str, Dict], None]


class Rejected(ValueError):
    """finalize hat keine einzige belegte Aussage gefunden."""

    def __init__(self, message: str, problems: List[str]):
        super().__init__(message)
        self.problems = problems


@dataclass
class Source:
    n: int
    url: str
    title: str
    site: str
    excerpt: str  # was das Modell sieht
    full_text: str  # wogegen Zitate geprueft werden
    score: float = 0.0
    stage: str = ""  # welcher Reranker die Reihenfolge bestimmt hat
    origin: str = "web"  # web | tavily | privat

    @property
    def label(self) -> str:
        return "privat" if self.origin == "privat" else "öffentlich"


@dataclass
class Claim:
    text: str
    citations: List[Dict]  # [{"source": n, "quote": "..."}]


@dataclass
class Result:
    question: str
    answer: str
    claims: List[Claim]
    sources: List[Source]
    rejected: List[str] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)
    mode: str = "nemotron"
    rerank: str = DEFAULT_RERANK
    timings: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        used = {c["source"] for cl in self.claims for c in cl.citations}
        return {
            "question": self.question,
            "answer": self.answer,
            "claims": [{"text": c.text, "citations": c.citations} for c in self.claims],
            "sources": [{"n": s.n, "url": s.url, "title": s.title, "label": s.label}
                        for s in self.sources if s.n in used],
            "rejected": self.rejected,
            "variants": self.variants,
            "mode": self.mode,
            "model": nemotron.model_name() if self.mode == "nemotron" else None,
            "rerank": self.rerank,
            "timings_s": self.timings,
        }


# --- Text-Normalisierung fuer den Zitat-Abgleich -------------------------------------

_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                         "–": "-", "—": "-", " ": " ", "­": ""})


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_QUOTES).lower()
    return re.sub(r"\s+", " ", text).strip()


def site_of(url: str) -> str:
    from .rerank import _site  # noqa: PLC0415

    return _site(url)


# --- 6. finalize: die Quellenpruefung im Code ----------------------------------------

def finalize(draft: Dict, sources: List[Source], min_sources: int = MIN_SOURCES) -> tuple:
    """Prueft einen Entwurf {"claims": [{"text", "citations": [{"source", "quote"}]}]}.

    Ein Zitat gilt nur, wenn es (normalisiert) woertlich im Volltext seiner Quelle
    steht und mindestens 4 Woerter hat. Eine Aussage gilt nur mit min_sources
    gueltigen Zitaten von verschiedenen Seiten. Rueckgabe (gueltige Claims, Gruende
    fuer abgelehnte). Keine gueltige Aussage -> Rejected.
    """
    by_n = {s.n: s for s in sources}
    norm_cache: Dict[int, str] = {}
    accepted: List[Claim] = []
    problems: List[str] = []
    for raw in (draft or {}).get("claims") or []:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        good: List[Dict] = []
        sites = set()
        for cit in raw.get("citations") or []:
            try:
                n = int(cit.get("source"))
            except (TypeError, ValueError):
                problems.append(f"'{text[:60]}': Quellennummer fehlt")
                continue
            quote = str(cit.get("quote", "")).strip()
            src = by_n.get(n)
            if src is None:
                problems.append(f"'{text[:60]}': Quelle [{n}] gibt es nicht")
                continue
            if len(quote.split()) < 4:
                problems.append(f"'{text[:60]}': Zitat aus [{n}] zu kurz")
                continue
            if n not in norm_cache:
                norm_cache[n] = normalize(src.full_text)
            if normalize(quote) not in norm_cache[n]:
                problems.append(f"'{text[:60]}': Zitat steht nicht in Quelle [{n}]")
                continue
            if src.site in sites:
                continue  # zweites Zitat derselben Seite zaehlt nicht als zweite Quelle
            sites.add(src.site)
            good.append({"source": n, "quote": quote})
        if len(good) >= min_sources:
            accepted.append(Claim(text=text, citations=good))
        else:
            problems.append(f"'{text[:60]}': nur {len(good)} gepruefte Quelle(n), {min_sources} noetig")
    if not accepted:
        raise Rejected("Keine Aussage mit geprueften Zitaten aus genug Quellen", problems)
    return accepted, problems


def render(claims: List[Claim]) -> str:
    return " ".join(
        f"{c.text} " + "".join(f"[{cit['source']}]" for cit in c.citations) for c in claims
    ).strip()


# --- 1. plan -------------------------------------------------------------------------

_STOP = set(
    "was wer wie wo wann warum welche welcher welches ist sind war hat haben der die das ein eine "
    "und oder in im am an auf fuer für mit von zu den dem des what who how where when why which "
    "is are was were does do did the a an of in on for to and or with".split()
)


def _keywords(question: str) -> str:
    words = re.findall(r"[\w\-\.]+", question)
    return " ".join(w for w in words if w.lower() not in _STOP)


def heuristic_variants(question: str) -> List[str]:
    from . import lang as lang_mod  # noqa: PLC0415

    kw = _keywords(question) or question
    de = lang_mod.detect(question) == "de"
    out = [question, kw, f"{kw} {'offizielle Angaben' if de else 'official'}",
           f"{kw} {'Fakten' if de else 'facts'}", f'"{kw}"']
    return _dedup(out)[:MAX_VARIANTS]


def _dedup(items: List[str]) -> List[str]:
    seen, out = set(), []
    for q in items:
        q = " ".join(str(q).split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


PLAN_PROMPT = (
    "You plan web searches for a research agent. Return ONLY a JSON array of {n} to {m} "
    "search queries for the question. Mix: the plain question, a keyword version, synonyms or "
    "the technical term, an English version if the question is not English (and German if it is "
    "English), and one query that targets an official or primary source. No explanations."
)


def plan(question: str, use_llm: bool) -> List[str]:
    variants: List[str] = []
    if use_llm:
        try:
            got = nemotron.chat_json([
                {"role": "system", "content": PLAN_PROMPT.format(n=MIN_VARIANTS, m=MAX_VARIANTS)},
                {"role": "user", "content": question},
            ], max_tokens=600)
            if isinstance(got, dict):
                got = got.get("queries") or next(iter(got.values()), [])
            variants = [str(q) for q in got if isinstance(q, str)]
        except Exception:  # noqa: BLE001 - Plan darf nie die Recherche kippen
            variants = []
    # Mindestzahl erzwingen: fehlende Varianten aus der Heuristik auffuellen
    variants = _dedup([question] + variants + heuristic_variants(question))
    return variants[:max(MIN_VARIANTS, min(len(variants), MAX_VARIANTS))]


# --- 2.-4. search, read, rerank ------------------------------------------------------

def search_all(variants: List[str]) -> List[Dict]:
    """Alle Varianten parallel; Treffer je kanonischer URL, sortiert nach Zahl der
    Varianten, die sie fanden, dann bester Platz."""
    from . import lang as lang_mod  # noqa: PLC0415
    from . import rerank as rerank_mod  # noqa: PLC0415
    from . import search as search_mod  # noqa: PLC0415

    def one(q: str) -> List[Dict]:
        try:
            return search_mod.web_search(q, HITS_PER_VARIANT, search_depth="basic",
                                         language=lang_mod.detect(q))
        except Exception:  # noqa: BLE001 - eine ausgefallene Variante ist kein Abbruch
            return []

    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        lists = list(pool.map(one, variants))
    merged: Dict[str, Dict] = {}
    for hits in lists:
        for rank, hit in enumerate(hits):
            url = hit.get("url", "")
            if not url.startswith(("http://", "https://")):
                continue  # z. B. relative Werbe-Weiterleitungen von ddgs ("/clev?event=...")
            key = rerank_mod.canonical_url(url)
            item = merged.setdefault(key, {**hit, "votes": 0, "best_rank": rank})
            item["votes"] += 1
            item["best_rank"] = min(item["best_rank"], rank)
    return sorted(merged.values(), key=lambda h: (-h["votes"], h["best_rank"]))


def tavily_all(variants: List[str]) -> List[Dict]:
    """Die ersten TAVILY_QUERIES Varianten parallel an Tavily; Tavily liefert den Seitentext
    gleich mit, ein eigener Lese-Schritt entfaellt."""
    from . import rerank as rerank_mod  # noqa: PLC0415
    from . import tavily_search  # noqa: PLC0415

    if not tavily_search.configured():
        return []

    def one(q: str) -> List[Dict]:
        try:
            return tavily_search.search(q, TAVILY_RESULTS)
        except Exception:  # noqa: BLE001 - eine ausgefallene Anfrage ist kein Abbruch
            return []

    queries = variants[:TAVILY_QUERIES]
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        lists = list(pool.map(one, queries))
    merged: Dict[str, Dict] = {}
    for hits in lists:
        for hit in hits:
            merged.setdefault(rerank_mod.canonical_url(hit["url"]), hit)
    return list(merged.values())


def gather(question: str, variants: List[str], quellen) -> tuple:
    """Seiten aus allen gewaehlten Quellen, je mit full_text und origin. Private Dokumente nur
    aus dem erkannten Regal (falls es Regale gibt). Rueckgabe (Seiten, Zahl je Herkunft, Regal)."""
    from . import private  # noqa: PLC0415

    jobs = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        if "web" in quellen:
            jobs["web"] = pool.submit(lambda: [{**p, "origin": "web"} for p in read(search_all(variants))])
        if "tavily" in quellen:
            jobs["tavily"] = pool.submit(tavily_all, variants)
        shelf = private.detect_shelf([question]) if "privat" in quellen else None
        if "privat" in quellen:
            jobs["privat"] = pool.submit(private.search, [question] + variants, PRIVATE_DOCS, None, shelf)
        got = {name: job.result() for name, job in jobs.items()}
    pages = [p for name in QUELLEN for p in got.get(name, [])]
    return pages, {name: len(got[name]) for name in got}, shelf


def read(hits: List[Dict]) -> List[Dict]:
    """Volltext der besten Seiten, hoechstens zwei je Seite (Vielfalt fuer 2 Quellen)."""
    from .pages import read_pages  # noqa: PLC0415

    picked, per_site = [], {}
    for h in hits:
        site = site_of(h["url"])
        if per_site.get(site, 0) >= 2:
            continue
        per_site[site] = per_site.get(site, 0) + 1
        picked.append(h)
        if len(picked) >= PAGES_TO_READ:
            break
    docs, _failed = read_pages([h["url"] for h in picked], deadline_s=READ_DEADLINE_S)
    text_by_url = {d.get("url"): d.get("content") or "" for d in docs}
    out = []
    for h in picked:
        full = text_by_url.get(h["url"], "")
        out.append({**h, "full_text": full or h.get("content", "")})
    return out



_L = None
_L_LOCK = threading.Lock()


def _l_model():
    global _L
    with _L_LOCK:
        if _L is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: PLC0415

            from .rerank import _pick_device  # noqa: PLC0415

            device = _pick_device()
            tok = AutoTokenizer.from_pretrained(L_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(L_MODEL).to(device).eval()
            _L = (tok, model, device)
    return _L


def l_scores(question: str, texts: List[str]) -> List[float]:
    """Roh-Logits von mMiniLM (Query-Text-Paare)."""
    import torch  # noqa: PLC0415

    from .rerank import _INFER_LOCK  # noqa: PLC0415 (MPS: Forward-Aufrufe nacheinander)

    tok, model, device = _l_model()
    with _INFER_LOCK, torch.inference_mode():
        enc = tok([question] * len(texts), texts, padding=True, truncation="only_second",
                  max_length=256, return_tensors="pt").to(device)
        return model(**enc).logits.view(-1).float().cpu().tolist()


def _chunks(question: str, pages: List[Dict], pool: int) -> List[Dict]:
    from . import rerank as rerank_mod  # noqa: PLC0415
    from .extract import chunk_text  # noqa: PLC0415

    chunks = []
    for i, p in enumerate(pages):
        own = [{"url": p["url"], "title": p.get("title", ""), "content": piece, "page": i}
               for piece in chunk_text(p["full_text"], 120, 30) or [p.get("content", "")] if piece]
        chunks.extend(rerank_mod.best_chunks(question, own, CHUNKS_PER_PAGE))
    return rerank_mod.best_chunks(question, chunks, pool)


def _mix(order: List[int], pages: List[Dict], k: int) -> List[int]:
    """Bei mehreren Herkuenften: jede bekommt bis zu MIX_MIN der k Plaetze, der Rest nach Rang.
    Die Reihenfolge innerhalb der Auswahl bleibt die des Rankings."""
    origins = {pages[i].get("origin", "web") for i in order}
    if len(origins) < 2:
        return order[:k]
    keep = set()
    for origin in origins:
        keep.update([i for i in order if pages[i].get("origin", "web") == origin][:MIX_MIN])
    for i in order:
        if len(keep) >= k:
            break
        keep.add(i)
    return [i for i in order if i in keep][:k]


def _per_page(ranked: List[Dict]) -> Dict[int, List[Dict]]:
    """Je Seite die zwei besten Abschnitte, Seiten in Reihenfolge ihres besten Abschnitts."""
    best: Dict[int, List[Dict]] = {}
    for ch in ranked:
        if len(best.setdefault(ch["page"], [])) < 2:
            best[ch["page"]].append(ch)
    return best


def rank(question: str, pages: List[Dict], mode: str = DEFAULT_RERANK) -> List[Source]:
    """Quellen sortieren. Modus "nemotron": llama-nemotron-rerank-1b-v2 bewertet alle
    Abschnitte. Modus "l+nemotron": mMiniLM bewertet alle, Nemotron sortiert nur den
    besten Abschnitt der Top NEMOTRON_TOP Quellen neu (Plan B aus der README).
    Modus "l": nur mMiniLM (Searchify L), zum Vergleich. Modus "bm25": ohne Modell.
    Private und oeffentliche Seiten werden gemeinsam sortiert (_mix haelt beide im Bild)."""
    from . import rerank as rerank_mod  # noqa: PLC0415

    if mode == "nemotron":
        best = _per_page(rerank_mod.rerank(question, _chunks(question, pages, RERANK_POOL)))
        order = sorted(best, key=lambda i: -best[i][0]["score"])
        stage = {i: "nemotron" for i in order}
    elif mode == "bm25":
        chunks = _chunks(question, pages, L_POOL)
        raw = rerank_mod.lexical_scores(question, [f"{c['title']}. {c['content']}" for c in chunks])
        top = max(raw, default=0.0) or 1.0
        for ch, r in zip(chunks, raw):
            ch["score"] = round(r / top, 4)
        best = _per_page(sorted(chunks, key=lambda c: -c["score"]))
        order = sorted(best, key=lambda i: -best[i][0]["score"])
        stage = {i: "bm25" for i in order}
    else:
        chunks = _chunks(question, pages, L_POOL)
        raw = l_scores(question, [f"{c['title']}. {c['content']}" for c in chunks]) if chunks else []
        for ch, r in zip(chunks, raw):
            ch["score"] = round(1 / (1 + math.exp(-r)), 4)
        best = _per_page(sorted(chunks, key=lambda c: -c["score"]))
        order = sorted(best, key=lambda i: -best[i][0]["score"])
        head = order[:NEMOTRON_TOP] if mode == "l+nemotron" else []
        renked = rerank_mod.rerank(question, [best[i][0] for i in head]) if head else []
        for ch in renked:
            best[ch["page"]][0] = {**best[ch["page"]][0], "score": ch["score"]}
        order = [ch["page"] for ch in renked] + order[len(head):]
        stage = {i: ("nemotron" if i in head else "L") for i in order}

    sources = []
    for n, i in enumerate(_mix(order, pages, SOURCES_FOR_LLM), start=1):
        p = pages[i]
        excerpt = "\n...\n".join(c["content"] for c in best[i])[:CHARS_PER_SOURCE]
        sources.append(Source(n=n, url=p["url"], title=p.get("title", ""), site=site_of(p["url"]),
                              excerpt=excerpt, full_text=p["full_text"], score=best[i][0]["score"],
                              stage=stage[i], origin=p.get("origin", "web")))
    return sources


# --- 5. draft ------------------------------------------------------------------------

DRAFT_PROMPT = (
    "You are a careful research agent. Answer the question ONLY from the numbered sources. "
    "Return ONLY JSON: {\"claims\": [{\"text\": \"one factual sentence in the language of the "
    "question\", \"citations\": [{\"source\": <number>, \"quote\": \"exact words copied from that "
    "source\"}]}]}. Rules: every claim needs at least 2 citations from DIFFERENT websites; each "
    "quote is copied character for character from the source text (8 to 30 words, no ellipsis, "
    "no translation); if only one source supports something, leave it out; 2 to 5 claims; "
    "if the sources do not answer the question, return {\"claims\": []}. Sources marked PRIVATE are "
    "internal company documents, PUBLIC ones are from the open web. When both kinds are present, "
    "write comparison claims: each claim says what the internal document requires AND what the "
    "public source says (for example 'Our policy requires X, while the law/guideline requires Y'), "
    "citing one PRIVATE and one PUBLIC source; an internal fact alone with only one internal "
    "document behind it will be rejected."
)


def _source_block(sources: List[Source]) -> str:
    return "\n\n".join(f"[{s.n}] {'PRIVATE' if s.origin == 'privat' else 'PUBLIC'} {s.title} ({s.site})\n{s.excerpt}"
                       for s in sources)


def draft_llm(question: str, sources: List[Source], feedback: Optional[List[str]] = None) -> Dict:
    user = f"Question: {question}\n\nSources:\n{_source_block(sources)}"
    if feedback:
        user += ("\n\nYour previous draft was rejected by the citation checker:\n- "
                 + "\n- ".join(feedback[:10])
                 + "\nFix it: copy quotes exactly from the source text above, use 2 different websites per claim.")
    got = nemotron.chat_json([
        {"role": "system", "content": DRAFT_PROMPT},
        {"role": "user", "content": user},
    ], max_tokens=2500)
    return got if isinstance(got, dict) else {"claims": got}


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"])")


def _sentences(text: str) -> List[str]:
    """Saetze fuer den Stub; Markdown-Menues und Links (Tavily-raw_content) zaehlen nicht."""
    return [s.strip() for s in _SENT.split(text or "")
            if 8 <= len(s.split()) <= 40 and not re.search(r"\]\(|#|\*|\|", s)]


def _toks(text: str) -> set:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3 and w not in _STOP}


def draft_stub(question: str, sources: List[Source]) -> Dict:
    """Ohne LLM: Satz aus einer Quelle nehmen und den aehnlichsten Satz einer anderen
    Seite als zweiten Beleg suchen. Rein extraktiv, damit finalize echt pruefen kann."""
    q = _toks(question)
    # nur Saetze, die vollstaendig im Quelltext stehen (der Auszug kann mitten im Satz enden)
    sents = [(s, sent) for s in sources for sent in _sentences(s.excerpt)
             if normalize(sent) in normalize(s.full_text)]
    sents.sort(key=lambda x: -len(q & _toks(x[1])))
    claims, used = [], set()
    for src, sent in sents:
        if len(claims) >= 3 or sent in used:
            continue
        a = _toks(sent)
        best, best_sim = None, 0.0
        for other, cand in sents:
            if other.site == src.site:
                continue
            b = _toks(cand)
            sim = len(a & b) / max(1, len(a | b))
            if sim > best_sim:
                best, best_sim = (other, cand), sim
        if best and best_sim >= 0.2:
            used.update({sent, best[1]})
            claims.append({"text": sent, "citations": [
                {"source": src.n, "quote": sent}, {"source": best[0].n, "quote": best[1]}]})
    return {"claims": claims}


# --- Gesamtablauf --------------------------------------------------------------------

def _noop(_step: str, _info: Dict) -> None:
    pass


def _warm_reranker(mode: str = DEFAULT_RERANK) -> None:
    from . import rerank as rerank_mod  # noqa: PLC0415

    if mode == "bm25":
        return

    def warm():
        if mode != "nemotron":
            try:
                l_scores("warm up", ["warm up"])
            except Exception:  # noqa: BLE001 - Fehler zeigt sich spaeter beim echten Aufruf
                pass
        if mode != "l":
            rerank_mod.warm_up()

    threading.Thread(target=warm, daemon=True).start()


def research(question: str, on_step: Step = _noop, rerank_mode: str = DEFAULT_RERANK,
             quellen=DEFAULT_QUELLEN) -> Result:
    question = " ".join(question.split())
    use_llm = nemotron.configured()
    mode = "nemotron" if use_llm else "stub"
    timings: Dict[str, float] = {}

    def timed(name: str, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        timings[name] = round(time.perf_counter() - t0, 2)
        return out

    if rerank_mode not in RERANK_MODES:
        raise ValueError(f"rerank_mode muss einer von {RERANK_MODES} sein")
    quellen = tuple(quellen)
    if not quellen or any(q not in QUELLEN for q in quellen):
        raise ValueError(f"quellen muss aus {QUELLEN} sein")
    _warm_reranker(rerank_mode)  # Modell laedt (~20 s), waehrend Suche und Lesen laufen
    on_step("start", {"question": question, "mode": mode, "quellen": list(quellen),
                      "model": nemotron.model_name() if use_llm else "Stub (kein NEBIUS_API_KEY)"})
    variants = timed("plan", plan, question, use_llm)
    on_step("plan", {"variants": variants})
    pages, per_origin, shelf = timed("search", gather, question, variants, quellen)
    if "privat" in quellen:
        on_step("regal", {"name": shelf})
    on_step("search", {"hits": len(pages), "per_origin": per_origin, "top": [p["url"] for p in pages[:5]]})
    on_step("read", {"pages": len(pages), "with_text": sum(1 for p in pages if len(p["full_text"]) > 500)})
    sources = timed("rerank", rank, question, pages, rerank_mode)
    on_step("rerank", {"mode": rerank_mode, "sources": [(s.n, s.site, s.score, s.stage) for s in sources]})

    feedback: Optional[List[str]] = None
    claims: List[Claim] = []
    problems: List[str] = []
    for attempt in range(REPAIR_ROUNDS + 1):
        t0 = time.perf_counter()
        try:
            draft = draft_llm(question, sources, feedback) if use_llm else draft_stub(question, sources)
        except Exception as exc:  # noqa: BLE001 - LLM weg: extraktiv weitermachen
            on_step("draft_error", {"error": type(exc).__name__})
            draft, mode = draft_stub(question, sources), "stub"
        timings[f"draft{attempt + 1}"] = round(time.perf_counter() - t0, 2)
        on_step("draft", {"attempt": attempt + 1, "claims": len(draft.get("claims") or [])})
        try:
            got, why = finalize(draft, sources)
        except Rejected as rej:
            got, why = [], rej.problems
        on_step("finalize", {"accepted": len(got), "rejected": why})
        if attempt == 0 or len(got) > len(claims):  # Korrektur zaehlt nur, wenn sie mehr belegt
            claims, problems = got, why
        if (claims and not problems) or mode == "stub":
            break
        feedback = why  # Korrekturrunde mit den Gruenden der Pruefung

    if claims:
        answer = render(claims)
    else:
        answer = "Keine Aussage ließ sich mit zwei geprüften Quellen belegen."
    result = Result(question=question, answer=answer, claims=claims, sources=sources,
                    rejected=problems, variants=variants, mode=mode, rerank=rerank_mode,
                    timings=timings)
    on_step("done", {"claims": len(claims)})
    return result
