"""The optimal end-to-end RAG pipeline — the consolidated verdict of Phases 1-6.

Every design choice here is a *finding*, not a default:

* **Encoder:** E5-base-v2, **whole-document** (no chunking). Phase 1-2 law — with a 512-token
  encoder, splitting docs below the window is statistically equal to (often worse than) not
  splitting. So we don't.
* **Index:** exact cosine top-k via one NumPy matmul. Phase 3 — below ~40k vectors an ANN
  index (IVF/HNSW) *loses* nDCG to save sub-millisecond latency; the crossover where it pays
  is ~40k. FiQA (57k) is just past it, but Flat is still correct and trivially reproducible,
  so we default to Flat and document the HNSW switch.
* **Re-ranker:** **none.** Phase 4 — every cross-encoder tested *hurt* a strong E5 first stage;
  Phase 6 — it stays redundant even on HyDE×N's higher-recall candidates.
* **Query transform:** **HyDE×N** (average the query vector with N hypothetical-answer vectors)
  when an LLM budget exists. Phase 5 — the only transform that helped on all three corpora;
  single-HyDE and step-back flipped sign by domain.
* **Generator:** a cheap model (Haiku) with **citation enforcement** for the easy/mid majority;
  Phase 6 — frontier generators only win the hard tail, at 15-83x the cost.

The headline Phase-6 finding the pipeline is built to respect: retrieval gains *past
"good enough" barely move the answer, but wrong retrieval actively poisons it*. So the
contract here is retrieval **reliability**, not extra ranking precision — and the
``ContextCondition`` enum exists so the UI can demonstrate poisoning live.
"""
from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import llm
from .retrieval_eval import topk_search

E5_MODEL = "intfloat/e5-base-v2"
EMB_TAG = "E5-base-v2"          # cache-file prefix written by Phases 2-6
DEFAULT_HYDE_N = 4              # query + 4 hypotheticals → 5-vector mean (Phase-5 best)


class ContextCondition(str, enum.Enum):
    """The five retrieval conditions swept in Phase 6 — the UI's poisoning toggle."""
    HYDE_N = "hydeN"            # optimal pipeline: HyDE×N top-5
    STRONG = "strong"          # naive E5 top-5 (the baseline the project optimised against)
    ORACLE = "oracle"          # gold passages — the unreachable ceiling
    ADVERSARIAL = "adversarial"  # plausible-but-wrong (E5 ranks 40-60) — the poisoning demo
    CLOSED_BOOK = "closed_book"  # no retrieval at all


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class E5Retriever:
    """Dense exact-search retriever over pre-encoded, L2-normalised E5 document vectors.

    The encoder is lazy-loaded only when a query actually needs encoding (so importing the
    pipeline, or running it over a cached query vector, never pays the model-load cost).
    """

    def __init__(self, doc_emb: np.ndarray, doc_ids: List[str]):
        assert doc_emb.shape[0] == len(doc_ids), (doc_emb.shape, len(doc_ids))
        self.doc_emb = doc_emb.astype("float32")
        self.doc_ids = doc_ids
        self._encoder = None

    @property
    def n_docs(self) -> int:
        return self.doc_emb.shape[0]

    def _load_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(E5_MODEL, device="cpu")
            m.max_seq_length = 512
            self._encoder = m
        return self._encoder

    def encode_query(self, texts: List[str]) -> np.ndarray:
        """E5 requires the ``query:`` prefix on the query side; docs were encoded with ``passage:``."""
        enc = self._load_encoder()
        return enc.encode([f"query: {t}" for t in texts], normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def encode_passage(self, texts: List[str]) -> np.ndarray:
        enc = self._load_encoder()
        return enc.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False).astype("float32")

    def search_vec(self, qvec: np.ndarray, k: int = 100) -> List[Tuple[str, float]]:
        """Return ``[(doc_id, cosine)]`` ranked best→worst for one query vector."""
        qvec = np.asarray(qvec, dtype="float32").reshape(1, -1)
        sims, idx = topk_search(self.doc_emb, qvec, k)
        return [(self.doc_ids[j], float(s)) for j, s in zip(idx[0], sims[0])]

    def search(self, question: str, k: int = 100) -> List[Tuple[str, float]]:
        return self.search_vec(self.encode_query([question])[0], k)

    # -- HyDE×N -------------------------------------------------------------
    def hyde_vectors(self, question: str, hypotheticals: List[str]) -> np.ndarray:
        """Mean of the query vector and each hypothetical-passage vector, renormalised.

        This *is* the original HyDE recipe (Gao et al. 2022) extended to N hypotheticals:
        averaging in embedding space, not concatenating text. Phase-5 ablation confirmed the
        averaged form is what generalised; single-HyDE was a coin-flip by domain.
        """
        qv = self.encode_query([question])
        if hypotheticals:
            hv = self.encode_passage(hypotheticals)
            stack = np.vstack([qv, hv])
        else:
            stack = qv
        mv = stack.mean(axis=0)
        mv /= (np.linalg.norm(mv) + 1e-12)
        return mv.astype("float32")


# ---------------------------------------------------------------------------
# Corpus container
# ---------------------------------------------------------------------------

@dataclass
class RAGCorpus:
    name: str
    docs: Dict[str, str]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    retriever: E5Retriever
    domain: str = "the document collection"

    def doc_text(self, doc_id: str, words: int = 110) -> str:
        return " ".join(self.docs.get(doc_id, "").split()[:words])

    def gold_doc_ids(self, qid: str) -> List[str]:
        return [d for d, g in self.qrels.get(qid, {}).items()
                if g > 0 and d in self.docs]


def load_corpus(name: str, emb_dir: str, meta_dir: Optional[str] = None,
                domain: str = "the document collection") -> RAGCorpus:
    """Load a BEIR corpus + its cached E5 embeddings into a ready-to-query ``RAGCorpus``.

    Document/query *id order* is pinned from ``meta/{name}_doc_ids.json`` when present (that
    is the exact order the ``.npy`` cache was built in); otherwise it falls back to the BEIR
    load order, which is how the cache was originally written.
    """
    from collections import defaultdict

    from datasets import load_dataset
    import datasets as _ds
    _ds.disable_progress_bars()

    corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_t = load_dataset(f"BeIR/{name}-qrels", split="test")
    docs = {str(r["_id"]): ((r["title"] + ". " + r["text"]).strip() if r["title"] else r["text"])
            for r in corpus}
    qtext_all = {str(r["_id"]): r["text"] for r in queries}
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in qrels_t:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    qrels = dict(qrels)
    queries_d = {q: t for q, t in qtext_all.items() if q in qrels}

    doc_ids = _load_ids(meta_dir, f"{name}_doc_ids.json") or list(docs)
    D = np.load(os.path.join(emb_dir, f"{EMB_TAG}__{name}_docs.npy")).astype("float32")
    assert D.shape[0] == len(doc_ids), (name, D.shape, len(doc_ids))

    return RAGCorpus(name=name, docs=docs, queries=queries_d, qrels=qrels,
                     retriever=E5Retriever(D, doc_ids), domain=domain)


def _load_ids(meta_dir: Optional[str], fname: str) -> Optional[List[str]]:
    if not meta_dir:
        return None
    path = os.path.join(meta_dir, fname)
    if os.path.exists(path):
        import json
        return [str(x) for x in json.load(open(path))]
    return None


# ---------------------------------------------------------------------------
# Prompts (shared by pipeline + Streamlit + frontier comparison)
# ---------------------------------------------------------------------------

def hyde_prompt(question: str, domain: str, n: int) -> str:
    return (
        f"You are a {domain} expert. Write {n} SHORT hypothetical passages (2-3 sentences each) "
        f"that would directly and factually answer the question below, as if quoted from an "
        f"authoritative document. Number them 1..{n}. Do not hedge, do not say 'I think'. "
        f"Just write the passages.\n\nQuestion: {question}"
    )


def parse_hypotheticals(raw: str, n: int) -> List[str]:
    """Split a numbered HyDE response into individual passages. Defensive: degrades to []."""
    if llm.is_error(raw):
        return []
    import re
    parts = re.split(r"\n\s*\d+[\.\):]\s*", "\n" + raw.strip())
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:  # model ignored numbering — fall back to paragraph split
        parts = [p.strip() for p in raw.split("\n\n") if p.strip()]
    return parts[:n]


def generation_prompt(question: str, domain: str,
                      context: List[Tuple[str, str, str]]) -> str:
    """Build the answer prompt. ``context`` = ``[(label, doc_id, text)]``; empty → closed-book."""
    if not context:
        return (f"You are a {domain} QA assistant. Answer the question in at most 3 sentences "
                f"using your own knowledge. Be specific and factual.\n\n"
                f"Question: {question}\n\nAnswer:")
    block = "\n".join(f"[{lab}] {txt}" for lab, _id, txt in context)
    return (f"You are a {domain} QA assistant. Answer the question in at most 3 sentences using "
            f"ONLY the context below. Cite the passages you use as [d1], [d2]. If the context "
            f"does not contain the answer, say so explicitly.\n\n"
            f"Context:\n{block}\n\nQuestion: {question}\n\nAnswer:")


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    question: str
    condition: str
    context: List[Tuple[str, str, str]]   # (label, doc_id, snippet)
    answer: str
    citation_grounding: Optional[float]
    latency_s: float = 0.0
    hyde_passages: List[str] = field(default_factory=list)
    retrieval: List[Tuple[str, float]] = field(default_factory=list)  # (doc_id, score) top-k


class RAGPipeline:
    """Orchestrates retrieve → build context → generate, with the Phase-6 condition switch."""

    def __init__(self, corpus: RAGCorpus, gen_model: str = "haiku",
                 hyde_n: int = DEFAULT_HYDE_N, top_k: int = 5, snippet_words: int = 110):
        self.corpus = corpus
        self.gen_model = gen_model
        self.hyde_n = hyde_n
        self.top_k = top_k
        self.snippet_words = snippet_words

    # -- retrieval per condition -------------------------------------------
    def _ranked_ids(self, qid_or_question: str, condition: ContextCondition,
                    is_qid: bool) -> Tuple[List[Tuple[str, float]], List[str]]:
        """Return (full ranked (id,score) list for display, ranked-id list for context build)."""
        c = self.corpus
        question = c.queries.get(qid_or_question, qid_or_question) if is_qid else qid_or_question

        if condition == ContextCondition.CLOSED_BOOK:
            return [], []
        if condition == ContextCondition.ORACLE and is_qid:
            gold = c.gold_doc_ids(qid_or_question)
            return [(d, 1.0) for d in gold], gold

        hyde_passages: List[str] = []
        if condition == ContextCondition.HYDE_N:
            raw = llm.call_claude(hyde_prompt(question, c.domain, self.hyde_n), self.gen_model)
            hyde_passages = parse_hypotheticals(raw, self.hyde_n)
            qvec = c.retriever.hyde_vectors(question, hyde_passages)
            ranked = c.retriever.search_vec(qvec, 100)
        else:  # STRONG and ADVERSARIAL both start from the naive E5 ranking
            ranked = c.retriever.search(question, 100)

        self._last_hyde = hyde_passages
        if condition == ContextCondition.ADVERSARIAL:
            gold = set(c.gold_doc_ids(qid_or_question)) if is_qid else set()
            picked = [(d, s) for d, s in ranked[40:60] if d not in gold][:self.top_k]
            return ranked, [d for d, _ in picked]
        return ranked, [d for d, _ in ranked[:self.top_k]]

    def build_context(self, ids: List[str]) -> List[Tuple[str, str, str]]:
        return [(f"d{i+1}", d, self.corpus.doc_text(d, self.snippet_words))
                for i, d in enumerate(ids)]

    def answer(self, query: str, condition: ContextCondition = ContextCondition.HYDE_N,
               is_qid: bool = False) -> RAGResult:
        """End-to-end: retrieve under ``condition`` → assemble context → generate an answer.

        ``is_qid=True`` treats ``query`` as a corpus query-id (enables the oracle/adversarial
        conditions, which need the gold labels); ``False`` treats it as free text.
        """
        import time
        self._last_hyde = []
        t0 = time.time()
        ranked, ids = self._ranked_ids(query, condition, is_qid)
        context = self.build_context(ids)
        question = self.corpus.queries.get(query, query) if is_qid else query
        prompt = generation_prompt(question, self.corpus.domain, context)
        ans = llm.call_claude(prompt, self.gen_model)
        dt = time.time() - t0
        return RAGResult(
            question=question, condition=condition.value, context=context, answer=ans,
            citation_grounding=llm.citation_grounding(ans, len(context)),
            latency_s=round(dt, 2), hyde_passages=getattr(self, "_last_hyde", []),
            retrieval=ranked[:max(self.top_k, 10)],
        )
