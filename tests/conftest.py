"""Make ``import src...`` work no matter where pytest is invoked from, and provide the
lightweight fixtures (a word-level mock tokenizer, a tiny synthetic corpus) that let the
whole suite run on a clean checkout — no embeddings, no models, no network."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


class MockTokenizer:
    """Whitespace tokenizer: 'ids' are the words themselves, so encode/decode round-trip."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)


@pytest.fixture
def tok():
    return MockTokenizer()


@pytest.fixture
def synth_corpus(monkeypatch):
    """A 70-doc synthetic FiQA-shaped corpus with one query whose gold docs are at the top.

    70 docs lets the adversarial condition (E5 ranks 40-60) have something to slice. The
    retriever's encoders are monkeypatched to stay in this 8-dim space, so no real E5 model
    is ever loaded — the whole pipeline runs offline. The query vector points at D0 (so D0/D1
    rank first), and gold = {D0, D1}.
    """
    from src.pipeline import E5Retriever, RAGCorpus

    n, dim = 70, 8
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, dim)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    doc_ids = [f"D{i}" for i in range(n)]
    docs = {d: f"document {i} body text about finance topic {i}" for i, d in enumerate(doc_ids)}
    qid = "Q1"
    queries = {qid: "how is income taxed"}
    qvec = emb[0].copy()                       # ranks D0 first, D1 near top
    qrels = {qid: {"D0": 1, "D1": 1}}
    retr = E5Retriever(emb, doc_ids)

    # keep every query/passage encode in the synthetic 8-dim space (no model load)
    monkeypatch.setattr(retr, "encode_query",
                        lambda texts: np.tile(qvec, (len(texts), 1)).astype("float32"))
    monkeypatch.setattr(retr, "encode_passage",
                        lambda texts: np.tile(emb[1], (len(texts), 1)).astype("float32"))

    corpus = RAGCorpus(name="synth", docs=docs, queries=queries, qrels=qrels,
                       retriever=retr, domain="personal finance")
    return corpus, qid, qvec
