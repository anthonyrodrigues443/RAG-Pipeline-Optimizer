"""Metric correctness — hand-computed expectations for the TREC/BEIR harness used in every
phase. If these drift, every comparison table in the repo is wrong, so they are exact."""
import numpy as np

from src.retrieval_eval import dcg, evaluate, topk_search


def test_dcg_exponential_gain():
    # single relevant doc at rank 1: (2^1-1)/log2(2) = 1
    assert dcg([1]) == 1.0
    # rank 2: (2^1-1)/log2(3)
    assert abs(dcg([0, 1]) - 1 / np.log2(3)) < 1e-9
    assert dcg([]) == 0.0


def test_perfect_ranking_scores_one():
    qrels = {"q": {"a": 1, "b": 1}}
    run = {"q": ["a", "b", "c", "d"]}
    m = evaluate(run, qrels, ks=(2, 10))
    assert abs(m["ndcg@2"] - 1.0) < 1e-9
    assert abs(m["recall@2"] - 1.0) < 1e-9
    assert abs(m["mrr@2"] - 1.0) < 1e-9


def test_recall_partial_and_mrr_rank():
    qrels = {"q": {"a": 1, "b": 1}}
    run = {"q": ["x", "a", "y", "b"]}        # first relevant at rank 2
    m = evaluate(run, qrels, ks=(1, 3, 10))
    assert m["recall@1"] == 0.0
    assert abs(m["recall@3"] - 0.5) < 1e-9   # only 'a' in top-3 of 2 relevant
    assert abs(m["recall@10"] - 1.0) < 1e-9
    assert abs(m["mrr@3"] - 0.5) < 1e-9      # 1/2


def test_missing_query_scores_zero_not_dropped():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": ["a"]}                       # q2 absent from run
    m = evaluate(run, qrels, ks=(10,))
    # mean over BOTH queries: q1=1.0, q2=0.0 -> 0.5
    assert abs(m["ndcg@10"] - 0.5) < 1e-9


def test_duplicate_doc_ids_cannot_inflate_recall():
    qrels = {"q": {"a": 1}}
    run = {"q": ["a", "a", "a"]}              # malformed run with dupes
    m = evaluate(run, qrels, ks=(10,))
    assert m["recall@10"] == 1.0             # not 3.0


def test_topk_search_returns_sorted_neighbours():
    docs = np.eye(4, dtype="float32")        # orthonormal one-hot doc vectors
    q = np.array([[1.0, 0.5, 0.0, 0.0]], dtype="float32")
    q /= np.linalg.norm(q)
    sims, idx = topk_search(docs, q, k=2)
    assert idx[0, 0] == 0                     # doc 0 most similar
    assert idx[0, 1] == 1
    assert sims[0, 0] >= sims[0, 1]


def test_topk_clamps_k_to_corpus_size():
    docs = np.eye(3, dtype="float32")
    q = docs[:1]
    sims, idx = topk_search(docs, q, k=99)
    assert idx.shape[1] == 3
