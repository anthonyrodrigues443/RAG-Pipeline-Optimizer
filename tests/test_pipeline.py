"""Pipeline orchestration — the Phase-7 production logic. The LLM is monkeypatched, the
retriever runs real NumPy over a tiny synthetic corpus, so the conditions, HyDE averaging,
context assembly and citation scoring are all exercised with no model and no CLI."""
import numpy as np
import pytest

from src import pipeline as P
from src.pipeline import (ContextCondition, E5Retriever, RAGPipeline,
                          generation_prompt, hyde_prompt, parse_hypotheticals)


# ---- prompts & parsers -----------------------------------------------------

def test_parse_hypotheticals_numbered():
    raw = "1. first passage.\n2. second passage.\n3. third passage."
    assert parse_hypotheticals(raw, 4) == ["first passage.", "second passage.", "third passage."]


def test_parse_hypotheticals_paragraph_fallback():
    raw = "para one here\n\npara two here"
    assert parse_hypotheticals(raw, 4) == ["para one here", "para two here"]


def test_parse_hypotheticals_caps_at_n():
    raw = "1. a\n2. b\n3. c\n4. d\n5. e"
    assert len(parse_hypotheticals(raw, 3)) == 3


def test_parse_hypotheticals_error_returns_empty():
    assert parse_hypotheticals("__ERROR__x", 4) == []


def test_generation_prompt_closed_book_has_no_context_block():
    p = generation_prompt("Q?", "finance", [])
    assert "Context:" not in p
    assert "own knowledge" in p


def test_generation_prompt_with_context_demands_citations():
    p = generation_prompt("Q?", "finance", [("d1", "X1", "text one"), ("d2", "X2", "text two")])
    assert "[d1]" in p and "[d2]" in p
    assert "ONLY the context" in p


def test_hyde_prompt_requests_n_passages():
    assert "4" in hyde_prompt("q", "finance", 4)


# ---- retriever -------------------------------------------------------------

def test_search_vec_ranks_by_cosine():
    emb = np.eye(4, dtype="float32")
    r = E5Retriever(emb, ["a", "b", "c", "d"])
    out = r.search_vec(np.array([1.0, 0.3, 0.0, 0.0]), k=3)
    assert out[0][0] == "a"
    assert out[0][1] >= out[1][1]


def test_hyde_vectors_average_is_normalised(monkeypatch):
    emb = np.eye(3, dtype="float32")
    r = E5Retriever(emb, ["a", "b", "c"])
    monkeypatch.setattr(r, "encode_query", lambda t: np.array([[1.0, 0.0, 0.0]], dtype="float32"))
    monkeypatch.setattr(r, "encode_passage", lambda t: np.array([[0.0, 1.0, 0.0]], dtype="float32"))
    v = r.hyde_vectors("q", ["hyp1"])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    # mean of (1,0,0) and (0,1,0) normalised -> (0.707, 0.707, 0)
    assert abs(v[0] - v[1]) < 1e-6 and v[2] == 0.0


def test_retriever_asserts_shape_mismatch():
    with pytest.raises(AssertionError):
        E5Retriever(np.zeros((3, 4), dtype="float32"), ["a", "b"])


# ---- end-to-end conditions (LLM monkeypatched) -----------------------------

@pytest.fixture
def patched_llm(monkeypatch):
    """call_claude returns a cited canned answer; HyDE prompt yields fixed hypotheticals."""
    def fake_call(prompt, model="haiku", timeout=90):
        if "hypothetical passages" in prompt:
            return "1. hypo one about tax.\n2. hypo two about tax."
        return "The answer is grounded in the context [d1]."
    monkeypatch.setattr(P.llm, "call_claude", fake_call)
    return fake_call


def test_closed_book_has_empty_context(synth_corpus, patched_llm):
    corpus, qid, _ = synth_corpus
    pipe = RAGPipeline(corpus)
    res = pipe.answer(qid, ContextCondition.CLOSED_BOOK, is_qid=True)
    assert res.context == []
    assert res.condition == "closed_book"


def test_oracle_returns_gold_docs(synth_corpus, patched_llm):
    corpus, qid, _ = synth_corpus
    pipe = RAGPipeline(corpus)
    res = pipe.answer(qid, ContextCondition.ORACLE, is_qid=True)
    got = {doc_id for _lab, doc_id, _txt in res.context}
    assert got == {"D0", "D1"}


def test_strong_returns_topk_and_labels(synth_corpus, patched_llm):
    corpus, qid, _ = synth_corpus
    pipe = RAGPipeline(corpus, top_k=5)
    res = pipe.answer(qid, ContextCondition.STRONG, is_qid=True)
    assert len(res.context) == 5
    assert [lab for lab, _, _ in res.context] == ["d1", "d2", "d3", "d4", "d5"]
    # D0 is the query target so it ranks first
    assert res.context[0][1] == "D0"


def test_hydeN_triggers_hyde_and_generates(synth_corpus, patched_llm):
    corpus, qid, _ = synth_corpus
    pipe = RAGPipeline(corpus, top_k=5)
    res = pipe.answer(qid, ContextCondition.HYDE_N, is_qid=True)
    assert len(res.hyde_passages) == 2          # from the fake HyDE response
    assert len(res.context) == 5
    assert res.citation_grounding == 1.0        # canned answer cites [d1], in range


def test_adversarial_excludes_gold_and_uses_mid_ranks(synth_corpus, patched_llm):
    corpus, qid, _ = synth_corpus
    pipe = RAGPipeline(corpus, top_k=5)
    res = pipe.answer(qid, ContextCondition.ADVERSARIAL, is_qid=True)
    got = {doc_id for _lab, doc_id, _txt in res.context}
    assert "D0" not in got and "D1" not in got   # gold excluded
    assert len(res.context) <= 5


def test_free_text_query_uses_text_not_qid(synth_corpus, patched_llm):
    corpus, _qid, _ = synth_corpus
    pipe = RAGPipeline(corpus)
    res = pipe.answer("how is income taxed", ContextCondition.STRONG, is_qid=False)
    assert res.question == "how is income taxed"
    assert len(res.context) > 0
