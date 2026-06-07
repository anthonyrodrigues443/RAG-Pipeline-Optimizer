"""Generation-scorecard aggregation — the pure replay logic behind the Phase-7 end-to-end
table. Synthetic gen+judge records, no caches, no I/O."""
from src.evaluate import _looks_refused, mean_metrics, score_generation_records


def test_mean_metrics_skips_none():
    assert mean_metrics([1.0, None, 0.0]) == 0.5
    assert mean_metrics([None, None]) is None


def test_looks_refused_detects_refusal_phrases():
    assert _looks_refused("The context does not contain the answer.")
    assert _looks_refused("I cannot answer based on this.")
    assert not _looks_refused("Freelance income is taxed as self-employment income [d1].")


def test_score_generation_records_aggregates_per_condition():
    gen = {
        "haiku::q1::strong": {"qid": "q1", "cond": "strong",
                              "answer": "grounded answer [d1]", "ctx_ids": ["A", "B"]},
        "haiku::q2::strong": {"qid": "q2", "cond": "strong",
                              "answer": "context does not contain it", "ctx_ids": ["C", "D"]},
        "haiku::q1::closed_book": {"qid": "q1", "cond": "closed_book",
                                   "answer": "from memory", "ctx_ids": []},
    }
    judge = {
        "haiku::q1::strong": {"parsed": {"claims": [{"supported_by_context": True}],
                                         "correctness": "SUPPORTED"}},
        "haiku::q2::strong": {"parsed": {"claims": [{"supported_by_context": False}],
                                         "correctness": "CONTRADICT"}},
        "haiku::q1::closed_book": {"parsed": {"claims": [], "correctness": "PARTIAL"}},
    }
    out = score_generation_records(gen, judge)

    # strong: correctness mean of (1.0 SUPPORTED, 0.0 CONTRADICT) = 0.5
    assert abs(out["strong"]["correctness"] - 0.5) < 1e-9
    # faithfulness mean of (1.0, 0.0) = 0.5
    assert abs(out["strong"]["faithfulness"] - 0.5) < 1e-9
    # one of the two strong answers reads as a refusal
    assert abs(out["strong"]["refused"] - 0.5) < 1e-9
    assert out["strong"]["n"] == 2

    # closed-book: PARTIAL -> 0.5 correctness, no claims -> faithfulness None
    assert abs(out["closed_book"]["correctness"] - 0.5) < 1e-9
    assert out["closed_book"]["faithfulness"] is None


def test_score_records_parses_raw_when_parsed_missing():
    gen = {"haiku::q1::strong": {"qid": "q1", "cond": "strong",
                                 "answer": "ans [d1]", "ctx_ids": ["A"]}}
    judge = {"haiku::q1::strong": {"raw": '{"claims":[{"supported_by_context":true}],'
                                          '"correctness":"SUPPORTED"}'}}
    out = score_generation_records(gen, judge)
    assert out["strong"]["correctness"] == 1.0
    assert out["strong"]["faithfulness"] == 1.0
