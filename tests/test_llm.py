"""LLM harness parsers — pure string functions, fully testable with no CLI present."""
from src import llm


def test_is_error():
    assert llm.is_error(None)
    assert llm.is_error("__ERROR__timeout")
    assert not llm.is_error("a real answer")


def test_citation_grounding_in_range():
    assert llm.citation_grounding("uses [d1] and [d2]", n_ctx=3) == 1.0


def test_citation_grounding_out_of_range():
    # [d5] points past a 2-passage window
    assert llm.citation_grounding("see [d1] and [d5]", n_ctx=2) == 0.5


def test_citation_grounding_none_when_no_citations():
    assert llm.citation_grounding("no citations at all", n_ctx=3) is None


def test_parse_judge_fenced_json():
    raw = ('```json\n{"claims":[{"text":"x","supported_by_context":true},'
           '{"text":"y","supported_by_context":false}],'
           '"subquestions":["a","b"],"correctness":"SUPPORTED"}\n```')
    p = llm.parse_judge(raw)
    assert p["correctness"] == "SUPPORTED"
    assert len(p["claims"]) == 2
    assert p["subquestions"] == ["a", "b"]


def test_parse_judge_prose_wrapped():
    raw = 'Here is my evaluation: {"claims":[],"correctness":"UNKNOWN"} hope that helps'
    p = llm.parse_judge(raw)
    assert p["correctness"] == "UNKNOWN"
    assert p["claims"] == []


def test_parse_judge_malformed_returns_none():
    assert llm.parse_judge("not json at all") is None
    assert llm.parse_judge("__ERROR__x") is None
    assert llm.parse_judge('{"correctness": broken') is None


def test_parse_judge_coerces_unknown_correctness_label():
    p = llm.parse_judge('{"claims":[],"correctness":"GREAT"}')
    assert p["correctness"] == "UNKNOWN"


def test_faithfulness_fraction_of_supported_claims():
    p = {"claims": [{"supported_by_context": True}, {"supported_by_context": False},
                    {"supported_by_context": True}], "correctness": "PARTIAL"}
    assert abs(llm.faithfulness(p) - 2 / 3) < 1e-9


def test_faithfulness_none_when_no_claims():
    assert llm.faithfulness({"claims": [], "correctness": "UNKNOWN"}) is None
    assert llm.faithfulness(None) is None


def test_correctness_score_mapping():
    assert llm.CORRECTNESS_SCORE["SUPPORTED"] == 1.0
    assert llm.CORRECTNESS_SCORE["PARTIAL"] == 0.5
    assert llm.CORRECTNESS_SCORE["CONTRADICT"] == 0.0
    assert llm.CORRECTNESS_SCORE["UNKNOWN"] == 0.0
