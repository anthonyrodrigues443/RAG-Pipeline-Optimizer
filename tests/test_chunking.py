"""Chunkers — the Phase-1 registry. Behaviour is verified with a word-level mock tokenizer
so the test needs no HF download."""
from src.chunking import build_chunkers, chunk_fixed, chunk_recursive, chunk_sentence


def test_doc_chunker_is_identity():
    chunkers = build_chunkers(None)
    assert chunkers["doc"]("hello world") == ["hello world"]


def test_fixed_short_text_returns_single_chunk(tok):
    out = chunk_fixed("one two three", tok, size=10)
    assert out == ["one two three"]


def test_fixed_splits_with_overlap(tok):
    text = " ".join(f"w{i}" for i in range(20))
    out = chunk_fixed(text, tok, size=8, overlap_ratio=0.25)
    assert len(out) > 1
    # every chunk is at most `size` tokens
    assert all(len(c.split()) <= 8 for c in out)
    # overlap: step = 8*(1-0.25)=6, so chunk2 starts at w6
    assert out[1].split()[0] == "w6"


def test_recursive_respects_size_budget(tok):
    text = "a b c. d e f. g h i. j k l."
    out = chunk_recursive(text, tok, size=4)
    assert len(out) >= 2
    assert all(len(c.split()) <= 4 for c in out)


def test_recursive_short_text_single_chunk(tok):
    assert chunk_recursive("a b c", tok, size=10) == ["a b c"]


def test_sentence_chunker_splits_sentences():
    import pytest
    pytest.importorskip("nltk")
    try:
        out = chunk_sentence("First sentence. Second sentence. Third one.")
    except LookupError:
        pytest.skip("nltk punkt/punkt_tab data unavailable — offline test run")
    assert len(out) == 3


def test_registry_has_all_phase1_strategies(tok):
    reg = build_chunkers(tok)
    assert set(reg) == {"doc", "fixed_128", "fixed_256", "fixed_512",
                        "fixed_1024", "recursive_256", "sentence"}
