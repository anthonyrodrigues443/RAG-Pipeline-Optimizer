"""Chunking strategies benchmarked in Phase 1.

Each function takes raw document text and a HuggingFace tokenizer (for token-accurate
sizing) and returns a list of chunk strings. Phase 1 finding: with a 256-token encoder,
nominal chunk sizes above the window are statistically equal to no chunking.
"""
from __future__ import annotations

from typing import Callable, Dict, List


def _tok_len(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_fixed(text: str, tokenizer, size: int, overlap_ratio: float = 0.15) -> List[str]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]
    step = max(1, int(size * (1 - overlap_ratio)))
    out: List[str] = []
    for s in range(0, len(ids), step):
        piece = ids[s : s + size]
        if piece:
            out.append(tokenizer.decode(piece))
        if s + size >= len(ids):
            break
    return out


def chunk_recursive(text: str, tokenizer, size: int = 256) -> List[str]:
    seps = ["\n\n", "\n", ". ", " "]

    def split(t: str, depth: int = 0) -> List[str]:
        if _tok_len(t, tokenizer) <= size or depth >= len(seps):
            return [t]
        parts, cur = [], ""
        for p in t.split(seps[depth]):
            cand = (cur + seps[depth] + p) if cur else p
            if _tok_len(cand, tokenizer) <= size:
                cur = cand
            else:
                if cur:
                    parts.append(cur)
                if _tok_len(p, tokenizer) <= size:
                    cur = p
                else:
                    parts.extend(split(p, depth + 1))
                    cur = ""
        if cur:
            parts.append(cur)
        return [x for x in parts if x.strip()]

    return split(text) or [text]


def chunk_sentence(text: str) -> List[str]:
    from nltk.tokenize import sent_tokenize

    s = [x for x in sent_tokenize(text) if x.strip()]
    return s or [text]


def build_chunkers(tokenizer) -> Dict[str, Callable[[str], List[str]]]:
    """Return the named chunker registry used in the Phase 1 ablation."""
    return {
        "doc": lambda t: [t],
        "fixed_128": lambda t: chunk_fixed(t, tokenizer, 128),
        "fixed_256": lambda t: chunk_fixed(t, tokenizer, 256),
        "fixed_512": lambda t: chunk_fixed(t, tokenizer, 512),
        "fixed_1024": lambda t: chunk_fixed(t, tokenizer, 1024),
        "recursive_256": lambda t: chunk_recursive(t, tokenizer, 256),
        "sentence": chunk_sentence,
    }
