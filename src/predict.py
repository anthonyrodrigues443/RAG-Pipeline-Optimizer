"""Single-query inference over the optimal RAG pipeline.

    python -m src.predict "How is freelance income taxed?"
    python -m src.predict --condition strong --qid 2498
    python -m src.predict --condition adversarial --qid 2498   # watch the poisoning

Loads the FiQA corpus + its cached E5 embeddings, runs retrieve→(HyDE×N)→generate with
citation enforcement, and prints the retrieved passages, the grounded answer, and the
citation-grounding score. Free-text queries support HyDE×N / strong / closed-book; the
``oracle`` and ``adversarial`` conditions need ``--qid`` (they read the gold labels).
"""
from __future__ import annotations

import argparse
import os

from .pipeline import ContextCondition, RAGPipeline, load_corpus

EMB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data/processed/emb_cache")
META_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data/processed/meta")


def main():
    ap = argparse.ArgumentParser(description="RAG single-query inference")
    ap.add_argument("query", nargs="?", default=None, help="free-text question")
    ap.add_argument("--qid", default=None, help="use a FiQA query-id (enables oracle/adversarial)")
    ap.add_argument("--condition", default="hydeN",
                    choices=[c.value for c in ContextCondition])
    ap.add_argument("--corpus", default="fiqa")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--hyde-n", type=int, default=4)
    args = ap.parse_args()

    if not args.query and not args.qid:
        ap.error("provide a free-text query or --qid")

    print(f"loading {args.corpus} corpus + E5 cache (first run downloads BEIR)...", flush=True)
    corpus = load_corpus(args.corpus, EMB_DIR, META_DIR, domain="personal finance and investing")
    pipe = RAGPipeline(corpus, gen_model=args.model, hyde_n=args.hyde_n)

    cond = ContextCondition(args.condition)
    key = args.qid if args.qid else args.query
    is_qid = args.qid is not None
    res = pipe.answer(key, condition=cond, is_qid=is_qid)

    print("\n" + "=" * 78)
    print(f"Q: {res.question}")
    print(f"condition={res.condition}  |  model={args.model}  |  latency={res.latency_s}s")
    if res.hyde_passages:
        print(f"\nHyDE hypotheticals generated: {len(res.hyde_passages)}")
        for i, p in enumerate(res.hyde_passages, 1):
            print(f"  H{i}: {p[:120]}")
    print("\nRetrieved context:")
    if not res.context:
        print("  (closed-book — no retrieval)")
    for lab, did, txt in res.context:
        print(f"  [{lab}] doc={did}: {txt[:130]}...")
    print("\nAnswer:")
    print(f"  {res.answer}")
    cg = res.citation_grounding
    print(f"\ncitation grounding: {cg:.2f}" if cg is not None else "\ncitation grounding: (no citations)")
    print("=" * 78)


if __name__ == "__main__":
    main()
