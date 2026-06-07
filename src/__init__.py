"""RAG Pipeline Optimizer — reusable retrieval + chunking + evaluation library.

The research itself lives in `notebooks/` (Phases 1-6). This package holds the clean,
importable building blocks those notebooks call into, plus the Phase-7 production pipeline:

    from src.pipeline import RAGPipeline, load_corpus, ContextCondition

``llm``/``chunking``/``retrieval_eval`` are dependency-light and unit-tested on a clean
checkout; ``pipeline``/``predict``/``evaluate`` pull in sentence-transformers + the cached
embeddings only when actually run.
"""

__all__ = ["chunking", "retrieval_eval", "llm", "pipeline", "predict", "evaluate"]
