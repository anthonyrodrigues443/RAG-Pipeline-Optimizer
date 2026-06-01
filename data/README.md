# Datasets

Both datasets are part of **BEIR** (Thakur et al., NeurIPS 2021), loaded at runtime from
the Hugging Face Hub — nothing is committed to the repo.

| Dataset | HF id | Corpus | Test queries | Qrels | Role |
|---------|-------|-------:|-------------:|-------|------|
| SciFact | `BeIR/scifact` | 5,183 | 300 | 339 (binary) | Clean baseline — validate the eval harness against published BM25 nDCG@10 ≈ 0.665 |
| NFCorpus | `BeIR/nfcorpus` | 3,633 | 323 | 12,334 (graded 1/2) | Chunking study — graded relevance, longer medical documents |

## Why these two

- **SciFact** is a sparse-judgment factoid retrieval task (scientific claim verification).
  Its published BM25/dense baselines let us prove our nDCG/Recall/MRR implementation is
  correct before trusting it on anything else.
- **NFCorpus** has graded relevance and ~38 relevant docs per query, with documents whose
  token length frequently exceeds the encoder's 256-token window — the regime where
  chunking strategy actually changes retrieval quality.

## License
BEIR datasets are released for research use under their respective source licenses; see the
[BEIR repository](https://github.com/beir-cellar/beir) for per-dataset details.

## Reproduce
```python
from datasets import load_dataset
corpus  = load_dataset("BeIR/scifact", "corpus", split="corpus")
queries = load_dataset("BeIR/scifact", "queries", split="queries")
qrels   = load_dataset("BeIR/scifact-qrels", split="test")
```
