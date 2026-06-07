"""Evaluation suite + the Phase-7 end-to-end comparison.

Two layers:

1. **Pure metric primitives** (``score_generation_records``, ``mean_metrics``) — operate on
   plain dicts, no I/O, no LLM, no ML deps → fully unit-testable on a clean checkout.
2. **The Phase-7 driver** (``run_phase7`` / ``python -m src.evaluate --phase7``) — loads the
   cached embeddings + the Phase-5 HyDE generations + the Phase-6 generation judgments and
   produces the headline *optimal-pipeline vs naive-RAG* table, end to end (retrieval **and**
   answer correctness), with **zero new LLM calls** so the result is deterministic.

The retrieval axis is recomputed live from the committed-elsewhere ``.npy`` cache; the
generation axis is replayed from the Phase-6 judge cache. The two are stitched into one
story: *what does the whole pipeline buy you over naive top-k retrieval + generate?*
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .llm import CORRECTNESS_SCORE, citation_grounding, faithfulness, parse_judge
from .retrieval_eval import evaluate as retrieval_metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
EMB_DIR = os.path.join(HERE, "data/processed/emb_cache")
P5LLM = os.path.join(RES, "phase5_llm_cache")
P6LLM = os.path.join(RES, "phase6_llm_cache")

# Phase-6 condition → human label used in the end-to-end table.
COND_LABEL = {
    "hydeN": "Optimal (HyDE×N top-5)",
    "strong": "Naive (E5 top-5)",
    "oracle": "Oracle (gold ceiling)",
    "adversarial": "Adversarial (poisoned)",
    "closed_book": "Closed-book (no retrieval)",
}


# ---------------------------------------------------------------------------
# Pure metric primitives  (unit-testable, no I/O)
# ---------------------------------------------------------------------------

def score_generation_records(gen: Dict[str, dict], judge: Dict[str, dict]) -> Dict[str, dict]:
    """Aggregate cached generation+judge records into per-condition RAGAS means.

    ``gen``/``judge`` are keyed ``model::qid::cond``. For each condition returns
    correctness / faithfulness / citation-grounding / refusal-rate, plus the n that
    actually contributed to each mean (``None`` values are excluded, not zeroed).
    """
    by_cond: Dict[str, dict] = {}
    for key, g in gen.items():
        cond = g["cond"]
        ans = g.get("answer", "")
        parsed = judge.get(key, {}).get("parsed") or parse_judge(judge.get(key, {}).get("raw"))
        n_ctx = len(g.get("ctx_ids", [])) or (5 if cond != "closed_book" else 0)

        corr = CORRECTNESS_SCORE.get((parsed or {}).get("correctness", "UNKNOWN"))
        faith = faithfulness(parsed)
        cite = citation_grounding(ans, n_ctx)
        refused = 1.0 if _looks_refused(ans) else 0.0

        b = by_cond.setdefault(cond, {k: [] for k in
                                      ("correctness", "faithfulness", "citation", "refused")})
        if corr is not None:
            b["correctness"].append(corr)
        if faith is not None:
            b["faithfulness"].append(faith)
        if cite is not None:
            b["citation"].append(cite)
        b["refused"].append(refused)

    out = {}
    for cond, b in by_cond.items():
        out[cond] = {m: (float(np.mean(v)) if v else None) for m, v in b.items()}
        out[cond]["n"] = len(b["refused"])
    return out


def _looks_refused(ans: str) -> bool:
    a = (ans or "").lower()
    return any(p in a for p in ("does not contain", "doesn't contain", "no information",
                                "cannot answer", "can't answer", "not enough information",
                                "context does not", "unable to answer"))


def mean_metrics(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Phase-7 driver  (replays caches → deterministic; needs ML deps + caches present)
# ---------------------------------------------------------------------------

def _load_phase5_hyde():
    """(hyde_parsed_by_dsqid, embed_lookup, sample_qids, prf_best) from the Phase-5 cache."""
    import hashlib
    gen5 = json.load(open(f"{P5LLM}/gen_haiku.json"))
    z = np.load(f"{P5LLM}/gen_emb.npz", allow_pickle=True)
    hemb = {k: z[k] for k in z.files if k != "__keys__"}

    def h(t):
        return hashlib.md5(t.encode()).hexdigest()

    samp = {n: json.load(open(f"{P5LLM}/sample_{n}.json")) for n in ("scifact", "nfcorpus", "fiqa")}
    prf = json.load(open(f"{RES}/metrics.json")).get("phase5", {}).get("prf_best", {})
    return gen5, hemb, h, samp, prf


def _corpus_meta(name: str):
    """doc_ids, q_ids, qrels, D, Q for a corpus — pinned from the committed meta when present."""
    from collections import defaultdict

    from datasets import load_dataset
    import datasets as _ds
    _ds.disable_progress_bars()

    qrels_t = load_dataset(f"BeIR/{name}-qrels", split="test")
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in qrels_t:
        qrels[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    qrels = dict(qrels)

    meta_dir = os.path.join(HERE, "data/processed/meta")
    di = os.path.join(meta_dir, f"{name}_doc_ids.json")
    qi = os.path.join(meta_dir, f"{name}_q_ids.json")
    if os.path.exists(di) and os.path.exists(qi):
        doc_ids = [str(x) for x in json.load(open(di))]
        q_ids = [str(x) for x in json.load(open(qi))]
    else:  # rebuild order from BEIR (matches how the cache was written)
        corpus = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
        queries = load_dataset(f"BeIR/{name}", "queries", split="queries")
        doc_ids = [str(r["_id"]) for r in corpus]
        q_ids = [str(r["_id"]) for r in queries if str(r["_id"]) in qrels]

    D = np.load(f"{EMB_DIR}/E5-base-v2__{name}_docs.npy").astype("float32")
    Q = np.load(f"{EMB_DIR}/E5-base-v2__{name}_q.npy").astype("float32")
    return doc_ids, q_ids, qrels, D, Q


def build_runs(name: str):
    """Naive-E5 and HyDE×N runs over the Phase-5 sample for one corpus (deterministic replay)."""
    gen5, hemb, h, samp, _prf = _load_phase5_hyde()
    doc_ids, q_ids, qrels, D, Q = _corpus_meta(name)
    qpos = {q: i for i, q in enumerate(q_ids)}
    qids = [q for q in samp[name] if q in qpos]

    def rank(vec, k=100):
        sims = D @ vec
        k = min(k, D.shape[0])
        part = np.argpartition(-sims, k - 1)[:k]
        return [doc_ids[j] for j in part[np.argsort(-sims[part])]]

    naive, hyde = {}, {}
    for qid in qids:
        qv = Q[qpos[qid]]
        naive[qid] = rank(qv)
        parsed = gen5.get(f"{name}::{qid}", {}).get("parsed")
        passages = (parsed or {}).get("hyde") or []
        vecs = [qv] + [hemb[h(p)] for p in passages if h(p) in hemb]
        mv = np.mean(vecs, 0)
        mv /= (np.linalg.norm(mv) + 1e-12)
        hyde[qid] = rank(mv)
    sub = {q: qrels[q] for q in qids}
    return naive, hyde, sub


def run_phase7(write: bool = True) -> dict:
    """Produce the end-to-end *optimal vs naive* comparison across retrieval + generation."""
    # --- retrieval axis: naive E5 vs HyDE×N on all three corpora (fresh, deterministic) ---
    retrieval = {}
    for name in ("scifact", "nfcorpus", "fiqa"):
        naive, hyde, sub = build_runs(name)
        retrieval[name] = {
            "naive": retrieval_metrics(naive, sub),
            "hydeN": retrieval_metrics(hyde, sub),
            "n": len(sub),
        }

    # --- generation axis: replay the Phase-6 FiQA judgments (no new LLM calls) ---
    gen = json.load(open(f"{P6LLM}/gen_answers.json"))
    judge = json.load(open(f"{P6LLM}/judge.json"))
    scorecard = score_generation_records(gen, judge)

    summary = {"retrieval": retrieval, "generation_fiqa": scorecard,
               "cond_label": COND_LABEL}
    if write:
        _write_phase7_outputs(summary)
    return summary


def _write_phase7_outputs(summary: dict):
    import pandas as pd
    rows = []
    for name, m in summary["retrieval"].items():
        for ret in ("naive", "hydeN"):
            d = m[ret]
            rows.append(dict(dataset=name, n=m["n"],
                             pipeline="Optimal (HyDE×N)" if ret == "hydeN" else "Naive (E5)",
                             ndcg10=round(d["ndcg@10"], 4), recall10=round(d["recall@10"], 4),
                             recall100=round(d["recall@100"], 4), mrr10=round(d["mrr@10"], 4)))
    df_ret = pd.DataFrame(rows)
    df_ret.to_csv(f"{RES}/phase7_retrieval.csv", index=False)

    grows = []
    for cond, s in summary["generation_fiqa"].items():
        grows.append(dict(condition=cond, label=COND_LABEL.get(cond, cond), n=s["n"],
                          correctness=_r(s["correctness"]), faithfulness=_r(s["faithfulness"]),
                          citation=_r(s["citation"]), refused=_r(s["refused"])))
    df_gen = pd.DataFrame(grows)
    order = {"oracle": 0, "hydeN": 1, "strong": 2, "closed_book": 3, "adversarial": 4}
    df_gen = df_gen.sort_values("condition", key=lambda s: s.map(order)).reset_index(drop=True)
    df_gen.to_csv(f"{RES}/phase7_end_to_end.csv", index=False)
    _plot_phase7(df_ret, df_gen)
    print(df_ret.to_string(index=False))
    print()
    print(df_gen.to_string(index=False))


def _r(x):
    return round(x, 4) if x is not None else None


def _plot_phase7(df_ret, df_gen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    # left: retrieval nDCG@10 naive vs HyDE×N per corpus
    ds = list(df_ret.dataset.unique())
    x = np.arange(len(ds))
    naive = [df_ret[(df_ret.dataset == d) & (df_ret.pipeline == "Naive (E5)")]["ndcg10"].iloc[0] for d in ds]
    hyde = [df_ret[(df_ret.dataset == d) & (df_ret.pipeline == "Optimal (HyDE×N)")]["ndcg10"].iloc[0] for d in ds]
    ax1.bar(x - 0.2, naive, 0.4, label="Naive (E5 top-k)", color="#9aa6b2")
    ax1.bar(x + 0.2, hyde, 0.4, label="Optimal (HyDE×N)", color="#2563eb")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds)
    ax1.set_ylabel("nDCG@10")
    ax1.set_title("Retrieval: HyDE×N vs naive (3 BEIR corpora)")
    ax1.legend()
    for xi, (a, b) in enumerate(zip(naive, hyde)):
        ax1.text(xi + 0.2, b + 0.005, f"+{b - a:+.3f}".replace("++", "+"), ha="center", fontsize=8)

    # right: FiQA answer correctness by context condition (the poisoning story)
    order = ["oracle", "hydeN", "strong", "closed_book", "adversarial"]
    g = df_gen.set_index("condition").reindex([o for o in order if o in df_gen.condition.values])
    colors = {"oracle": "#16a34a", "hydeN": "#2563eb", "strong": "#9aa6b2",
              "closed_book": "#f59e0b", "adversarial": "#dc2626"}
    ax2.bar(range(len(g)), g["correctness"], color=[colors[c] for c in g.index])
    ax2.set_xticks(range(len(g)))
    ax2.set_xticklabels([COND_LABEL[c].split(" (")[0] for c in g.index], rotation=20, ha="right")
    ax2.set_ylabel("Answer correctness vs gold")
    ax2.set_title("Generation: retrieval quality → answer (FiQA, n=24)")
    for i, v in enumerate(g["correctness"]):
        ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle("Phase 7 — end-to-end: the optimal pipeline buys retrieval, "
                 "but wrong retrieval poisons the answer", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{RES}/phase7_end_to_end.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {RES}/phase7_end_to_end.png")


def build_demo(write: bool = True) -> dict:
    """Freeze the 24 FiQA demo queries (question + per-condition answer + scores + context)
    into ``results/phase7_demo.json`` so the Streamlit app runs on a clean checkout with no
    embeddings, no LLM, no BEIR download — just the committed cache. This is what the UI
    screenshot is rendered from."""
    from datasets import load_dataset
    import datasets as _ds
    _ds.disable_progress_bars()

    gen = json.load(open(f"{P6LLM}/gen_answers.json"))
    judge = json.load(open(f"{P6LLM}/judge.json"))
    q_text = {str(r["_id"]): r["text"] for r in load_dataset("BeIR/fiqa", "queries", split="queries")}
    docs = {str(r["_id"]): ((r["title"] + ". " + r["text"]).strip() if r["title"] else r["text"])
            for r in load_dataset("BeIR/fiqa", "corpus", split="corpus")}

    def snip(d, w=70):
        return " ".join(docs.get(d, "").split()[:w])

    qids, demo = [], {}
    for key, g in gen.items():
        qid, cond = g["qid"], g["cond"]
        if qid not in qids:
            qids.append(qid)
        parsed = judge.get(key, {}).get("parsed") or parse_judge(judge.get(key, {}).get("raw"))
        ctx_ids = g.get("ctx_ids", [])
        demo.setdefault(qid, {"question": q_text.get(qid, qid), "conditions": {}})
        demo[qid]["conditions"][cond] = dict(
            answer=g.get("answer", ""),
            context=[dict(label=f"d{i+1}", doc_id=d, snippet=snip(d)) for i, d in enumerate(ctx_ids)],
            correctness=CORRECTNESS_SCORE.get((parsed or {}).get("correctness", "UNKNOWN")),
            faithfulness=faithfulness(parsed),
            citation=citation_grounding(g.get("answer", ""), len(ctx_ids) or 5),
            judge_correctness=(parsed or {}).get("correctness", "UNKNOWN"),
        )
    out = {"queries": qids, "data": demo}
    if write:
        json.dump(out, open(f"{RES}/phase7_demo.json", "w"), indent=1)
        print(f"wrote {RES}/phase7_demo.json | {len(qids)} demo queries")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="RAG eval suite")
    ap.add_argument("--phase7", action="store_true", help="run the end-to-end comparison")
    ap.add_argument("--demo", action="store_true", help="freeze the Streamlit demo JSON")
    args = ap.parse_args()
    if args.phase7:
        run_phase7(write=True)
    if args.demo:
        build_demo(write=True)
    if not (args.phase7 or args.demo):
        ap.print_help()


if __name__ == "__main__":
    main()
