"""RAG Pipeline Optimizer — interactive demo.

    streamlit run app.py

The headline of the whole 7-day project, made tangible: a slider over **retrieval quality**
that shows what happens *to the generated answer*. Drag from a perfect oracle down to
adversarial (plausible-but-wrong) context and watch correctness collapse below closed-book
while the model stays "faithful" to the garbage — the Phase-6 poisoning finding, live.

Two modes:
* **Cached demo** (default) — 24 real FiQA queries × 5 conditions replayed from
  ``results/phase7_demo.json``. No model, no embeddings, no downloads → runs anywhere, and is
  what the README screenshot is taken from.
* **Live** — type your own question; runs the real pipeline (E5 + HyDE×N + Haiku). Requires
  the cached embeddings + the local ``claude`` CLI.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DEMO_PATH = os.path.join(RES, "phase7_demo.json")

COND_META = {
    "oracle":      ("Oracle (gold passages)",          "🟢", "The unreachable ceiling — only the gold-labelled passages."),
    "hydeN":       ("Optimal (HyDE×N top-5)",          "🔵", "The shipped pipeline: query averaged with 4 hypothetical-answer vectors."),
    "strong":      ("Naive (E5 top-5)",                "⚪", "Plain dense retrieval — the baseline every other phase optimised against."),
    "closed_book": ("Closed-book (no retrieval)",      "🟠", "No context at all — the model answers from parametric memory."),
    "adversarial": ("Adversarial (plausible-wrong)",   "🔴", "E5 ranks 40-60: on-topic but not the answer. The poisoning case."),
}
COND_ORDER = ["oracle", "hydeN", "strong", "closed_book", "adversarial"]

st.set_page_config(page_title="RAG Pipeline Optimizer", page_icon="🔍", layout="wide")


@st.cache_data
def load_demo():
    return json.load(open(DEMO_PATH))


def score_badges(c: dict):
    cols = st.columns(3)
    corr = c.get("correctness")
    faith = c.get("faithfulness")
    cite = c.get("citation")
    cols[0].metric("Answer correctness", f"{corr:.2f}" if corr is not None else "—",
                   help="vs the gold reference (judge: SUPPORTED=1, PARTIAL=0.5)")
    cols[1].metric("Faithfulness", f"{faith:.2f}" if faith is not None else "—",
                   help="fraction of the answer's claims entailed by the shown context")
    cols[2].metric("Citation grounding", f"{cite:.2f}" if cite is not None else "—",
                   help="fraction of [dN] citations pointing at a real passage in the window")


def render_condition(qrow: dict, cond: str):
    c = qrow["conditions"].get(cond)
    if not c:
        st.info("no cached result for this condition")
        return
    label, icon, blurb = COND_META[cond]
    st.markdown(f"#### {icon} {label}")
    st.caption(blurb)
    score_badges(c)
    st.markdown("**Answer**")
    st.write(c["answer"])
    if c["context"]:
        with st.expander(f"Retrieved context ({len(c['context'])} passages)"):
            for p in c["context"]:
                st.markdown(f"`[{p['label']}]` **doc {p['doc_id']}** — {p['snippet']}…")
    else:
        st.caption("_(closed-book — no passages retrieved)_")


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.title("🔍 RAG Pipeline Optimizer")
    st.markdown("**The optimal pipeline (Phases 1-6 verdict):**")
    st.markdown(
        "- **Encoder** E5-base-v2, *whole-doc* (no chunking)\n"
        "- **Index** exact NumPy top-k (Flat < 40k vectors)\n"
        "- **Re-ranker** ❌ *none* — every one tested hurt\n"
        "- **Query** HyDE×N (avg of 4 hypotheticals)\n"
        "- **Generator** Haiku + citation enforcement"
    )
    st.divider()
    st.markdown("**Hard-won findings**")
    st.markdown(
        "1. Chunking below the encoder window ≈ no-op\n"
        "2. ANN index *loses* nDCG below ~40k vectors\n"
        "3. Cross-encoder re-rankers drag a strong retriever down\n"
        "4. Only **HyDE×N** helped on all 3 corpora\n"
        "5. **Wrong retrieval poisons the answer** (below closed-book)"
    )
    st.divider()
    st.caption("Primary metric: nDCG@10 (retrieval) · answer-correctness (generation). "
               "Demo data: 24 FiQA queries, Haiku generator, fixed Haiku judge.")


# ----------------------------- main -----------------------------
st.title("Does better retrieval change the answer?")
st.markdown(
    "Five phases of this project moved one retrieval number. This demo closes the loop: "
    "pick a question, then **slide the retrieval quality** and watch the generated answer."
)

demo = load_demo()
qids = demo["queries"]
labels = {q: demo["data"][q]["question"] for q in qids}

tab_explore, tab_poison, tab_live = st.tabs(
    ["🎛️ Explore a query", "☠️ Poisoning demo", "⚡ Live pipeline"])

with tab_explore:
    qid = st.selectbox("FiQA question", qids,
                       format_func=lambda q: f"[{q}] {labels[q][:80]}")
    qrow = demo["data"][qid]
    st.markdown(f"**Q:** {qrow['question']}")
    cond = st.radio("Retrieval condition", COND_ORDER, horizontal=True,
                    format_func=lambda c: COND_META[c][1] + " " + COND_META[c][0].split(" (")[0])
    st.divider()
    render_condition(qrow, cond)

with tab_poison:
    st.markdown(
        "#### Same question, same cheap generator — only the *context* changes.\n"
        "The optimal retriever **ties the gold oracle**. But feed it plausible-but-wrong "
        "passages and correctness falls **below closed-book**, while the model stays "
        "confidently *faithful* to the wrong context."
    )
    qid2 = st.selectbox("FiQA question ", qids, key="poison_q",
                        format_func=lambda q: f"[{q}] {labels[q][:80]}")
    qrow2 = demo["data"][qid2]
    st.markdown(f"**Q:** {qrow2['question']}")
    left, right = st.columns(2)
    with left:
        render_condition(qrow2, "hydeN")
    with right:
        render_condition(qrow2, "adversarial")

    st.divider()
    st.markdown("**Mean over all 24 demo queries** (the headline table):")
    rows = []
    for c in COND_ORDER:
        vals = [demo["data"][q]["conditions"].get(c, {}) for q in qids]
        def avg(k):
            xs = [v.get(k) for v in vals if v.get(k) is not None]
            return round(sum(xs) / len(xs), 3) if xs else None
        rows.append(dict(Condition=COND_META[c][0], Correctness=avg("correctness"),
                         Faithfulness=avg("faithfulness"), Citation=avg("citation")))
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    img = os.path.join(RES, "phase7_end_to_end.png")
    if os.path.exists(img):
        st.image(img, caption="Phase 7 — end-to-end: retrieval lift (left) and the poisoning cliff (right)")

with tab_live:
    st.markdown("Run the **real** pipeline on your own question (needs cached E5 embeddings + the `claude` CLI).")
    q = st.text_input("Your question", "How is freelance income taxed for a sole proprietor?")
    cond_live = st.selectbox("Condition", ["hydeN", "strong", "closed_book"], index=0)
    if st.button("Run pipeline", type="primary"):
        try:
            from src.pipeline import ContextCondition, RAGPipeline, load_corpus
            with st.spinner("loading corpus + E5 (first run downloads BEIR)…"):
                corpus = load_corpus("fiqa", os.path.join(HERE, "data/processed/emb_cache"),
                                     os.path.join(HERE, "data/processed/meta"),
                                     domain="personal finance and investing")
                pipe = RAGPipeline(corpus)
            with st.spinner("retrieving + generating (HyDE×N issues 2 LLM calls)…"):
                res = pipe.answer(q, ContextCondition(cond_live), is_qid=False)
            if res.hyde_passages:
                with st.expander(f"HyDE hypotheticals ({len(res.hyde_passages)})"):
                    for h in res.hyde_passages:
                        st.caption(h)
            st.markdown("**Answer**")
            st.write(res.answer)
            cg = res.citation_grounding
            st.metric("Citation grounding", f"{cg:.2f}" if cg is not None else "—")
            st.metric("Latency (incl. CLI overhead)", f"{res.latency_s}s")
            if res.context:
                with st.expander("Retrieved context"):
                    for lab, did, txt in res.context:
                        st.markdown(f"`[{lab}]` **doc {did}** — {txt[:200]}…")
        except Exception as e:
            st.error(f"Live mode unavailable: {e}")
            st.caption("Use the Explore / Poisoning tabs — they run from the cached demo data.")
