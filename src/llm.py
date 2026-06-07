"""LLM harness — local CLI calls to Claude / Codex, plus defensive parsers.

Factored out of ``notebooks/phase6_generation_ragas.ipynb`` so the production pipeline,
the Streamlit app and the test-suite all share *one* implementation of every LLM call.

The model is invoked through the locally-installed ``claude`` and ``codex`` CLIs (Anthony's
Max plan → $0 marginal cost). All call paths are wrapped so a missing CLI, a timeout or a
usage-limit cap degrades to a sentinel ``__ERROR__...`` string instead of raising — callers
filter those out rather than crash. None of this module imports heavy ML deps, so it is
safe to unit-test on a clean checkout.
"""
from __future__ import annotations

import json
import re
import subprocess
from typing import Optional

# Costs are representative 2026 API token prices (~250 in / ~120 out per RAG-gen call),
# NOT the CLI's wall-clock — see the README "honest reporting" note. Used only for the
# cost-per-1k column in the head-to-head tables.
COST_PER_CALL_USD = {
    "haiku": 0.0006,   # $1/MTok in,  $5/MTok out
    "opus": 0.0090,    # $15/MTok in, $75/MTok out
    "codex": 0.0500,   # codex agent overhead ~25k tokens/call
}


def call_claude(prompt: str, model: str = "haiku", timeout: int = 90) -> str:
    """Run the Claude CLI in one-shot print mode. Returns stdout, or ``__ERROR__...``.

    ``--no-session-persistence`` and ``--disable-slash-commands`` keep the call hermetic so
    repeated invocations are deterministic and cannot pick up stray session state.
    """
    try:
        r = subprocess.run(
            ["claude", "--print", "--model", model,
             "--no-session-persistence", "--disable-slash-commands"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() or "__ERROR__empty"
    except Exception as e:  # FileNotFoundError (no CLI), TimeoutExpired, usage caps
        return "__ERROR__" + str(e)[:80]


def call_codex(prompt: str, timeout: int = 180) -> str:
    """Run the Codex (GPT) CLI read-only. Slices the response out of the session wrapper.

    Codex echoes session metadata around the answer; the real text sits between the last
    ``codex\\n`` marker and the ``tokens used`` footer. Longer default timeout because the
    codex agent loop is slower than Claude's print mode.
    """
    try:
        r = subprocess.run(
            ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"],
            input=prompt, capture_output=True, text=True, timeout=timeout,
        )
        out = r.stdout
        if "tokens used" in out:
            out = out.rsplit("tokens used", 1)[0]
        if "codex\n" in out:
            out = out.rsplit("codex\n", 1)[-1]
        return out.strip() or "__ERROR__empty"
    except Exception as e:
        return "__ERROR__" + str(e)[:80]


def is_error(text: Optional[str]) -> bool:
    return (text is None) or text.startswith("__ERROR__")


# ---------------------------------------------------------------------------
# Parsers — both are pure functions over a string, so they unit-test with no CLI.
# ---------------------------------------------------------------------------

_CITE = re.compile(r"\[d(\d+)\]")


def citation_grounding(answer: str, n_ctx: int) -> Optional[float]:
    """Fraction of ``[dN]`` citations that point at a passage actually in the window.

    Returns ``None`` when the answer emits no citation at all (excluded from the mean
    rather than scored 0 — a closed-book answer legitimately cites nothing).
    """
    cites = _CITE.findall(answer or "")
    if not cites:
        return None
    ok = sum(1 for x in cites if 1 <= int(x) <= n_ctx)
    return ok / len(cites)


def parse_judge(raw: Optional[str]) -> Optional[dict]:
    """Parse the strict-JSON RAGAS judge response into ``{claims, subquestions, correctness}``.

    Strips an optional ```` ```json ```` fence, then takes the outermost ``{...}`` span so a
    chatty judge that prepends prose still parses. Returns ``None`` on any malformed output;
    callers drop those from the metric means and report a parse-success rate instead.
    """
    if is_error(raw):
        return None
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*", "", t).strip().rstrip("`").strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        o = json.loads(t[a:b + 1])
    except Exception:
        return None
    claims = [cl for cl in o.get("claims", [])
              if isinstance(cl, dict) and "supported_by_context" in cl]
    subq = [s for s in o.get("subquestions", []) if isinstance(s, str) and s.strip()][:3]
    corr = o.get("correctness", "UNKNOWN")
    if corr not in {"SUPPORTED", "PARTIAL", "CONTRADICT", "UNKNOWN"}:
        corr = "UNKNOWN"
    return dict(claims=claims, subquestions=subq, correctness=corr)


# Maps the judge's categorical correctness onto the [0,1] scale used in every results table
# (Phase 6 convention: PARTIAL is half credit, an UNKNOWN/refusal scores 0).
CORRECTNESS_SCORE = {"SUPPORTED": 1.0, "PARTIAL": 0.5, "CONTRADICT": 0.0, "UNKNOWN": 0.0}


def faithfulness(parsed: Optional[dict]) -> Optional[float]:
    """RAGAS faithfulness = fraction of atomic claims entailed by the context.

    ``None`` when there are no claims to score (e.g. a closed-book / refused answer), so it
    is excluded from the mean rather than counted as a perfect or zero score.
    """
    if not parsed:
        return None
    claims = parsed.get("claims", [])
    if not claims:
        return None
    return sum(1 for c in claims if c.get("supported_by_context")) / len(claims)
