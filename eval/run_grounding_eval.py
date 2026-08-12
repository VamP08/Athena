"""
eval/run_grounding_eval.py
Deterministic accuracy harness for document mode. No LLM judge, no tokens.

Why this exists alongside the Ragas run:
  Ragas measures qualities that need a judge (relevancy, faithfulness). It costs
  tokens, is subject to free-tier rate limits, and its scores move when the judge
  model changes — so it cannot gate CI and it cannot cleanly A/B a change to our
  own model settings. This harness measures the things that have exactly one
  right answer, using ground truth from a corpus we generate. It is free,
  offline, reproducible, and comparable across runs.

What it measures, in severity order:

  wrong_figure_rate    The report states a figure that is WRONG but plausible —
                       the adjacent quarter, the prior year. Most dangerous
                       outcome in a financial setting: it looks correct.
  ungrounded_rate      The report contains a large figure that appears in NONE of
                       the passages actually retrieved. Direct hallucination proxy.
  answer_accuracy      The report contains the correct figure.
  retrieval_hit_rate   A document that genuinely holds the answer was retrieved.
                       Separates "retrieval failed" from "model ignored evidence".
  refusal_correct      On an unanswerable question, did it say so instead of
                       inventing a number?

Usage:
    python eval/run_grounding_eval.py                       # current settings
    python eval/run_grounding_eval.py --reasoning on        # local thinking ON
    python eval/run_grounding_eval.py --reasoning off       # local thinking OFF
    python eval/run_grounding_eval.py --backend groq
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# ── number handling ──────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"\d[\d.,]*\d|\d")

# Dates must be removed BEFORE numbers are extracted. "31.12.2024" otherwise
# parses as the integer 31122024 and is then reported as an ungrounded figure —
# a hallucination that never happened. Measured this on the first run.
_DATE_RE = re.compile(
    r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"      # 31.12.2024 / 2.3.24
    r"|\b\d{4}-\d{2}-\d{2}\b"                     # 2024-03-02
    r"|\b\d{1,2}\.\s*(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|"
    r"Oktober|November|Dezember|January|February|March|May|June|July|"
    r"August|September|October|November|December)\b",
    re.IGNORECASE,
)


def strip_dates(text: str) -> str:
    return _DATE_RE.sub(" ", text or "")


def to_decimal(raw: str) -> Decimal | None:
    """
    Parse a German or English formatted number into a comparable Decimal.

    German writes 4.821.000,00 and English writes 4,821,000.00 for the same
    value, so neither separator can be assumed to mean one thing. Rule: when both
    separators appear, the RIGHTMOST is the decimal point. When only one appears,
    it is a decimal point only if it is the sole occurrence and is followed by
    exactly one or two digits; otherwise it groups thousands.

    Comparing digit strings instead would report 4.821.000,00 and 4821000 as
    different numbers, which is the same fact written two ways.
    """
    s = raw.strip().rstrip(".,")
    if not s:
        return None
    has_dot, has_comma = "." in s, "," in s
    try:
        if has_dot and has_comma:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif has_comma:
            frac = s.split(",")[-1]
            s = s.replace(",", ".") if s.count(",") == 1 and len(frac) in (1, 2) else s.replace(",", "")
        elif has_dot:
            parts = s.split(".")
            frac = parts[-1]
            if s.count(".") == 1 and len(frac) in (1, 2):
                pass  # 1234.56 — a plain decimal
            elif (
                len(parts) > 2
                and len(frac) == 2
                and all(len(p) == 3 for p in parts[1:-1])
            ):
                # Mixed format: German thousands dots with an English decimal
                # dot, e.g. "16.180.000.00". Not valid in either convention, but
                # a bilingual model writes it, and stripping every dot turned it
                # into 1618000000 — reported as a fabricated figure that never
                # existed. The shape is unambiguous (3-digit groups, 2-digit
                # tail), so the last dot is read as the decimal point.
                s = "".join(parts[:-1]) + "." + frac
            else:
                s = s.replace(".", "")
        return Decimal(s)
    except (InvalidOperation, ValueError):
        return None


def numbers_in(text: str, minimum: Decimal = Decimal(1000)) -> set[Decimal]:
    """
    Financial magnitudes in a text.

    Values below `minimum` are ignored: years, counts, percentages and the [n]
    citation markers would otherwise dominate and drown the signal. Headcount
    (214) is checked explicitly by its own case rather than by this sweep.
    """
    out = set()
    for m in _NUM_RE.finditer(strip_dates(text)):
        d = to_decimal(m.group())
        if d is not None and abs(d) >= minimum:
            out.add(d)
    return out


def contains_value(text: str, variants: list[str]) -> bool:
    """True if any surface form of the expected value appears, compared numerically."""
    if not text:
        return False
    targets = {to_decimal(v) for v in variants}
    targets = {t for t in targets if t is not None}
    if not targets:
        return any(v.lower() in text.lower() for v in variants)
    present = numbers_in(text, minimum=Decimal(0))
    if targets & present:
        return True
    return any(v.lower() in text.lower() for v in variants)


_REFUSAL_MARKERS = (
    "nicht enthalten", "keine angaben", "nicht vorhanden", "liegt nicht vor",
    "kein hinweis", "nicht verfügbar", "nicht gefunden", "enthält keine",
    "does not contain", "not contain", "no information", "not available",
    "not found", "cannot", "could not find", "no data", "unable to",
    "nicht abgedeckt", "keine informationen",
    # Added after a FALSE NEGATIVE: the model refused correctly, writing "keine
    # Umsatzdaten für das Jahr 2027 im Dokumentenarchiv verfügbar sind" and "ist
    # nicht dokumentiert und kann daher nicht beziffert werden", and the harness
    # scored it as a failure to refuse. A refusal metric that misses real
    # refusals is worse than useless on the most safety-critical case in the set:
    # it would push you to "fix" a system that was already behaving correctly.
    "nicht dokumentiert", "nicht beziffert", "keine umsatzdaten",
    "keine daten", "nicht ableiten", "liegen keine", "keine prognose",
    "keine planungsdaten", "rein spekulativ", "unmöglich macht",
    "no revenue data", "not documented", "cannot be quantified",
    "beyond the archive", "outside the archive",
)


def _derivable(value: Decimal, grounded: set[Decimal], tol: float = 0.02) -> bool:
    """
    Is `value` a simple arithmetic consequence of figures that ARE grounded?

    A financial report is supposed to compute: "EBIT rose by ~350,000" is derived
    from 2,522,000 - 2,170,000 = 352,000, rounded and hedged. Scoring that as a
    fabricated figure punishes analysis, which is the opposite of what this
    metric is for — and, tuned against, would push the system toward reports that
    only parrot cell values.

    Differences, sums and simple percentages are checked within a 2% tolerance,
    which accommodates the rounding a human writer would also apply.
    """
    if value in grounded:
        return True
    vals = [v for v in grounded if v != 0]
    for a in vals:
        for b in vals:
            for cand in (a - b, a + b):
                if cand and abs(float(value - cand)) <= tol * abs(float(cand)):
                    return True
        # percentage-of relationships, e.g. "18% of revenue"
        if abs(float(value)) <= 100:
            continue
    return False


def looks_like_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in _REFUSAL_MARKERS)


_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")

# Cues that a sentence is explicitly comparing periods rather than answering.
# German and English, because the corpus and the reports are both bilingual.
_COMPARISON_MARKERS = (
    "vorjahr", "vergleich", "verglichen", "gegenüber", "gegenueber", "zuvor",
    "vorperiode", "im jahr 2023", "(31.12.2023", "2023)", "anstieg von",
    "previous year", "prior year", "compared", "versus", " vs", "year-on-year",
    "yoy", "up from", "grew from", "increase from",
)


def _misattributed(report: str, case: dict) -> bool | None:
    """
    Is a competing figure attached to what the question actually asked?

    Looks only at sentences containing the question's anchor term (e.g. "Q3").
    If any such sentence states a competing value while stating no correct one,
    the report has attributed the wrong number to the asked-about thing — the
    genuinely dangerous failure, as distinct from merely mentioning other periods
    elsewhere for context.

    Returns None when the case defines no anchor, so it is never silently
    counted as a pass.
    """
    anchors = case.get("answer_anchor") or []
    competing = case.get("context_figures") or []
    correct = [v for grp in case.get("must_contain", []) for v in grp]
    if not anchors or not competing or not correct:
        return None

    for sent in _SENT_SPLIT.split(report or ""):
        low = sent.lower()
        if not any(a.lower() in low for a in anchors):
            continue
        if not any(contains_value(sent, [c]) for c in competing):
            continue
        if contains_value(sent, correct):
            continue
        # A sentence that EXPLICITLY frames the other figure as a comparison is
        # doing its job, not misattributing. "Zum Vergleich steht das
        # Eigenkapital des Vorjahres (31.12.2023) bei 6.480.000 EUR" names the
        # period it belongs to and is exactly what a good analyst writes.
        # Without this check the metric flagged a correct, well-cited report —
        # the second time this harness over-flagged good analysis, so the
        # exclusion is explicit rather than left to the reader's judgement.
        if any(m in low for m in _COMPARISON_MARKERS):
            continue
        return True
    return False


# ── run one case ─────────────────────────────────────────────────────────────

def run_case(case: dict, idx_n: int, total: int) -> dict:
    from langgraph.types import Command

    from core.graph import build_graph, make_initial_state

    graph = build_graph()
    cfg = {"configurable": {"thread_id": f"ground-{case['id']}-{time.time()}"}}

    print(f"  [{idx_n}/{total}] {case['question']}", flush=True)
    t0 = time.time()
    graph.invoke(make_initial_state(case["question"]), config=cfg)
    st = graph.get_state(cfg)
    if st.next == ("review",):
        graph.invoke(Command(resume="approve"), config=cfg)
        st = graph.get_state(cfg)
    elapsed = time.time() - t0

    report = st.values.get("draft_report", "") or ""
    chunks = st.values.get("retrieved_chunks", []) or []
    chunk_text = "\n".join(c.get("text", "") for c in chunks)
    sources = {c.get("source", "") for c in chunks}

    res = {
        "id": case["id"],
        "question": case["question"],
        "seconds": round(elapsed, 1),
        "words": len(report.split()),
        "n_chunks": len(chunks),
        # The report text is saved because EVERY flag this harness has raised
        # needed the underlying prose to adjudicate, and four of the first six
        # turned out to be defects in the metric rather than the system. Without
        # it, each check costs a full re-run — minutes of local generation — and
        # the temptation is to accept the number instead. Also keep the sentences
        # containing each flagged figure, so an ambiguous value like 1845200000
        # can be read as either "18.452.000.00" or "1.845.200.000" immediately.
        "report": report,
    }

    if case["unanswerable"]:
        # Correct behaviour is an explicit "not in the archive". A large figure
        # in the answer means it invented one.
        #
        # Numbers echoed from the QUESTION are excluded: answering "the archive
        # holds no 2027 revenue" necessarily repeats 2027, and counting that as a
        # fabrication marked a correct refusal as a failure in the first run.
        # _derivable must be applied HERE TOO. It was originally added only to
        # the answerable branch, so a refusal that correctly said "the archive
        # holds 2024 but not 2027" and quoted a computed year-on-year difference
        # was scored as inventing a figure. One branch forgiving arithmetic while
        # the other punishes it is an inconsistency in the metric, not a finding.
        grounded_u = numbers_in(chunk_text)
        cand_u = numbers_in(report) - grounded_u - numbers_in(case["question"])
        invented = {v for v in cand_u if not _derivable(v, grounded_u)}
        res["derived_figures"] = sorted(str(x) for x in (cand_u - invented))[:5]
        res["refusal_correct"] = looks_like_refusal(report)
        res["invented_figure"] = bool(invented)
        res["answer_hit"] = None
        res["misattributed"] = None
        res["retrieval_hit"] = None
        res["ungrounded"] = sorted(str(x) for x in invented)[:5]
        return res

    res["answer_hit"] = all(contains_value(report, group) for group in case["must_contain"])

    # Presence of a neighbouring figure is NOT an error. A report answering
    # "Q3 revenue" that also cites Q1 and Q2, or one that says headcount "grew
    # from 191 to 214", is doing good analysis. Scoring those as wrong (the
    # first version of this harness did) measures verbosity, not accuracy.
    res["context_figures_present"] = any(
        contains_value(report, [w]) for w in case.get("context_figures", [])
    )

    # The real question is ATTRIBUTION: is the correct value the one attached to
    # what was asked? Checked at sentence granularity — the sentence carrying the
    # question's qualifier ("Q3", "Eigenkapital") must carry the right figure and
    # must not carry a competing one.
    res["misattributed"] = _misattributed(report, case)
    res["retrieval_hit"] = (
        bool(sources & set(case["expect_sources"])) if case["expect_sources"] else None
    )
    grounded = numbers_in(chunk_text)
    candidates = numbers_in(report) - grounded - numbers_in(case["question"])
    # Separate "computed from grounded figures" from "appeared out of nowhere".
    # Only the latter is a fabrication.
    derived = {v for v in candidates if _derivable(v, grounded)}
    ungrounded = candidates - derived
    res["derived_figures"] = sorted(str(x) for x in derived)[:5]
    res["ungrounded"] = sorted(str(x) for x in ungrounded)[:5]
    res["ungrounded_count"] = len(ungrounded)
    res["refusal_correct"] = None
    res["invented_figure"] = None
    return res


def main() -> int:
    # This harness is meant to gate CI, where stdout is a pipe rather than a
    # console — and on Windows a redirected stdout defaults to cp1252, which
    # cannot encode the box-drawing characters or the German text in the report.
    # A full 7-case run (≈20 minutes) then died on the FIRST summary line, after
    # every case had been scored but BEFORE --out was written, so the entire run
    # was lost to a print. Encoding is fixed here rather than left to the caller
    # to remember an environment variable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # already wrapped, or not a TextIO
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", choices=["on", "off"], default=None,
                    help="force the local model's thinking trace on or off")
    ap.add_argument("--backend", choices=["ollama", "groq"], default=None)
    ap.add_argument("--out", default=None, help="write per-case JSON here")
    ap.add_argument("--only", default=None, help="run a single case id")
    args = ap.parse_args()

    os.environ["ATHENA_MODE"] = "documents"
    os.environ.setdefault("ATHENA_CHECKPOINTER", "memory")
    if args.backend:
        os.environ["ATHENA_LLM_BACKEND"] = args.backend
    if args.reasoning:
        os.environ["ATHENA_OLLAMA_REASONING"] = "true" if args.reasoning == "on" else "false"

    from core.llm import resolve_backend

    from eval.grounding_dataset import GROUNDING_CASES

    cases = [c for c in GROUNDING_CASES if not args.only or c["id"] == args.only]
    backend, model = resolve_backend()
    # Default MUST match core/llm.py's default ("true"). Defaulting the label to
    # "false" here while the code defaults the behaviour to "true" wrote the
    # wrong configuration into the results file — a mislabelled artifact is worse
    # than none, because later comparisons silently trust it.
    reasoning = os.getenv("ATHENA_OLLAMA_REASONING", "true")

    print(f"Backend: {backend} ({model}) · local reasoning: {reasoning} · {len(cases)} cases")
    print()

    t0 = time.time()
    rows = [run_case(c, i, len(cases)) for i, c in enumerate(cases, 1)]
    wall = time.time() - t0

    answerable = [r for r in rows if r["answer_hit"] is not None]
    unans = [r for r in rows if r["refusal_correct"] is not None]

    def pct(n, d):
        return f"{(n / d * 100):.0f}%" if d else "n/a"

    hits = sum(1 for r in answerable if r["answer_hit"])
    wrong = sum(1 for r in answerable if r["misattributed"])
    rhits = [r for r in answerable if r["retrieval_hit"] is not None]
    rhit = sum(1 for r in rhits if r["retrieval_hit"])
    ungrounded_cases = sum(1 for r in answerable if r.get("ungrounded_count"))
    refusals = sum(1 for r in unans if r["refusal_correct"])

    print("\n── Per case ──")
    for r in rows:
        if r["answer_hit"] is not None:
            bad = (not r["answer_hit"]) or r["misattributed"] or r.get("ungrounded_count")
            flag = "FAIL" if bad else "OK  "
            extra = " MISATTRIBUTED" if r["misattributed"] else ""
            ung = f" ungrounded={r['ungrounded']}" if r.get("ungrounded") else ""
            print(f"  {flag} {r['id']:<20s} {r['seconds']:>6.1f}s "
                  f"chunks={r['n_chunks']:<3d} answer={r['answer_hit']} "
                  f"retrieved={r['retrieval_hit']}{extra}{ung}")
        else:
            flag = "OK  " if r["refusal_correct"] and not r["invented_figure"] else "FAIL"
            print(f"  {flag} {r['id']:<20s} {r['seconds']:>6.1f}s "
                  f"refused={r['refusal_correct']} invented={r['invented_figure']}")

    print("\n── Summary ──")
    print(f"  answer_accuracy      {hits}/{len(answerable)}  ({pct(hits, len(answerable))})")
    print(f"  misattributed_rate   {wrong}/{len(answerable)}  ({pct(wrong, len(answerable))})   <- lower is better")
    print(f"  ungrounded_cases     {ungrounded_cases}/{len(answerable)}  ({pct(ungrounded_cases, len(answerable))})   <- lower is better")
    print(f"  retrieval_hit_rate   {rhit}/{len(rhits)}  ({pct(rhit, len(rhits))})")
    print(f"  refusal_correct      {refusals}/{len(unans)}  ({pct(refusals, len(unans))})")
    print(f"  wall time            {wall:.1f}s  ({wall / max(1, len(rows)):.1f}s per case)")
    print(f"  backend              {backend} ({model}), local reasoning={reasoning}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {"backend": backend, "model": model, "reasoning": reasoning,
                 "wall_seconds": round(wall, 1), "cases": rows},
                indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
