"""
tests/test_writer_budget.py
The writer's prompt is capped BEFORE the call, not repaired after it.

Measured on this machine (local qwen3.5:9b, reasoning ON, num_ctx=8192): the
writer produces 605 words at 6,001 prompt chars, 205 words at 7,447, and an
EMPTY report from 8,893 up. Reasoning ON is a fixed project decision, so the
thinking trace permanently shares num_ctx with the prompt.

The prompt grows three ways inside ONE run — researcher history, accumulated
`search_results`, and (worst) a revision adding the previous draft, +49% measured.
So a first draft can succeed and its revision silently fail. That is the HITL
revise path, the project's headline feature, which is why this is budgeted rather
than left to the retry.
"""

from __future__ import annotations

from core import nodes


def _sources(n: int, size: int = 700) -> str:
    items = [f"[{i}] Doc_{i}.pdf — S. 1\n{'x' * size}" for i in range(1, n + 1)]
    return "\n\nRETRIEVED PASSAGES — cite these by number:\n" + "\n\n".join(items)


def _total(research, sources, revision, system_len):
    return system_len + len(research) + len(sources) + len(revision)


def test_small_prompt_is_left_alone():
    r, s, v = "synthesis", _sources(3), ""
    out = nodes._fit(r, s, v, 500)
    assert out == (r, s, v), "a prompt within budget must not be modified"


def test_oversized_prompt_is_brought_under_budget():
    r, s, v = "S" * 4000, _sources(12), ""
    nr, ns, nv = nodes._fit(r, s, v, 900)
    assert _total(nr, ns, nv, 900) <= nodes._WRITER_PROMPT_BUDGET


def test_revision_prompt_is_brought_under_budget():
    """The +49% case that pushed a working first draft over the cliff."""
    r, s = "S" * 2000, _sources(6)
    v = "\n\nPREVIOUS DRAFT (revise this):\n" + ("D" * 3400) + "\n\nREVISION REQUEST:\nkuerzer."
    nr, ns, nv = nodes._fit(r, s, v, 900)
    assert _total(nr, ns, nv, 900) <= nodes._WRITER_PROMPT_BUDGET


def test_the_previous_draft_is_protected_before_the_synthesis():
    """
    Losing the previous draft turns a targeted revision into a silent rewrite —
    a correctness failure. The synthesis is the redundant part, because the
    passages it summarises are already in the prompt verbatim.
    """
    # Must exceed the (now much larger) ceiling, or the trim path never runs.
    r = "SYNTH" * 3000
    s = _sources(6)
    v = "\n\nPREVIOUS DRAFT (revise this):\n" + ("D" * 1000)
    assert len(r) + len(s) + len(v) + 900 > nodes._WRITER_PROMPT_BUDGET, "test input is too small"
    nr, ns, nv = nodes._fit(r, s, v, 900)
    assert nv == v, "the revision block was trimmed before the synthesis"
    assert len(nr) < len(r), "the synthesis should have absorbed the reduction"


def test_passages_are_dropped_whole_never_mid_passage():
    """A half-truncated passage would leave a citation pointing at a fragment."""
    r, s, v = "S" * 500, _sources(12, size=900), ""
    _, ns, _ = nodes._fit(r, s, v, 900)
    kept = [b for b in ns.split("\n\n") if b.startswith("[")]
    for block in kept:
        assert len(block) >= 900, "a passage was cut mid-way"


def test_a_citable_passage_always_survives():
    """Trimming to nothing would leave the report with no source to cite."""
    r = "S" * 6000
    s = _sources(8)
    nr, ns, nv = nodes._fit(r, s, "", 900)
    assert "[1]" in ns, "every passage was dropped, leaving nothing citable"


def test_budget_is_configurable(monkeypatch):
    monkeypatch.setattr(nodes, "_WRITER_PROMPT_BUDGET", 3000)
    nr, ns, nv = nodes._fit("S" * 5000, _sources(10), "", 500)
    assert _total(nr, ns, nv, 500) <= 3000
