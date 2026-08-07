"""
tests/test_grounding_metrics.py
Tests for the grounding harness itself.

This harness has now produced four false alarms before producing a true one, so
its own arithmetic is pinned. A metric that over-flags is not merely noisy: tuned
against, it would push the system toward terser, less thorough reports — the
opposite of what it exists to protect.

Each test below corresponds to a specific false positive that was observed and
then diagnosed.
"""

from __future__ import annotations

from decimal import Decimal

from eval.run_grounding_eval import (
    _derivable,
    looks_like_refusal,
    numbers_in,
    strip_dates,
    to_decimal,
)


class TestRefusalDetection:
    def test_real_german_refusals_are_recognised(self):
        """
        FALSE ALARM #5: the model refused correctly on the unanswerable question
        and the harness scored refusal_correct = 0/1. A refusal metric that
        misses real refusals is worst exactly where it matters most — it would
        push you to "fix" a system that was already right.
        """
        assert looks_like_refusal(
            "Es bestätigt, dass keine Umsatzdaten für das Jahr 2027 im "
            "Dokumentenarchiv verfügbar sind."
        )
        assert looks_like_refusal(
            "Der Umsatz im Jahr 2027 ist nicht dokumentiert und kann daher aus "
            "dem Archiv nicht beziffert werden."
        )
        assert looks_like_refusal("Ohne Planungsdaten lässt sich keine Prognose ableiten.")

    def test_a_confident_answer_is_not_a_refusal(self):
        assert not looks_like_refusal(
            "Der Umsatz im Jahr 2027 betrug 21.000.000 EUR laut Archiv."
        )


class TestDerivedFigures:
    def test_a_difference_of_two_grounded_figures_is_not_fabricated(self):
        """
        FALSE ALARM #6: "Das EBIT stieg um rund 350.000 EUR" was flagged as a
        fabricated figure. 2.522.000 - 2.170.000 = 352.000; the model computed,
        rounded and hedged it. That is analysis, not hallucination.
        """
        grounded = {Decimal("2522000"), Decimal("2170000")}
        assert _derivable(Decimal("350000"), grounded)
        assert _derivable(Decimal("352000"), grounded)

    def test_a_sum_is_also_derivable(self):
        grounded = {Decimal("7240000"), Decimal("9115000")}
        assert _derivable(Decimal("16355000"), grounded)

    def test_an_unrelated_figure_is_still_flagged(self):
        """The metric must not become permissive enough to miss real fabrication."""
        grounded = {Decimal("2522000"), Decimal("2170000")}
        assert not _derivable(Decimal("10355000"), grounded)
        assert not _derivable(Decimal("8489"), grounded)


class TestNumberParsing:
    def test_german_and_english_forms_are_the_same_value(self):
        assert to_decimal("18.452.000,00") == Decimal("18452000.00")
        assert to_decimal("18,452,000.00") == Decimal("18452000.00")
        assert to_decimal("18452000") == Decimal("18452000")

    def test_mixed_dot_format_is_not_read_as_a_giant_integer(self):
        """
        FALSE ALARM #4: the model wrote "16.180.000.00" (German thousands dots
        with an English decimal dot). Stripping every dot produced 1618000000,
        which matched nothing in the sources and was reported as a fabricated
        figure. The value was in fact correct and present.
        """
        assert to_decimal("16.180.000.00") == Decimal("16180000.00")
        assert to_decimal("18.452.000.00") == Decimal("18452000.00")
        assert to_decimal("2.522.000.00") == Decimal("2522000.00")

    def test_plain_thousands_dots_still_work(self):
        assert to_decimal("16.180.000") == Decimal("16180000")
        assert to_decimal("1.234") == Decimal("1234")

    def test_plain_decimal_still_works(self):
        assert to_decimal("1234.56") == Decimal("1234.56")
        assert to_decimal("0.5") == Decimal("0.5")

    def test_junk_returns_none(self):
        assert to_decimal("") is None
        assert to_decimal("abc") is None


class TestDateHandling:
    def test_dates_are_not_numbers(self):
        """
        FALSE ALARM #2: "31.12.2024" parsed as the integer 31122024 and was
        reported as a fabricated figure.
        """
        assert numbers_in("Stichtag 31.12.2024") == set()
        assert numbers_in("am 2024-03-02 gebucht") == set()
        assert "31.12.2024" not in strip_dates("Stichtag 31.12.2024")

    def test_real_figures_beside_a_date_survive(self):
        found = numbers_in("Zum 31.12.2024 betrug das Eigenkapital 7.240.000,00 EUR")
        assert Decimal("7240000.00") in found


class TestMagnitudeFloor:
    def test_citation_markers_and_small_counts_are_ignored(self):
        """[n] markers and headcounts would otherwise dominate the comparison."""
        assert numbers_in("siehe [3] und [4], 214 Mitarbeitende") == set()

    def test_bare_years_are_above_the_floor_and_are_counted(self):
        """
        Deliberate, and worth pinning: the floor is 1000, so a bare year like
        2024 IS treated as a figure. That is why dates are stripped first and why
        numbers appearing in the QUESTION are subtracted before anything is
        called ungrounded — without both, answering "there is no 2027 revenue"
        counted 2027 itself as an invented figure.
        """
        assert Decimal("2024") in numbers_in("im Jahr 2024 wurde gebucht")

    def test_financial_magnitudes_are_kept(self):
        assert Decimal("4821000") in numbers_in("Umsatz 4.821.000 EUR")
