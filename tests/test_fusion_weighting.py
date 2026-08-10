"""
tests/test_fusion_weighting.py
The lexical channel's weight in RRF is a measured decision, not a preference.

Equal-weight RRF made hybrid retrieval WORSE than dense alone on a 1,200-document
/ 18,591-chunk archive (MRR 0.923 vs 0.937, hit@1 0.864 vs 0.883, 360 queries),
because BM25 scores hit@1 = 0.138 on paraphrased queries — near noise — while
still casting a full vote. It is 1.000 on exact identifiers and amount lookups,
where dense dips to 0.975.

So the weight is conditional on the query carrying a literal anchor. These tests
pin the anchor detector, because if it silently stopped firing, identifier
lookups would quietly regress and no other test would notice.
"""

from __future__ import annotations

from core.index import _has_literal_anchor


class TestLiteralAnchorDetection:
    def test_reference_codes_are_anchors(self):
        assert _has_literal_anchor("Belegnummer 2025-Q3-00123")
        assert _has_literal_anchor("Beleg BEL-2024-3015")
        assert _has_literal_anchor("invoice INV/778")

    def test_figures_are_anchors(self):
        assert _has_literal_anchor("Welches Dokument weist genau 128.400,00 EUR aus?")
        assert _has_literal_anchor("Umsatz 4821000")

    def test_iban_like_tokens_are_anchors(self):
        assert _has_literal_anchor("Zahlung auf DE44 5001 0517 5407 3249 31")

    def test_plain_prose_is_not_an_anchor(self):
        """
        These are exactly the queries where BM25 measured 0.138 hit@1, so they
        must NOT grant it a full vote.
        """
        assert not _has_literal_anchor("Welche Summe wurde Nordwind GmbH in Rechnung gestellt?")
        assert not _has_literal_anchor("Ausgangsrechnung Elbe AG drittes Quartal")
        assert not _has_literal_anchor("Wer hat den Jahresabschluss geprueft?")

    def test_a_bare_year_does_not_count_as_an_anchor(self):
        """
        A year is shared by hundreds of documents, so it identifies nothing.

        This test previously asserted on strings containing NO year at all, so it
        passed while testing nothing — and the shipped regex did match bare years,
        which made the whole weighting mechanism a no-op (the A/B came back
        byte-identical to equal-weight RRF). The strings below now actually
        contain the years they claim to.
        """
        assert not _has_literal_anchor("Nettobetrag Nordwind GmbH Q3 2025")
        assert not _has_literal_anchor("Ausgangsrechnung Elbe AG drittes Quartal 2024")
        assert not _has_literal_anchor("Welche Summe wurde 2023 in Rechnung gestellt?")
        assert not _has_literal_anchor("Umsatz im Jahr")
        assert not _has_literal_anchor("drittes Quartal")

    def test_the_real_benchmark_queries_are_classified_as_intended(self):
        """
        Pin the actual query shapes the retrieval eval uses, so a regex change
        cannot silently flip the weighting for a whole tier again.
        """
        # anchored: lexical is measured at hit@1 = 1.000 on these
        assert _has_literal_anchor("Belegnummer 2025-Q3-00123")
        assert _has_literal_anchor("Welches Dokument weist genau 128.400,00 EUR aus?")
        # not anchored: lexical is measured at hit@1 = 0.138 on these
        assert not _has_literal_anchor(
            "Welche Summe wurde Elbe AG im dritten Quartal 2024 in Rechnung gestellt?"
        )
        assert not _has_literal_anchor("Buchungsjournal Hansa KG viertes Quartal 2025")

    def test_empty_and_none_are_safe(self):
        assert not _has_literal_anchor("")
        assert not _has_literal_anchor(None)
