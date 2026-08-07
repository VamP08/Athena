"""
eval/grounding_dataset.py
Golden dataset for document mode, with EXACT ground truth.

Every value here is a figure the synthetic corpus provably contains (see
corpus_tools/generate_corpus.py, which generates from a fixed seed). Because the
truth is known exactly, these cases can be scored deterministically — no LLM
judge, no tokens, no rate limits — which is what lets them run as a CI gate.

Field meanings:
  question        asked of the graph
  must_contain    the correct answer, in any accepted surface form. Scored as a
                  hit if ANY variant appears (German 4.821.000 and raw 4821000
                  are the same fact).
  context_figures neighbouring values — the adjacent quarter, the prior year, the
                  gross rather than net total. Their PRESENCE is not an error: a
                  report that compares Q1 -> Q2 -> Q3, or says headcount "grew
                  from 191 to 214", is doing good analysis. An earlier version of
                  this harness scored these as wrong answers and produced an
                  alarming-looking 83% "wrong figure rate" that measured
                  verbosity rather than accuracy. They are used only for the
                  attribution test below.
  answer_anchor   terms identifying the sentence that is supposed to answer the
                  question. The attribution test asks: in a sentence about THIS
                  (e.g. "Q3"), is the stated figure the correct one? That is the
                  genuinely dangerous failure — right table, wrong row — as
                  opposed to merely mentioning other periods elsewhere.
  expect_sources  documents that genuinely contain the answer; retrieval is
                  scored a hit if at least one is retrieved.
  unanswerable    True if the archive does not contain the answer. The correct
                  behaviour is to SAY SO. Inventing a figure here is the single
                  worst outcome the system can produce.
"""

GROUNDING_CASES = [
    {
        "id": "q3_2025_revenue",
        "question": "Wie hoch war der Umsatz in Q3 2025?",
        "must_contain": [["4.821.000", "4821000", "4,821,000"]],
        "context_figures": ["4.655.000", "4655000", "4.310.000", "4310000"],
        "answer_anchor": ["Q3"],
        "expect_sources": ["Bilanz_und_GuV_2024.xlsx"],
        "unanswerable": False,
    },
    {
        "id": "auditor",
        "question": "Wer hat den Jahresabschluss 2024 geprüft?",
        "must_contain": [["Wagner & Petersen", "Wagner &amp; Petersen", "Wagner und Petersen"]],
        "context_figures": [],
        "answer_anchor": [],
        "expect_sources": [
            "Geschaeftsbericht_2024.pdf", "Audit_Report_2024_EN.docx",
            "Steuerunterlagen_2024.md",
        ],
        "unanswerable": False,
    },
    {
        "id": "equity_2024",
        "question": "Wie hoch war das Eigenkapital zum 31.12.2024?",
        "must_contain": [["7.240.000", "7240000", "7,240,000"]],
        "context_figures": ["6.480.000", "6480000", "9.115.000", "9115000"],
        "answer_anchor": ["Eigenkapital"],
        "expect_sources": ["Geschaeftsbericht_2024.pdf", "Bilanz_und_GuV_2024.xlsx"],
        "unanswerable": False,
    },
    {
        "id": "revenue_2024_en",
        # Asked in English although the authoritative figure is in German
        # documents too — tests cross-language retrieval end to end.
        "question": "What was total revenue in the 2024 financial year?",
        "must_contain": [["18.452.000", "18452000", "18,452,000"]],
        "context_figures": ["16.180.000", "16180000", "16,180,000"],
        "answer_anchor": ["2024"],
        "expect_sources": ["Annual_Report_2024_EN.pdf", "Geschaeftsbericht_2024.pdf"],
        "unanswerable": False,
    },
    {
        "id": "invoice_1042",
        "question": "Was war der Nettobetrag der Rechnung 2025-1042?",
        "must_contain": [["128.400", "128400"]],
        "context_figures": ["152.796", "152796"],  # the gross total
        "answer_anchor": ["Nettobetrag", "netto"],
        "expect_sources": ["Rechnung_2025_1042.pdf"],
        "unanswerable": False,
    },
    {
        "id": "headcount_2024",
        "question": "Wie viele Mitarbeitende hatte das Unternehmen 2024?",
        "must_contain": [["214"]],
        "context_figures": ["191"],  # prior year
        "answer_anchor": ["2024"],
        "expect_sources": ["Geschaeftsbericht_2024.pdf", "Annual_Report_2024_EN.pdf"],
        "unanswerable": False,
    },
    {
        "id": "unanswerable_2027",
        # The archive stops at the 2026 budget. There is no 2027 revenue anywhere.
        "question": "Wie hoch war der Umsatz im Jahr 2027?",
        "must_contain": [],
        "context_figures": [],
        "answer_anchor": [],
        "expect_sources": [],
        "unanswerable": True,
    },
]
