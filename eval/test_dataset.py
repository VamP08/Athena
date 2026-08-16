"""
eval/test_dataset.py
Ground truth dataset for Ragas evaluation.

Each entry has:
  topic          — the research question passed to the graph
  expected_facts — objective, verifiable facts that a good report should contain

Choose topics where expected facts are specific and checkable — not opinions.
run_eval.py joins expected_facts into the `reference` string that Ragas'
context_precision metric compares retrieved research against.

Run evaluation with:
    python eval/run_eval.py
"""

# Document-mode golden cases (python eval/run_eval.py --mode documents).
#
# The questions are the grounding harness's answerable cases, so both harnesses
# examine the same behaviour from two angles: the grounding harness scores the
# figures deterministically, Ragas scores relevance/faithfulness/precision with
# a judge. The unanswerable case is deliberately absent — Ragas has no notion of
# "refusing was correct", and rewarding an answer there would grade the WORST
# possible behaviour as the best. Refusal is covered by the grounding harness.
#
# expected_facts double as the context_precision reference, so they state the
# answer the archive actually contains, in the surface form the documents use.
DOCUMENT_CASES = [
    {
        "topic": "Wie hoch war der Umsatz in Q3 2025?",
        "expected_facts": [
            "Der Umsatz in Q3 2025 betrug 4.821.000 EUR",
            "Quartalsumsatz",
        ],
    },
    {
        "topic": "Wer hat den Jahresabschluss 2024 geprüft?",
        "expected_facts": [
            "Wagner & Petersen Wirtschaftsprüfung GmbH hat den Jahresabschluss 2024 geprüft",
            "uneingeschränkter Bestätigungsvermerk",
        ],
    },
    {
        "topic": "Wie hoch war das Eigenkapital zum 31.12.2024?",
        "expected_facts": [
            "Das Eigenkapital betrug zum 31.12.2024 7.240.000 EUR",
            "Bilanz",
        ],
    },
    {
        "topic": "What was total revenue in the 2024 financial year?",
        "expected_facts": [
            "Total revenue in the 2024 financial year was 18.452.000 EUR",
            "Umsatzerlöse",
        ],
    },
    {
        "topic": "Was war der Nettobetrag der Rechnung 2025-1042?",
        "expected_facts": [
            "Der Nettobetrag der Rechnung 2025-1042 betrug 128.400,00 EUR",
            "Kranich Handels AG",
        ],
    },
    {
        "topic": "Wie viele Mitarbeitende hatte das Unternehmen 2024?",
        "expected_facts": [
            "Das Unternehmen hatte 2024 214 Mitarbeitende",
            "Vorjahr 191",
        ],
    },
]

TEST_CASES = [
    {
        "topic": "The invention and history of the internet",
        "expected_facts": [
            "ARPANET",
            "TCP/IP",
            "Tim Berners-Lee",
            "1969",
            "World Wide Web",
        ],
    },
    {
        "topic": "Climate change effects on coral reefs",
        "expected_facts": [
            "bleaching",
            "ocean acidification",
            "Great Barrier Reef",
            "temperature",
            "carbon dioxide",
        ],
    },
    {
        "topic": "The 2008 global financial crisis",
        "expected_facts": [
            "subprime mortgage",
            "Lehman Brothers",
            "bailout",
            "recession",
            "housing bubble",
        ],
    },
    {
        "topic": "History and development of the Python programming language",
        "expected_facts": [
            "Guido van Rossum",
            "1991",
            "open source",
            "indentation",
            "readability",
        ],
    },
    {
        "topic": "SpaceX Starship development and goals",
        "expected_facts": [
            "reusable",
            "Boca Chica",
            "Elon Musk",
            "orbital",
            "Mars",
        ],
    },
]
