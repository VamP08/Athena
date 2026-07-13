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
