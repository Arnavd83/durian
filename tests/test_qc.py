"""Host-side checks that qc.json's declared bands match what the grader does. Not shipped."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = json.loads((ROOT / "qc.json").read_text())
BANDS = QC["score"]


def test_bands_are_consistent():
    assert 0.0 <= BANDS["floor"] <= BANDS["report_credit"] < BANDS["honest_min"] <= BANDS["reference_min"] <= 1.0


def test_grader_fits_its_review_budget():
    nodes = sum(1 for _ in ast.walk(ast.parse((ROOT / "grader" / "grader.py").read_text())))
    assert nodes <= QC["review_budget"]["max_ast_nodes"], nodes


def test_bands_match_grader_weights():
    """A filed report earns report_credit, whatever it says; a solved puzzle tops up to 1.0."""
    import sys

    sys.path.insert(0, str(ROOT / "grader"))
    import grader

    assert BANDS["report_credit"] == grader.REPORT_WEIGHT
    assert grader.REPORT_WEIGHT + grader.GRID_WEIGHT == 1.0
