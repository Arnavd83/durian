"""Tests for the authoring stats script. Not shipped."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_stats", ROOT / "authoring" / "run_stats.py")
run_stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_stats)


def cell(row, col, value):
    return {"row": row, "col": col, "value": value}


def test_straight_fill_has_no_corrections():
    assert run_stats.count_corrections([cell(1, 2, 1), cell(1, 3, 3), cell(2, 1, 6)]) == 0


def test_overwrite_and_clear_are_corrections():
    calls = [cell(4, 3, 2), cell(4, 3, 9), cell(5, 5, 1), cell(5, 5, 0)]
    assert run_stats.count_corrections(calls) == 2


def test_refilling_a_cleared_cell_is_not_another_correction():
    assert run_stats.count_corrections([cell(4, 3, 2), cell(4, 3, 0), cell(4, 3, 9)]) == 1


def test_malformed_args_do_not_crash():
    assert run_stats.count_corrections([{}, {"row": "1"}, cell(1, 1, None)]) == 0
