"""Authoring check, not shipped: a puzzle must have exactly one solution, and it
must be the solution we store.

    python authoring/check_puzzle.py            # checks grader_data/puzzles.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PUZZLES = Path(__file__).resolve().parents[1] / "grader" / "grader_data" / "puzzles.json"


def peers(i: int) -> set[int]:
    r, c = divmod(i, 9)
    br, bc = 3 * (r // 3), 3 * (c // 3)
    out = {r * 9 + k for k in range(9)} | {k * 9 + c for k in range(9)}
    out |= {(br + a) * 9 + bc + b for a in range(3) for b in range(3)}
    return out - {i}


PEERS = [peers(i) for i in range(81)]


def solutions(grid: list[int], limit: int = 2) -> list[str]:
    """Up to `limit` solutions by backtracking on the most constrained cell."""
    found: list[str] = []

    def search() -> None:
        best, options = -1, None
        for i in range(81):
            if grid[i] == 0:
                opts = set(range(1, 10)) - {grid[p] for p in PEERS[i]}
                if options is None or len(opts) < len(options):
                    best, options = i, opts
        if options is None:
            found.append("".join(map(str, grid)))
            return
        for v in sorted(options):
            grid[best] = v
            search()
            grid[best] = 0
            if len(found) >= limit:
                return

    search()
    return found


def check(name: str, givens: str, solution: str) -> list[str]:
    errors = []
    if len(givens) != 81 or len(solution) != 81:
        return [f"{name}: givens/solution must be 81 characters"]
    grid = [0 if ch == "X" else int(ch) for ch in givens]
    for i, g in enumerate(grid):
        if g and str(g) != solution[i]:
            errors.append(f"{name}: given at r{i // 9 + 1}c{i % 9 + 1} disagrees with solution")
        if g and any(grid[p] == g for p in PEERS[i]):
            errors.append(f"{name}: givens repeat {g} around r{i // 9 + 1}c{i % 9 + 1}")
    if errors:
        return errors
    found = solutions(grid)
    if not found:
        return [f"{name}: no solution"]
    if len(found) > 1:
        return [f"{name}: more than one solution"]
    if found[0] != solution:
        return [f"{name}: unique solution differs from the stored one"]
    return []


def main() -> int:
    puzzles = json.loads(PUZZLES.read_text())
    errors = []
    for name, p in puzzles.items():
        errs = check(name, p["givens"], p["solution"])
        errors += errs
        blanks = p["givens"].count("X")
        print(f"{name}: {'FAIL' if errs else 'ok'} ({81 - blanks} givens, {blanks} blanks)")
    for e in errors:
        print("  " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
