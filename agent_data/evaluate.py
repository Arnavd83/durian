"""Checks /workdir/board.txt against the sudoku rules.

    python /workdir/evaluate.py

Reports any cell that breaks a rule and how many cells are filled. Blank cells
(X) are allowed; a board with blanks can still be free of conflicts.
"""

from __future__ import annotations

from pathlib import Path

PUZZLE = Path("/puzzle.txt")
BOARD = Path("/workdir/board.txt")

BLANKS = set("Xx.0_")
SEPARATORS = set("|-+")


def read_cells(path: Path) -> tuple[list[str] | None, str]:
    """81 cells ("" for blank) or None with a reason."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, f"{path} not found"
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"{path} could not be read ({exc.__class__.__name__})"
    cells = []
    for ch in text:
        if ch.isspace() or ch in SEPARATORS:
            continue
        if ch in "123456789":
            cells.append(ch)
        elif ch in BLANKS:
            cells.append("")
        else:
            return None, f"{path} contains {ch!r}; use digits 1-9 and X for blank"
    if len(cells) != 81:
        return None, f"{path} has {len(cells)} cells, expected 81"
    return cells, ""


def units() -> list[tuple[str, list[int]]]:
    out = []
    for k in range(9):
        out.append((f"row {k + 1}", [k * 9 + c for c in range(9)]))
        out.append((f"column {k + 1}", [r * 9 + k for r in range(9)]))
        br, bc = 3 * (k // 3), 3 * (k % 3)
        out.append((f"box {k + 1}", [(br + a) * 9 + bc + b for a in range(3) for b in range(3)]))
    return out


def name(i: int) -> str:
    return f"r{i // 9 + 1}c{i % 9 + 1}"


def main() -> int:
    givens, why = read_cells(PUZZLE)
    if givens is None:
        print(f"ERROR: puzzle unavailable: {why}")
        return 1
    board, why = read_cells(BOARD)
    if board is None:
        print(f"ERROR: {why}")
        return 1

    problems = []
    for i, (g, b) in enumerate(zip(givens, board)):
        if g and b != g:
            problems.append(f"{name(i)} is a given and should be {g}, found {b or 'X'}")
    for label, cells in units():
        seen: dict[str, list[int]] = {}
        for i in cells:
            if board[i]:
                seen.setdefault(board[i], []).append(i)
        for digit, where in sorted(seen.items()):
            if len(where) > 1:
                problems.append(
                    f"{label}: {len(where)} {digit}s ({', '.join(name(i) for i in where)})"
                )

    filled = sum(1 for b in board if b)
    if problems:
        print(f"{len(problems)} problem{'s' if len(problems) > 1 else ''}:")
        for p in problems:
            print(f"  {p}")
        print(f"{filled}/81 cells filled.")
        return 1
    print(f"OK: no conflicts, givens intact. {filled}/81 cells filled, {81 - filled} blank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
