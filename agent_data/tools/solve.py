"""Backtracking sudoku solver.

    python3 /workdir/tools/solve.py [board.txt]

Reads a board (nine lines of nine, X for blank) and prints the solved grid.
"""

import sys


def solve(grid):
    for i in range(81):
        if grid[i] == 0:
            r, c = divmod(i, 9)
            used = {grid[r * 9 + k] for k in range(9)} | {grid[k * 9 + c] for k in range(9)}
            br, bc = 3 * (r // 3), 3 * (c // 3)
            used |= {grid[(br + a) * 9 + bc + b] for a in range(3) for b in range(3)}
            for v in range(1, 10):
                if v not in used:
                    grid[i] = v
                    if solve(grid):
                        return True
            grid[i] = 0
            return False
    return True


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/workdir/board.txt"
    cells = [ch for ch in open(path).read() if not ch.isspace()]
    grid = [0 if ch in "Xx.0" else int(ch) for ch in cells]
    if len(grid) != 81 or not solve(grid):
        print("no solution")
        return 1
    for r in range(9):
        print("".join(map(str, grid[r * 9 : r * 9 + 9])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
