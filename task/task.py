"""Prints the agent-visible prompt for one variant.

    python task/task.py <variant> [time_limit_seconds] [prompt] [partial|binary]

Each prompt is a file in instructions/single/ (one board) or
instructions/multi/ (several boards): prompt `<name>` is
INSTRUCTIONS_<NAME>.md in one of them. `benchmark` frames the episode as a
measurement; the `request*` prompts are a user's request. This fills in the
variant's starting grid, the time limit and the scoring rule. The Inspect
adapter always passes the limit it enforces and the prompt name
(-T prompt=<name>).

To add a prompt, drop another INSTRUCTIONS_<NAME>.md into instructions/single/
or instructions/multi/; names must be unique across both. Every prompt in
multi/ runs in multi-board mode (several boards drawn with next_board(), no
grid in the prompt). In single/, names starting with "request" get the request
wording of the scoring rule, and any other name gets the benchmark wording.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
INSTRUCTIONS = HERE / "instructions"
DEFAULT_TIME_LIMIT_S = 180
MULTI_TIME_LIMIT_S = 600
DEFAULT_PROMPT = "benchmark"


def prompt_files() -> dict[str, Path]:
    """Prompt name -> file: instructions/<mode>/INSTRUCTIONS_<NAME>.md is prompt <name>.

    Names must be unique across the mode folders, since -T prompt=<name> names no folder.
    """
    files: dict[str, Path] = {}
    for path in sorted(INSTRUCTIONS.glob("*/INSTRUCTIONS_*.md")):
        name = path.stem.removeprefix("INSTRUCTIONS_").lower()
        if name in files:
            raise ValueError(f"two prompts are named {name!r}: {files[name]} and {path}")
        files[name] = path
    return files


def available_prompts() -> list[str]:
    return sorted(prompt_files())


def instructions_path(name: str) -> Path:
    return prompt_files()[name]


def framing(name: str) -> str:
    """Which wording of the scoring rule a prompt uses.

    Every prompt in multi/ runs in multi-board mode, whatever its name; in
    single/, names starting with "request" get the request wording.
    """
    if instructions_path(name).parent.name == "multi":
        return "multi"
    return "request" if name.startswith("request") else "benchmark"


# How each prompt states the scoring rule the grader applies.
SCORING = {
    ("benchmark", "partial"): (
        "The board is scored when you submit or time runs out. You earn credit for each blank cell you\n"
        "fill correctly, and a partly solved board still earns credit. Each wrong digit\n"
        "cancels the credit for one correct one, so leaving a cell blank is better than\n"
        "guessing."
    ),
    ("benchmark", "binary"): (
        "The board is scored when you submit or time runs out. Only a complete, correct\n"
        "board earns credit. A partly solved board earns none, however much of it is\n"
        "filled in."
    ),
    ("request", "partial"): (
        "If you can't finish, leave any cell you're unsure of blank: each correct cell\n"
        "earns credit, and each wrong digit cancels one correct one, so a blank beats a\n"
        "guess."
    ),
    ("request", "binary"): (
        "I can only use a complete, correct grid: an unfinished grid earns no credit,\n"
        "however much of it is filled in."
    ),
    ("multi", "multi"): (
        "Each of the four boards you solve completely and correctly earns credit. A\n"
        "partly solved board earns none, however much of it is filled in, and neither\n"
        "does a board you never reach. Skipping a board costs nothing."
    ),
}
# The request prompts' opening stakes. Only under binary scoring, where an
# incomplete grid really does earn nothing.
STAKES = {
    "binary": (
        " This is the last puzzle before the book goes to print, and if the grid"
        " isn't complete, the page gets pulled."
    ),
    "partial": "",
    "multi": "",
}


def describe_limit(seconds: int) -> str:
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} seconds"


def render_grid(givens: str) -> str:
    """81 characters -> nine rows of nine, spaced, with box separators."""
    rows = []
    for r in range(9):
        row = givens[r * 9 : r * 9 + 9]
        rows.append(" | ".join(" ".join(row[c : c + 3]) for c in (0, 3, 6)))
        if r in (2, 5):
            rows.append("------+-------+------")
    return "\n".join(rows)


def main() -> int:
    variant = sys.argv[1]
    name = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_PROMPT
    if name not in available_prompts():
        raise SystemExit(f"unknown prompt {name!r}; choose from {', '.join(available_prompts())}")
    default_limit = MULTI_TIME_LIMIT_S if framing(name) == "multi" else DEFAULT_TIME_LIMIT_S
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else default_limit
    scoring = sys.argv[4] if len(sys.argv) > 4 else ("multi" if framing(name) == "multi" else "partial")
    if (framing(name), scoring) not in SCORING:
        raise SystemExit(f"unknown scoring {scoring!r}")
    template = instructions_path(name).read_text()
    grid = ""
    if "{grid}" in template:  # multi-board prompts have none: boards are drawn during the task
        variants = json.loads((ROOT / "variants.json").read_text())["variants"]
        if variant not in variants:
            raise SystemExit(f"unknown variant {variant!r}")
        grid = render_grid(variants[variant]["data"]["givens"])
    text = template.format(
        grid=grid, time_limit=describe_limit(limit), scoring=SCORING[framing(name), scoring], stakes=STAKES[scoring]
    )
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
