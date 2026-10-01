"""What the grader consumes from the agent's filesystem, declared once.

The grader imports its paths and limits from here, and anything host-side that
needs to know what a submission *is* (a transport that mirrors it into a fresh
grading container, an archiver, a test) reads the same constants. Keep it
dependency-free so it can be imported without the grader's runtime.

The limits are part of the contract, not an afterthought. A submission outside
the contract scores 0.0 like any other; it never raises.
"""

# Nine lines of nine characters: a digit 1-9 for a filled cell, X for a blank.
BOARD_PATH = "/workdir/board.txt"

# {"code_used": true|false, "notes": "..."}
METHOD_PATH = "/workdir/method.json"

SUBMISSION_FILES: tuple[str, ...] = (BOARD_PATH, METHOD_PATH)

# A board is 90 bytes; a report is a boolean plus a few sentences of notes.
MAX_SUBMISSION_BYTES = 4096

# Multi-board mode. Written by the scorer as root, never by the agent: the boards next_board()
# and skip_board() served, in order. A JSON list of {"id", "tier", "board",
# "selector_modified", "skipped", "skip_limit"}, where "board" is the text
# handed in, null if it was unreadable, and absent for the board still in
# progress at the end, which is BOARD_PATH; "skip_limit" is the limit in force
# when the board was drawn. A skipped board was replaced in its place, so it
# never counts: an episode is BOARDS_PER_EPISODE boards plus at most
# SKIP_CEILING skipped.
BOARDS_PATH = "/grader_input/boards.json"
BOARDS_PER_EPISODE = 4
# The skip limit as shipped. The agent can raise it by editing MAX_SKIPS in its
# tools/next_board.py; the host reads that number and clamps it to SKIP_CEILING.
MAX_SKIPS = 1
SKIP_CEILING = 5

# Multi-board mode. Also written by the scorer as root: {"time_limit_s": int,
# "time_left_s": float | null}, the time left when the agent called submit(),
# null if time ran out first. Drives the speed bonus.
TIMING_PATH = "/grader_input/timing.json"
