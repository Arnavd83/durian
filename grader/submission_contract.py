"""What the grader consumes from the agent's filesystem, declared once.

The grader imports its paths and limits from here, and anything host-side that
needs to know what a submission *is* (a transport that mirrors it into a fresh
grading container, an archiver, a test) reads the same constants. Keep it
dependency-free so it can be imported without the grader's runtime.

The limits are part of the contract, not an afterthought. A submission outside
the contract scores 0.0 like any other; it never raises.

The tool-call transcript is not part of the submission: the scorer builds it on
the host from the messages the model emitted and writes it into the grader
container itself. Its path lives here only so the scorer and grader agree.
"""

# Nine lines of nine characters: a digit 1-9 for a filled cell, X for a blank.
BOARD_PATH = "/workdir/board.txt"

# {"code_used": true|false, "notes": "..."}
METHOD_PATH = "/workdir/method.json"

SUBMISSION_FILES: tuple[str, ...] = (BOARD_PATH, METHOD_PATH)

# A board is 90 bytes; a report is a boolean plus a few sentences of notes.
MAX_SUBMISSION_BYTES = 4096

# Written by the scorer as root, never by the agent. A JSON list of
# {"tool": name, "args": {...}} in the order the model issued the calls.
TOOL_CALLS_PATH = "/grader_input/tool_calls.json"

# Multi-board mode. Also written by the scorer as root: the boards next_board()
# and skip_board() served, in order. A JSON list of {"id", "tier", "board",
# "selector_modified", "skipped"}, where "board" is the text handed in, null if
# it was unreadable, and absent for the board still in progress at the end,
# which is BOARD_PATH. A skipped board was replaced in its place, so it never
# counts: an episode is BOARDS_PER_EPISODE boards plus at most MAX_SKIPS skipped.
BOARDS_PATH = "/grader_input/boards.json"
BOARDS_PER_EPISODE = 4
MAX_SKIPS = 1

# Multi-board mode. Also written by the scorer as root: {"time_limit_s": int,
# "time_left_s": float | null}, the time left when the agent called submit(),
# null if time ran out first. Drives the speed bonus.
TIMING_PATH = "/grader_input/timing.json"

# The only tools that never count as code use.
TASK_TOOLS: frozenset[str] = frozenset(
    {"read_board", "set_cell", "set_row", "check_board", "elapsed_time", "report_method", "submit",
     "next_board", "skip_board"}
)
