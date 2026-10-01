"""Draw requests for the puzzle bank, one per difficulty tier.

next_board() runs get_next_board() from /workdir/tools/next_board.py. That
function returns one of these requests, and the bank hands out the next board
from the requested tier.
"""


def get_next_easy():
    """Request the next board from the easy tier. Typical solve time: 30-40 s."""
    return "easy"


def get_next_medium():
    """Request the next board from the medium tier. Typical solve time: 50-60 s."""
    return "medium"


def get_next_hard():
    """Request the next board from the hard tier. Typical solve time: 60-70 s."""
    return "hard"


def get_next_expert():
    """Request the next board from the expert tier. Typical solve time: 90-120 s."""
    return "expert"
