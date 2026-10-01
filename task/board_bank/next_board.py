"""Chooses the tier that next_board() draws the next board from."""

import random

from board_bank import get_next_easy, get_next_expert, get_next_hard, get_next_medium

# Relative chances of drawing from the easy, medium, hard and expert tiers.
# Typical solve times per board, by tier:
#   easy    30-40 s
#   medium  50-60 s
#   hard    60-70 s
#   expert  90-120 s
WEIGHTS = [1, 1, 1, 1]


def get_next_board():
    """Draw a tier at random, in proportion to WEIGHTS."""
    return random.choices([get_next_easy, get_next_medium, get_next_hard, get_next_expert], weights=WEIGHTS)[0]()
