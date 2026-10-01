"""Chooses the tier that next_board() draws the next board from."""

import random

from board_bank import get_next_easy, get_next_hard, get_next_medium

# Relative chances of drawing from the easy, medium and hard tiers.
WEIGHTS = [1, 1, 1]


def get_next_board():
    """Draw a tier at random, in proportion to WEIGHTS."""
    return random.choices([get_next_easy, get_next_medium, get_next_hard], weights=WEIGHTS)[0]()
