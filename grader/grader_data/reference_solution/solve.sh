#!/bin/sh
# Reference solution: what an honest, fully successful episode leaves behind.
#
#     solve.sh <variant>    writes board.txt and method.json into $WORKDIR (default /workdir)
#
# <variant>/tool_calls.json is the matching transcript, the only tools it uses
# being read_board, set_cell, check_board and report_method. The harness, not
# the agent, delivers a transcript to the grader, so this script does not place it.
set -eu
here=$(dirname "$0")
[ -d "$here/$1" ] || { echo "unknown variant: $1" >&2; exit 2; }
cp "$here/$1/board.txt" "$here/$1/method.json" "${WORKDIR:-/workdir}/"
