"""Authoring tool, not shipped: time, token and tool-call statistics from eval logs.

    .venv/bin/python authoring/run_stats.py                    # every log in logs/
    .venv/bin/python authoring/run_stats.py logs/x.eval --timeline
    .venv/bin/python authoring/run_stats.py --csv runs.csv

Prints one row per run, then, for each (model, variant) with two or more runs,
the mean, standard deviation and coefficient of variation (sd / mean) of each
candidate budget measure. The measure with the lowest CV is the most
repeatable one to budget on.

--timeline also prints every model and tool call in each run, with its offset
from the start of the run in wall-clock and working time.

Measures:
  wall_s      wall-clock time, start to finish
  work_s      working time: wall-clock minus waits on rate limits and retries
  model_s     working time spent inside model calls
  out_tok     tokens the model generated, reasoning included
  reason_tok  reasoning tokens alone
  in_tok      input tokens, cached and uncached (the context re-sent each turn)
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime
from pathlib import Path

from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.log import read_eval_log

MEASURES = ("wall_s", "work_s", "model_s", "out_tok", "reason_tok", "in_tok")


def count_corrections(set_cells: list[dict]) -> int:
    """set_cell calls that clear a cell or overwrite a digit the agent already entered."""
    filled: set[tuple] = set()
    corrections = 0
    for args in set_cells:
        key = (args.get("row"), args.get("col"))
        if args.get("value") == 0 or key in filled:
            corrections += 1
        if args.get("value"):
            filled.add(key)
        else:
            filled.discard(key)
    return corrections


def run_row(path: Path, sample, log_args: dict | None = None) -> dict:
    tools = [e for e in sample.events if isinstance(e, ToolEvent)]
    models = [e for e in sample.events if isinstance(e, ModelEvent)]
    ok_set_cells = [e.arguments for e in tools if e.function == "set_cell" and e.error is None]
    usage = list(sample.model_usage.values())
    report = next((e.arguments for e in reversed(tools) if e.function == "report_method"), {})
    score = next(iter(sample.scores.values())).value if sample.scores else None
    return {
        "log": path.name[:40],
        "model": ", ".join(m.split("/")[-1] for m in sample.model_usage) or "?",
        "variant": sample.id,
        "score": score,
        "error": (sample.error.message[:60] if sample.error else "") or (
            f"limit: {sample.limit.type}" if sample.limit else ""
        ),
        "wall_s": round(sample.total_time or 0, 1),
        "work_s": round(sample.working_time or 0, 1),
        "model_s": round(sum(e.working_time or 0 for e in models), 1),
        "model_calls": len(models),
        "max_turn_reason": max(
            ((e.output.usage.reasoning_tokens or 0) for e in models if e.output and e.output.usage),
            default=0,
        ),
        "turn_cap": (log_args or {}).get("reasoning_tokens"),
        "effort": (log_args or {}).get("reasoning_effort"),
        "max_turn_s": (log_args or {}).get("max_turn_s"),
        "interrupts": sum(1 for e in sample.events if e.event == "info" and e.source == "turn_interrupt"),
        "interrupted_s": round(sum(
            e.data.get("turn_s", 0) for e in sample.events
            if e.event == "info" and e.source == "turn_interrupt" and isinstance(e.data, dict)
        ), 1),
        "limit_s": (log_args or {}).get("time_limit_s"),
        "tool_calls": len(tools),
        "set_cell": sum(1 for e in tools if e.function == "set_cell"),
        "corrections": count_corrections(ok_set_cells),
        "failed": sum(1 for e in tools if e.error is not None),
        "out_tok": sum(u.output_tokens for u in usage),
        "reason_tok": sum(u.reasoning_tokens or 0 for u in usage),
        "in_tok": sum(
            u.input_tokens + (u.input_tokens_cache_read or 0) + (u.input_tokens_cache_write or 0)
            for u in usage
        ),
        "reported": report.get("code_used"),
    }


def timeline(sample) -> list[str]:
    start = sample.started_at
    if isinstance(start, str):  # the log stores it as ISO text
        start = datetime.fromisoformat(start)
    lines = []
    for e in sample.events:
        if getattr(e, "event", None) == "info" and getattr(e, "source", None) == "turn_interrupt":
            wall = (e.timestamp - start).total_seconds() if start else float("nan")
            lines.append(f"  wall +{wall:7.1f}s  INTERRUPTED after {e.data.get('turn_s')}s; agent told the time")
            continue
        if not isinstance(e, (ModelEvent, ToolEvent)):
            continue
        wall = (e.timestamp - start).total_seconds() if start else float("nan")
        took = f"{e.working_time:6.1f}s" if e.working_time is not None else "     ?"
        if isinstance(e, ModelEvent):
            u = e.output.usage if e.output else None
            what = f"model  out={u.output_tokens if u else '?'} reason={(u.reasoning_tokens or 0) if u else '?'}"
        else:
            args = ", ".join(f"{k}={str(v)[:30]}" for k, v in (e.arguments or {}).items())
            what = f"tool   {e.function}({args}){'  FAILED' if e.error else ''}"
        lines.append(f"  wall +{wall:7.1f}s  work +{e.working_start:7.1f}s  took {took}  {what}")
    return lines


def load(paths: list[Path], show_timeline: bool) -> list[dict]:
    rows = []
    for path in paths:
        try:
            log = read_eval_log(str(path), resolve_attachments=True)
        except Exception as exc:  # a corrupt or partial log should not stop the report
            print(f"skipping {path.name}: {exc}", file=sys.stderr)
            continue
        for sample in log.samples or []:
            rows.append(run_row(path, sample, log.eval.task_args))
            if show_timeline:
                print(f"\n{path.name}  [{sample.id}]")
                print("\n".join(timeline(sample)))
    return rows


def print_table(rows: list[dict], columns: list[str]) -> None:
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in columns}
    print("  ".join(c.ljust(widths[c]) for c in columns))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in columns))


def print_variability(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if not r["error"]:
            groups.setdefault((r["model"], r["variant"]), []).append(r)
    printed = False
    for (model, variant), runs in sorted(groups.items()):
        if len(runs) < 2:
            continue
        if not printed:
            print("\nVariability across completed runs (CV = sd / mean; lower is more repeatable)")
            printed = True
        print(f"\n{model}  {variant}  (n={len(runs)})")
        for m in MEASURES:
            values = [r[m] for r in runs]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            cv = sd / mean if mean else float("nan")
            print(f"  {m:<11} mean {mean:10.1f}   sd {sd:9.1f}   CV {cv:5.2f}")
    if not printed:
        print("\n(no model/variant has two completed runs yet, so no variability to report)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["logs"], help="log files or directories (default: logs/)")
    parser.add_argument("--timeline", action="store_true", help="print every model and tool call per run")
    parser.add_argument("--csv", help="also write the per-run rows to this CSV file")
    args = parser.parse_args()

    paths: list[Path] = []
    for p in map(Path, args.paths):
        paths += sorted(p.glob("*.eval")) if p.is_dir() else [p]
    rows = load(paths, args.timeline)
    if not rows:
        print("no runs found")
        return 1

    print()
    print_table(rows, ["log", "model", "variant", "limit_s", "turn_cap", "effort", "max_turn_s", "score", "error", *MEASURES,
                       "model_calls", "interrupts", "interrupted_s", "max_turn_reason", "tool_calls", "set_cell", "corrections", "failed", "reported"])
    print_variability(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
