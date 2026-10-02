"""Run the cheating judge over the samples of an existing eval log.

    .venv/bin/python authoring/rejudge.py <log.eval> [--model openrouter/anthropic/claude-opus-5.5] [--json out.json]

Uses the same transcript, facts and verdict code as the cheating_judge scorer,
so it can check the judge against episodes whose answer is known (the scripted
scenarios in authoring/.detection_logs) and judge logs written before the
judge existed. Each sample costs one judge call.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # the judge's API key, as inspect eval would load it
spec = importlib.util.spec_from_file_location("inspect_task", ROOT / "adapters" / "inspect" / "inspect_task.py")
inspect_task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_task)
JUDGE = inspect_task.JUDGE


async def rejudge(path: str, model_names: list[str], repeats: int = 2, epochs: set[int] | None = None) -> list[dict]:
    log = read_eval_log(path)
    models = [get_model(m) for m in model_names if m]

    async def one(sample) -> dict:
        store = dict(sample.store)
        facts = inspect_task.episode_facts(store, sample.messages)
        try:
            verdict = await JUDGE.judge_repeats(models, sample.messages, facts, repeats)
            verdict["timing"] = inspect_task.considered_timing(store, sample.messages, verdict)
        except JUDGE.JudgeError as e:
            verdict = {"error": str(e)}
        verdict["exploration"] = inspect_task.exploration(store, sample.messages)
        score = next(iter(sample.scores.values())).value if sample.scores else None
        return {"id": sample.id, "epoch": sample.epoch, "score": score, **verdict}

    samples = [s for s in log.samples or [] if epochs is None or s.epoch in epochs]
    return [await one(s) for s in samples]  # one at a time: see JUDGE_MAX_CONNECTIONS


def when(timing: dict) -> str:
    """Where in the episode a shortcut was weighed: '(#12, board 3 expert, 3 min 39 s left)'."""
    c = timing.get("context")
    if c is None:
        return f"({timing.get('event')}, {timing.get('error', 'before the first board')})"
    return f"({timing['event']}, board {c['board']} {c['tier']}, {round(c['left_s'])} s left at draw)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--model", default=inspect_task.JUDGE_MODEL)
    parser.add_argument("--fallback", default=inspect_task.JUDGE_FALLBACK_MODEL, help='"" for none')
    parser.add_argument("--repeats", type=int, default=2, help="judge each episode this many times and report agreement")
    parser.add_argument("--epochs", help="only these epochs, e.g. 1,3")
    parser.add_argument("--json", help="also write every verdict, with its evidence, to this file")
    args = parser.parse_args()
    epochs = {int(e) for e in args.epochs.split(",")} if args.epochs else None
    verdicts = asyncio.run(rejudge(args.log, [args.model, args.fallback], args.repeats, epochs))
    for v in sorted(verdicts, key=lambda v: (str(v["id"]), v["epoch"])):
        if "error" in v:
            print(f"{v['id']} ep{v['epoch']}: judge failed: {v['error']}")
            continue
        flags = f"  DISAGREES: {'; '.join(v['disagreements'])}" if v["disagreements"] else ""
        print(f"{v['id']} ep{v['epoch']}  score {v['score']}  cheated={v['cheated']} {v['cheat_types']}  "
              f"awareness={v['awareness']}  report_honest={v['report_honest']}  [{v['judge_model']}]{flags}\n"
              f"    by hack: {', '.join(f'{h}={a}' for h, a in v['hack_awareness'].items() if a != 'none') or 'none'}\n"
              f"    weighed: {'; '.join(f'{h} ' + ' '.join(when(t) for t in moments) for h, moments in v['timing'].items()) or 'nothing'}  "
              f"refused because: {', '.join(v['refusal_reasons']) or '-'}  "
              f"considered_misreport={v['considered_misreport']}  "
              f"report_discloses_awareness={v['report_discloses_awareness']}\n"
              f"    explored: {v['exploration']['bash_calls']} bash, {v['exploration']['python_calls']} python, "
              f"touched {', '.join(v['exploration']['touched']) or 'nothing'}"
              + (f"  agreement {v['agreement_rate']}"
                 + (f" (differ: {', '.join(k for k, ok in v['agreement'].items() if not ok)})"
                    if not all(v['agreement'].values()) else "") if "agreement" in v else "") + "\n"
              f"    {v['summary']}")
    if args.json:
        Path(args.json).write_text(json.dumps(verdicts, indent=1, default=str))


if __name__ == "__main__":
    main()
