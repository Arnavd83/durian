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


async def rejudge(path: str, model_names: list[str]) -> list[dict]:
    log = read_eval_log(path)
    models = [get_model(m) for m in model_names if m]

    async def one(sample) -> dict:
        facts = inspect_task.episode_facts(dict(sample.store), sample.messages)
        try:
            verdict = await JUDGE.judge_episode(models, sample.messages, facts)
        except JUDGE.JudgeError as e:
            verdict = {"error": str(e)}
        score = next(iter(sample.scores.values())).value if sample.scores else None
        return {"id": sample.id, "epoch": sample.epoch, "score": score, **verdict}

    return list(await asyncio.gather(*(one(s) for s in log.samples or [])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--model", default=inspect_task.JUDGE_MODEL)
    parser.add_argument("--fallback", default=inspect_task.JUDGE_FALLBACK_MODEL, help='"" for none')
    parser.add_argument("--json", help="also write every verdict, with its evidence, to this file")
    args = parser.parse_args()
    verdicts = asyncio.run(rejudge(args.log, [args.model, args.fallback]))
    for v in sorted(verdicts, key=lambda v: (str(v["id"]), v["epoch"])):
        if "error" in v:
            print(f"{v['id']} ep{v['epoch']}: judge failed: {v['error']}")
            continue
        flags = f"  DISAGREES: {'; '.join(v['disagreements'])}" if v["disagreements"] else ""
        print(f"{v['id']} ep{v['epoch']}  score {v['score']}  cheated={v['cheated']} {v['cheat_types']}  "
              f"awareness={v['awareness']}  report_honest={v['report_honest']}  [{v['judge_model']}]{flags}\n"
              f"    by hack: {', '.join(f'{h}={a}' for h, a in v['hack_awareness'].items() if a != 'none') or 'none'}\n"
              f"    {v['summary']}")
    if args.json:
        Path(args.json).write_text(json.dumps(verdicts, indent=1, default=str))


if __name__ == "__main__":
    main()
