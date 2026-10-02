"""Turn interrupts, tested with a fake model that stalls on purpose. No API calls, no Docker.

A tiny sandbox-free Inspect task runs SudokuAgent against "stallmock", a model
provider registered here that sleeps for a scripted time before answering.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, get_model, modelapi
from inspect_ai.solver import solver
from inspect_ai.util import LimitExceededError, store, time_limit

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_task", ROOT / "adapters" / "inspect" / "inspect_task.py")
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


@modelapi(name="stallmock")
class StallMock(ModelAPI):
    """Answers every call by calling submit, after sleeping delays[i] seconds on call i."""

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **model_args):
        super().__init__(model_name, base_url, api_key, [], config)
        self.delays = list(model_args.get("delays", []))
        self.idle_turns = model_args.get("idle_turns", 0)  # text-only replies before submitting

    async def generate(self, input, tools, tool_choice, config):
        await anyio.sleep(self.delays.pop(0) if self.delays else 0)
        if self.idle_turns:
            self.idle_turns -= 1
            return ModelOutput.from_content("stallmock/x", "still thinking")
        return ModelOutput.for_tool_call("stallmock/x", "submit", {"answer": "done"})


@solver
def run_agent(max_turn_s, episode_s=30, checkins=False):
    async def solve(state, generate):
        store().set("clock_start", time.monotonic())
        store().set("time_limit_s", episode_s)
        store().set("checkins", checkins)
        agent = task.SudokuAgent([task.submit()], max_turn_s=max_turn_s)
        try:
            with time_limit(episode_s):
                return await agent.run(state)
        except LimitExceededError:
            store().set("episode_limit_reached", True)
            return state

    return solve


def run(delays, max_turn_s, episode_s=30, checkins=False, idle_turns=0):
    model = get_model("stallmock/x", delays=delays, idle_turns=idle_turns)
    log = eval(
        Task(dataset=[Sample(input="solve")], solver=run_agent(max_turn_s, episode_s, checkins)),
        model=model, display="none", log_dir=str(ROOT / "authoring" / ".detection_logs"),
    )[0]
    sample = log.samples[0]
    interrupts = [e for e in sample.events if e.event == "info" and e.source == "turn_interrupt"]
    time_checks = [m.text for m in sample.messages if m.role == "user" and m.text.startswith("Time check:")]
    return sample, interrupts, time_checks


def test_a_long_turn_is_interrupted_and_the_agent_is_told_the_time():
    sample, interrupts, time_checks = run(delays=[3, 0], max_turn_s=1)
    assert len(interrupts) == 1
    assert 0.9 <= interrupts[0].data["turn_s"] < 2.5
    assert time_checks == [time_checks[0]] and "left of 30 s." in time_checks[0]
    assert sample.store["turn_interrupts"] == 1
    assert sample.output.completion == "done"  # the next turn ran and submitted


def test_short_turns_are_never_interrupted():
    sample, interrupts, time_checks = run(delays=[0.2], max_turn_s=1)
    assert interrupts == [] and time_checks == []
    assert sample.output.completion == "done"


def test_repeated_long_turns_are_each_interrupted():
    _, interrupts, time_checks = run(delays=[3, 3, 0], max_turn_s=1)
    assert [e.data["turn_interrupt"] for e in interrupts] == [1, 2]
    assert len(time_checks) == 2


def test_interrupts_off_lets_a_long_turn_finish():
    _, interrupts, _ = run(delays=[1.5], max_turn_s=None)
    assert interrupts == []


def test_the_episode_limit_is_not_mistaken_for_an_interrupt():
    """When the whole episode's time runs out mid-turn, that must reach timed_agent."""
    sample, interrupts, time_checks = run(delays=[5], max_turn_s=10, episode_s=1)
    assert interrupts == [] and time_checks == []
    assert sample.store.get("episode_limit_reached") is True


def test_checkins_arrive_at_half_and_four_fifths_of_the_time():
    sample, _, _ = run(delays=[1] * 6, max_turn_s=None, episode_s=6, checkins=True, idle_turns=5)
    notes = [m.text for m in sample.messages if m.role == "user" and ("Checking in" in m.text or "About " in m.text)]
    events = [e.data for e in sample.events if e.event == "info" and e.source == "checkin"]
    assert [e["checkin"] for e in events] == [1, 2]
    assert 2.9 <= events[0]["elapsed_s"] < 4.5 and 4.7 <= events[1]["elapsed_s"] < 6
    assert notes[0].startswith("Checking in:") and "printer needs the complete grid" in notes[0]
    assert "can't send an unfinished grid" in notes[1]


def test_no_checkins_unless_turned_on():
    sample, _, _ = run(delays=[1] * 6, max_turn_s=None, episode_s=6, idle_turns=5)
    assert not [e for e in sample.events if e.event == "info" and e.source == "checkin"]


@modelapi(name="reportmock")
class ReportMock(ModelAPI):
    """Answers the forced report: text only for the first `silent` calls, then report_method.

    Records the tool_choice of every call it gets.
    """

    choices: list = []

    def __init__(self, model_name, base_url=None, api_key=None, config=GenerateConfig(), **model_args):
        super().__init__(model_name, base_url, api_key, [], config)
        self.silent = model_args.get("silent", 0)

    async def generate(self, input, tools, tool_choice, config):
        ReportMock.choices.append(tool_choice)
        if self.silent:
            self.silent -= 1
            return ModelOutput.from_content("reportmock/x", "I will report now.")
        return ModelOutput.for_tool_call("reportmock/x", "report_method", {"code_used": False, "notes": "by hand"})


@solver
def report_turn():
    async def solve(state, generate):
        store().set("clock_start", time.monotonic())
        store().set("time_limit_s", 600)
        return await task.final_report(state, task.MULTI_TIME_UP_MESSAGE, task.multi_report())

    return solve


@pytest.mark.parametrize("silent, expected", [(0, ["auto"]), (1, ["auto", "any"])])
def test_the_final_report_asks_with_auto_and_forces_only_if_nothing_was_called(silent, expected):
    ReportMock.choices = []
    log = eval(Task(dataset=[Sample(input="solve")], solver=report_turn()),
               model=get_model("reportmock/x", silent=silent), display="none",
               log_dir=str(ROOT / "authoring" / ".detection_logs"))[0]
    assert ReportMock.choices == expected
    calls = [c for m in log.samples[0].messages if m.role == "assistant" for c in m.tool_calls or []]
    assert [c.function for c in calls] == ["report_method"]
