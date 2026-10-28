#!/usr/bin/env python3
"""Shared runner for the per-benchmark scripts in this directory.

An experiment is one directory under ``benchmark/experiments/<benchmark>/<run>/``.
Every task in that run shares its ``.hermes`` profile. Task outputs live under
``agent/<task-id>/`` inside that same directory. A second ``--run`` name gets
its own directory.

``--split`` selects a named subset when the benchmark publishes one (PaperBench
split files, or a ScienceAgentBench split such as verified, validation, train,
or test). ``--start`` and ``--end`` are 0-based inclusive indexes into that
ordered list. ``list`` prints those indexes.

``--turns`` lists conversation-turn budgets. A turn is one user message or one
assistant message that calls tools. Tool results stay on the tool-call turn
they answer. The runner grades after each time it invokes the agent and records
the conversation depth reached, so a solution that took several tool-call
rounds does not count as success@1.

* macro success@k — fraction of tasks that passed every test case on some stored conversation turn at or before k
* micro success@k — fraction of test cases that passed on some stored conversation turn at or before k
* auc@k — mean of those cumulative rates from 1 through k
* final macro/micro — pass rate on each task's last turn
* execution time and input/output/total tokens — mean and sample standard deviation,
  once over tasks whose last turn passed every case, and once over every completed task

``summarize --turns`` reports those k values from the grades in each task record.
A k that was not one of the budgets the agent ran is still valid.

Re-running the same ``--run`` skips tasks whose ``result.json`` status is ``complete``.
A task left ``running``, ``timeout``, or ``error`` is started over.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass

import jail
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CONTINUE_PROMPT = (
    "The benchmark verifier has not accepted the workspace yet. "
    "The task statement is the only spec. "
    "Re-check each required file and each numeric target in that statement yourself. "
    "If those checks already pass, stop without changing the design. "
    "If a written requirement is still unmet, change only that part and keep every file that already meets the statement."
)
_SKIP_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"}
_LFS_MARK = "version https://git-lfs.github.com/spec/v1"
_SKIP_HERMES_DIRS = {"sessions", "logs", "memories", "cache", "checkpoints", "state-snapshots", "skins"}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".md", ".txt"}
_COMPLETE = "complete"


class Spec:
    def __init__(self, name, checkout, repo_url, list_tasks, replicate, prepare, grade, *, uses_split=False, after_clone=None, finished=None):
        self.name = name
        self.checkout = checkout
        self.repo_url = repo_url
        self.list_tasks = list_tasks
        self.replicate = replicate
        self.prepare = prepare
        self.grade = grade
        self.uses_split = uses_split
        self.after_clone = after_clone
        self.finished = finished or verifier_finished


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def benchmark_root() -> Path:
    return repo_root() / "benchmark"


def relocate(text: str, mount: str, workspace: Path) -> str:
    """Rewrite a mount path only at a path boundary (``/root`` but not ``/rooted``)."""
    if not mount:
        return text
    dest = str(workspace)

    def replacer(_match: re.Match) -> str:
        return dest

    return re.sub(re.escape(mount) + r"(?=/|\b)", replacer, text)


def instruction_body(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


@dataclass(frozen=True)
class SkillSources:
    """Which skill sources one ``--skill-mode`` offers the agent.

    ``hermes`` is the agent's own bundled skills (``skills_list``, ``skill_view``,
    ``skill_manage``). ``benchmark`` is the task's attached skills.
    ``library`` is SkillLib.
    """

    hermes: bool
    benchmark: bool
    library: bool


_SKILL_SOURCES = {
    "task": SkillSources(hermes=True, benchmark=True, library=False),
    "library": SkillSources(hermes=False, benchmark=True, library=True),
    "library-compatible": SkillSources(hermes=True, benchmark=True, library=True),
    "library-only": SkillSources(hermes=False, benchmark=False, library=True),
    "none": SkillSources(hermes=False, benchmark=False, library=False),
}
SKILL_MODES = tuple(_SKILL_SOURCES)


def skill_sources(mode: str) -> SkillSources:
    try:
        return _SKILL_SOURCES[mode]
    except KeyError:
        raise SystemExit(f"unknown --skill-mode {mode!r}") from None


def disabled_toolsets(mode: str) -> list[str]:
    """Toolsets to subtract so a mode cannot call the sources it does not offer."""
    sources = skill_sources(mode)
    disabled = []
    if not sources.hermes:
        disabled.append("skills")
    if not sources.library:
        disabled.append("skill_library")
    return disabled


def ensure_hermes_skills(home: Path) -> None:
    """Copy this checkout's bundled skills into the experiment profile.

    The runner does not copy the source profile's ``skills/`` directory.
    Syncing here is what makes Hermes' own skills present for modes that
    include them, and it writes only under this experiment's ``.hermes``.
    """
    previous = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    try:
        from tools.skills_sync import sync_skills
        sync_skills(quiet=True)
    finally:
        if previous is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = previous


def skill_note(mode: str, workspace: Path, *, has_task_skills: bool) -> str:
    from agent.skill_library import task_instruction, tool_names
    sources = skill_sources(mode)
    parts: list[str] = []
    if sources.benchmark and has_task_skills:
        parts.append(
            f"This task ships skills under {workspace / 'skills'}. "
            "Read each SKILL.md and follow those procedures."
        )
    if sources.library:
        text = task_instruction()
        if text:
            parts.append(text)
    else:
        ban = f"Do not call {', '.join(tool_names())}."
        if not (sources.benchmark and has_task_skills):
            ban = "Solve this from the task statement and the files in the workspace. " + ban
        parts.append(ban)
    return "\n\n" + "\n\n".join(parts) if parts else ""


def layout_note(workspace: Path) -> str:
    names = sorted(path.name for path in workspace.iterdir()) if workspace.is_dir() else []
    if not names:
        return "\n\nThe workspace has no input files yet. Create what the task asks for here."
    listed = ", ".join(names)
    return (
        f"\n\nFiles copied from the task environment are in {workspace}: {listed}. "
        "If a path in the statement is not there yet, a setup script in this directory "
        "creates it. Run that script before solving."
    )


def copy_inputs(src: Path, dst: Path, *, include_skills: bool) -> None:
    if not src.is_dir():
        return

    def ignore(directory: str, names: list[str]) -> list[str]:
        drop = [name for name in names if name in _SKIP_FILES]
        if not include_skills and Path(directory) == src and "skills" in names:
            drop.append("skills")
        return drop

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def copy_tree(src: Path, dest: Path, skip_names: set[str]) -> None:
    if not src.exists():
        raise FileNotFoundError(src)

    def ignore(_directory: str, names: list[str]) -> list[str]:
        return [name for name in names if name in skip_names]

    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
        return
    shutil.copytree(src, dest, dirs_exist_ok=True, ignore=ignore)


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 2048:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:80]
    except OSError:
        return False
    return head.startswith(_LFS_MARK)


def task_dirs(parent: Path) -> list[Path]:
    if not parent.is_dir():
        return []
    return sorted(path for path in parent.iterdir() if path.is_dir())


def parse_turns(text: str) -> list[int]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        budget = int(part)
        if budget < 1:
            raise SystemExit(f"turn budget must be >= 1, got {budget}")
        values.append(budget)
    if not values:
        raise SystemExit("pass at least one turn budget, for example --turns 1,2,4,8")
    return sorted(set(values))


def unnamed_split_warning(benchmark: str, split: str | None) -> str:
    if split and split != "all":
        return (
            f"{benchmark} has no named split such as train or test. "
            "Omit --split, or pass --split all, and select tasks with --start and --end."
        )
    return ""


def window_tasks(
    tasks: list[dict],
    *,
    task: str | None = None,
    start: int | None = None,
    end: int | None = None,
    limit: int | None = None,
) -> list[tuple[int, dict]]:
    """Return ``(index, task)`` for an inclusive 0-based index range.

    Indexes refer to ``tasks`` before ``--task`` or ``--limit`` narrows them.
    """
    count = len(tasks)
    if start is None and end is None:
        chosen = list(enumerate(tasks))
    else:
        lo = 0 if start is None else start
        hi = count - 1 if end is None else end
        if lo < 0 or hi < 0 or lo > hi:
            raise ValueError(f"index range {lo}..{hi} is invalid; indexes are 0-based and inclusive")
        if count == 0 or lo >= count:
            chosen = []
        else:
            hi = min(hi, count - 1)
            chosen = [(index, tasks[index]) for index in range(lo, hi + 1)]
    if task:
        wanted = {item.strip() for item in task.split(",") if item.strip()}
        chosen = [(index, item) for index, item in chosen if item["id"] in wanted]
    if limit is not None:
        chosen = chosen[:limit]
    return chosen


def resolve_selection(spec: Spec, args) -> list[tuple[int, dict]]:
    tasks, warning = spec.list_tasks(benchmark_root(), getattr(args, "split", None))
    if warning:
        print(warning, file=sys.stderr)
    try:
        return window_tasks(
            tasks,
            task=getattr(args, "task", None),
            start=getattr(args, "start", None),
            end=getattr(args, "end", None),
            limit=getattr(args, "limit", None),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return []


def experiment_dir(benchmark: str, run: str) -> Path:
    if not run or run in {".", ".."} or "/" in run or "\\" in run:
        raise SystemExit("--run must be a single directory name")
    return benchmark_root() / "experiments" / benchmark / run


def source_hermes(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("HERMES_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / ".hermes"


def _config_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name.startswith("state.db") or path.suffix in {".db", ".db-wal", ".db-shm"}:
        return False
    if path.stat().st_size > 2_000_000:
        return False
    return path.name in {".env", "config.yaml", "SOUL.md"} or path.suffix in _CONFIG_SUFFIXES


def replicate_hermes_config(src: Path, dest: Path) -> None:
    """Copy profile configuration. Sessions, logs, and databases stay behind."""
    dest.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        return
    for path in src.iterdir():
        if path.name in _SKIP_HERMES_DIRS or path.name.startswith("." ) and path.name not in {".env"}:
            if path.name != ".env":
                continue
        if path.is_dir():
            continue
        if _config_file(path):
            shutil.copy2(path, dest / path.name)


def ensure_experiment(run_dir: Path, source: Path, meta: dict) -> None:
    home = run_dir / ".hermes"
    marker = home / ".replica-source"
    if not marker.exists():
        replicate_hermes_config(source, home)
        marker.write_text(str(source) + "\n", encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def result_path(run_dir: Path, task_id: str) -> Path:
    return run_dir / "agent" / task_id / "result.json"


def task_is_complete(run_dir: Path, task_id: str) -> bool:
    record = load_json(result_path(run_dir, task_id))
    return bool(record) and record.get("status") == _COMPLETE


def write_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def token_counts(usage: dict) -> tuple[int, int, int]:
    aux = usage.get("auxiliary") if isinstance(usage.get("auxiliary"), dict) else {}
    incoming = int(usage.get("input_tokens") or 0) + int(aux.get("input_tokens") or 0)
    outgoing = int(usage.get("output_tokens") or 0) + int(aux.get("output_tokens") or 0)
    if usage.get("total_tokens") is None and aux.get("total_tokens") is None:
        total = incoming + outgoing
    else:
        total = int(usage.get("total_tokens") or 0) + int(aux.get("total_tokens") or 0)
    return incoming, outgoing, total


def opens_conversation_turn(message: dict) -> bool:
    """A user message, or an assistant message that calls tools, is one turn."""
    role = message.get("role")
    if role == "user":
        return True
    return role == "assistant" and bool(message.get("tool_calls"))


def stamp_conversation_turns(messages: list[dict], start: int = 0) -> int:
    """Write the conversation turn onto each message and return the depth reached.

    Tool results keep the turn of the call they answer. An assistant reply with
    no tool call keeps the turn already reached.
    """
    depth = start
    for message in messages:
        if message.get("type") == "turn" or message.get("role") == "runner":
            continue
        if opens_conversation_turn(message):
            depth += 1
        if message.get("role") in {"user", "assistant", "tool"}:
            message["turn"] = max(depth, 1)
    return depth


def conversation_grade_turns(lines: list[dict]) -> list[int]:
    """Conversation depth at the end of each runner invocation in a transcript."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        if line.get("type") == "turn":
            if current:
                batches.append(current)
            current = []
            continue
        if line.get("role") in {"user", "assistant", "tool"}:
            current.append(line)
    if current:
        batches.append(current)
    depths = []
    previous = 0
    acc: list[dict] = []
    for batch in batches:
        acc.extend(batch)
        depth = stamp_conversation_turns([dict(item) for item in acc])
        if depth <= previous:
            depth = previous + 1
        previous = depth
        depths.append(depth)
    return depths


def mean_std(values: list[float]) -> dict:
    count = len(values)
    if count == 0:
        return {"mean": None, "std": None, "n": 0}
    average = statistics.fmean(values)
    if count == 1:
        return {"mean": average, "std": 0.0, "n": 1}
    return {"mean": average, "std": statistics.stdev(values), "n": count}


def _snapshots(record: dict, k: int) -> list[dict]:
    grades = record.get("grades") or []
    return [grade for grade in grades if int(grade.get("turn") or 0) <= k]


def macro_succeeded_by(record: dict, k: int) -> bool:
    for grade in _snapshots(record, k):
        cases = grade.get("cases") or []
        if cases and all(case.get("passed") for case in cases):
            return True
    return False


def _case_ids(record: dict) -> list[str]:
    seen = []
    for grade in record.get("grades") or []:
        for case in grade.get("cases") or []:
            ident = str(case.get("id"))
            if ident not in seen:
                seen.append(ident)
    return seen


def micro_units(records: list[dict], k: int) -> list[bool]:
    units = []
    for record in records:
        passed = {}
        for grade in _snapshots(record, k):
            for case in grade.get("cases") or []:
                ident = str(case.get("id"))
                passed[ident] = passed.get(ident, False) or bool(case.get("passed"))
        for ident in _case_ids(record):
            units.append(bool(passed.get(ident)))
    return units


def final_macro(record: dict) -> bool:
    grades = record.get("grades") or []
    if not grades:
        return False
    cases = grades[-1].get("cases") or []
    return bool(cases) and all(case.get("passed") for case in cases)


def final_micro_units(records: list[dict]) -> list[bool]:
    units = []
    for record in records:
        last = {}
        for grade in record.get("grades") or []:
            for case in grade.get("cases") or []:
                last[str(case.get("id"))] = bool(case.get("passed"))
        units.extend(last[ident] for ident in _case_ids(record))
    return units


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def _rate_or_none(records: list[dict]) -> float | None:
    if not records:
        return None
    return _rate([final_macro(record) for record in records])


_SKILL_TOOLS = ("skill_library_find", "skill_library_load", "skill_library_visit")


def empty_skill_library_usage() -> dict:
    return {
        "involved": False,
        "calls": {name: 0 for name in _SKILL_TOOLS},
        "find_queries": [],
        "cards_offered": [],
        "skills_loaded": [],
        "loads_with_gap": 0,
        "repositories_visited": [],
        "errors": 0,
        "first_turn": None,
    }


def _remember(items: list, value: str, limit: int) -> None:
    text = value.strip()
    if text and text not in items and len(items) < limit:
        items.append(text)


def _call_name_and_args(call: dict) -> tuple[str, dict]:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = str(function.get("name") or call.get("name") or "")
    raw = function.get("arguments") if function else call.get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
    elif isinstance(raw, dict):
        parsed = raw
    else:
        parsed = {}
    return name, parsed if isinstance(parsed, dict) else {}


def _payload(content) -> dict:
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _note_skill_call(usage: dict, name: str, args: dict, turn) -> None:
    usage["calls"][name] = int(usage["calls"].get(name) or 0) + 1
    usage["involved"] = True
    if usage.get("first_turn") is None and isinstance(turn, int):
        usage["first_turn"] = turn
    if name == "skill_library_find":
        _remember(usage["find_queries"], str(args.get("task") or ""), 20)
    elif name == "skill_library_load":
        _remember(usage["skills_loaded"], str(args.get("skill_id") or ""), 30)
    elif name == "skill_library_visit":
        _remember(usage["repositories_visited"], str(args.get("repository") or ""), 30)


def _note_skill_result(usage: dict, name: str, payload: dict) -> None:
    if payload.get("success") is False or payload.get("error"):
        usage["errors"] = int(usage.get("errors") or 0) + 1
    if name == "skill_library_find":
        for card in payload.get("cards") or []:
            if isinstance(card, dict):
                _remember(usage["cards_offered"], str(card.get("id") or card.get("name") or ""), 40)
    elif name == "skill_library_load" and payload.get("gap"):
        usage["loads_with_gap"] = int(usage.get("loads_with_gap") or 0) + 1


def observe_skill_library(usage: dict, lines: list[dict]) -> dict:
    """Count skill_library_find, load, and visit calls in trajectory lines."""
    pending: dict[str, str] = {}
    for line in lines:
        if not isinstance(line, dict) or line.get("type") == "turn":
            continue
        role = line.get("role")
        if role == "assistant":
            for call in line.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name, args = _call_name_and_args(call)
                if name not in _SKILL_TOOLS:
                    continue
                _note_skill_call(usage, name, args, line.get("turn"))
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    pending[call_id] = name
        elif role == "tool":
            name = pending.pop(str(line.get("tool_call_id") or ""), "")
            if name:
                _note_skill_result(usage, name, _payload(line.get("content")))
    return usage


def skill_library_summary(records: list[dict]) -> dict:
    """Calling behavior, and final success split by whether those calls happened.

    The split describes the completed tasks. It is not an estimate of causation.
    """
    def usage_of(record: dict) -> dict:
        usage = record.get("skill_library")
        if isinstance(usage, dict) and isinstance(usage.get("calls"), dict):
            return usage
        return empty_skill_library_usage()

    involved = [record for record in records if usage_of(record).get("involved")]
    not_involved = [record for record in records if not usage_of(record).get("involved")]
    loaded = [record for record in records if usage_of(record).get("skills_loaded")]
    find_only = [
        record for record in involved
        if not usage_of(record).get("skills_loaded")
    ]
    totals = {name: 0 for name in _SKILL_TOOLS}
    series = {name: [] for name in _SKILL_TOOLS}
    cards: set[str] = set()
    skills: list[str] = []
    gaps = 0
    errors = 0
    visited = 0
    for record in records:
        usage = usage_of(record)
        calls = usage.get("calls") or {}
        for name in _SKILL_TOOLS:
            count = int(calls.get(name) or 0)
            totals[name] += count
            series[name].append(float(count))
        for card in usage.get("cards_offered") or []:
            cards.add(str(card))
        for skill in usage.get("skills_loaded") or []:
            if skill not in skills:
                skills.append(str(skill))
        gaps += int(usage.get("loads_with_gap") or 0)
        errors += int(usage.get("errors") or 0)
        if usage.get("repositories_visited"):
            visited += 1
    return {
        "tasks_involved": len(involved),
        "tasks_not_involved": len(not_involved),
        "tasks_loaded": len(loaded),
        "tasks_find_only": len(find_only),
        "tasks_visited": visited,
        "calls_total": totals,
        "calls_per_task": {name: mean_std(values) for name, values in series.items()},
        "final_macro_success": {
            "involved": _rate_or_none(involved),
            "not_involved": _rate_or_none(not_involved),
            "loaded": _rate_or_none(loaded),
            "find_only": _rate_or_none(find_only),
        },
        "unique_cards_offered": len(cards),
        "unique_skills_loaded": len(skills),
        "skills_loaded": skills[:30],
        "loads_with_gap": gaps,
        "errors": errors,
    }


def success_curve(records: list[dict], k: int, *, micro: bool) -> list[float]:
    curve = []
    for turn in range(1, k + 1):
        if micro:
            curve.append(_rate(micro_units(records, turn)))
        else:
            curve.append(_rate([macro_succeeded_by(record, turn) for record in records]))
    return curve


def summarize_records(records: list[dict], turns: list[int], *, benchmark: str, run: str) -> dict:
    """Aggregate completed task records. ``turns`` are the reported k values."""
    complete = [record for record in records if record.get("status") == _COMPLETE]
    budgets = sorted(set(turns))
    max_k = budgets[-1] if budgets else 1
    macro_curve = success_curve(complete, max_k, micro=False)
    micro_curve = success_curve(complete, max_k, micro=True)
    successful = [record for record in complete if final_macro(record)]

    def at(curve: list[float]) -> dict:
        return {str(k): curve[k - 1] for k in budgets}

    def auc(curve: list[float]) -> dict:
        return {str(k): (sum(curve[:k]) / k if k else 0.0) for k in budgets}

    def cost_block(field: str) -> dict:
        return {
            "successful_tasks": mean_std([float(record.get(field) or 0) for record in successful]),
            "all_tasks": mean_std([float(record.get(field) or 0) for record in complete]),
        }

    unjudged = 0
    for record in complete:
        grades = record.get("grades") or []
        if not grades:
            continue
        for case in grades[-1].get("cases") or []:
            if case.get("graded") is False:
                unjudged += 1
    return {
        "benchmark": benchmark,
        "run": run,
        "turns": budgets,
        "n_complete": len(complete),
        "n_successful": len(successful),
        "unjudged_cases": unjudged,
        "macro_success_at_k": at(macro_curve),
        "micro_success_at_k": at(micro_curve),
        "macro_auc_at_k": auc(macro_curve),
        "micro_auc_at_k": auc(micro_curve),
        "final_macro_success": _rate([final_macro(record) for record in complete]),
        "final_micro_success": _rate(final_micro_units(complete)),
        "execution_time_sec": cost_block("duration_sec"),
        "input_tokens": cost_block("input_tokens"),
        "output_tokens": cost_block("output_tokens"),
        "total_tokens": cost_block("total_tokens"),
        "skill_library": skill_library_summary(complete),
        "definitions": {
            "conversation_turn": "One user message, or one assistant message that calls tools, inside the task conversation. Tool results belong to the tool-call turn they answer.",
            "macro_success_at_k": "Fraction of completed tasks that passed every test case on any stored conversation turn at or before k.",
            "micro_success_at_k": "Fraction of test cases that passed on any stored conversation turn at or before k.",
            "auc_at_k": "Mean of success@1 through success@k. Each success@t counts passes at or before t.",
            "final_success": "Pass rate on each completed task's last turn.",
            "successful_tasks": "Completed tasks whose last turn passed every test case.",
            "all_tasks": "Completed tasks. Interrupted tasks are omitted until a rerun finishes them.",
            "skill_library": (
                "Counts of skill_library_find, skill_library_load, and skill_library_visit in the trajectory. "
                "involved means at least one of those calls. loaded means skill_library_load was called. "
                "find_only means find was called and load was not. "
                "final_macro_success splits the last-turn pass rate by that calling behavior."
            ),
        },
    }


def format_report(summary: dict, incomplete: list[str]) -> str:
    lines = [
        f"benchmark: {summary['benchmark']}",
        f"run: {summary['run']}",
        f"model: {summary.get('model') or ''}",
        f"provider: {summary.get('provider') or ''}",
        "turns: " + ",".join(str(k) for k in summary["turns"]),
        "success@k counts a pass on any conversation turn at or before k",
        f"completed: {summary['n_complete']}",
        f"successful: {summary['n_successful']}",
        f"unjudged_cases: {summary['unjudged_cases']}",
        "",
    ]
    for label, key in (
        ("macro_success", "macro_success_at_k"),
        ("micro_success", "micro_success_at_k"),
        ("macro_auc", "macro_auc_at_k"),
        ("micro_auc", "micro_auc_at_k"),
    ):
        for budget, value in summary[key].items():
            lines.append(f"{label}@{budget} = {value:.4f}")
    lines.append(f"final_macro_success = {summary['final_macro_success']:.4f}")
    lines.append(f"final_micro_success = {summary['final_micro_success']:.4f}")
    lines.append("")
    for field in ("execution_time_sec", "input_tokens", "output_tokens", "total_tokens"):
        for group in ("successful_tasks", "all_tasks"):
            stats = summary[field][group]
            mean = "n/a" if stats["mean"] is None else f"{stats['mean']:.4f}"
            std = "n/a" if stats["std"] is None else f"{stats['std']:.4f}"
            lines.append(f"{field} {group} mean={mean} std={std} n={stats['n']}")
    library = summary.get("skill_library") or {}
    if library:
        calls = library.get("calls_total") or {}
        split = library.get("final_macro_success") or {}
        lines.extend([
            "",
            (
                "skill_library "
                f"involved={library.get('tasks_involved', 0)}/{summary['n_complete']} "
                f"find={calls.get('skill_library_find', 0)} "
                f"load={calls.get('skill_library_load', 0)} "
                f"visit={calls.get('skill_library_visit', 0)} "
                f"errors={library.get('errors', 0)} "
                f"loads_with_gap={library.get('loads_with_gap', 0)}"
            ),
            (
                "skill_library final_macro "
                f"involved={_fmt_rate(split.get('involved'))} "
                f"not_involved={_fmt_rate(split.get('not_involved'))} "
                f"loaded={_fmt_rate(split.get('loaded'))} "
                f"find_only={_fmt_rate(split.get('find_only'))}"
            ),
            (
                "skill_library "
                f"cards_offered={library.get('unique_cards_offered', 0)} "
                f"skills_loaded={library.get('unique_skills_loaded', 0)} "
                f"tasks_visited={library.get('tasks_visited', 0)}"
            ),
        ])
        loaded_names = library.get("skills_loaded") or []
        if loaded_names:
            lines.append("skill_library loaded_ids=" + ",".join(loaded_names))
    if incomplete:
        lines.extend(["", "incomplete: " + ",".join(incomplete)])
    lines.append("")
    return "\n".join(lines)


def _fmt_rate(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _trajectory_lines_by_task(run_dir: Path) -> dict[str, list[dict]]:
    path = experiment_trajectory(run_dir)
    grouped: dict[str, list[dict]] = {}
    if not path.is_file():
        return grouped
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("task"):
            grouped.setdefault(str(item["task"]), []).append(item)
    return grouped


def skill_library_from_trajectory(run_dir: Path) -> dict[str, dict]:
    """Rebuild per-task SkillLib usage from the shared transcript."""
    return {
        task_id: observe_skill_library(empty_skill_library_usage(), lines)
        for task_id, lines in _trajectory_lines_by_task(run_dir).items()
    }


def attach_skill_library(run_dir: Path, records: list[dict]) -> None:
    """Use usage stored on the task, otherwise the transcript."""
    parsed = skill_library_from_trajectory(run_dir)
    for record in records:
        stored = record.get("skill_library")
        if isinstance(stored, dict) and isinstance(stored.get("calls"), dict):
            continue
        record["skill_library"] = parsed.get(str(record.get("id"))) or empty_skill_library_usage()


def attach_conversation_turns(run_dir: Path, records: list[dict]) -> None:
    """Retarget stored grades to conversation depth when the transcript is deeper.

    Older runs labeled every message of one agent call as turn 1. The transcript
    still has the user and tool-call messages, so summarize can recover the depth.
    """
    grouped = _trajectory_lines_by_task(run_dir)
    for record in records:
        lines = grouped.get(str(record.get("id") or ""))
        if not lines:
            continue
        depths = conversation_grade_turns(lines)
        grades = record.get("grades") or []
        if not grades or not depths:
            continue
        if len(depths) != len(grades):
            if len(grades) != 1:
                continue
            depths = [depths[-1]]
        changed = False
        for grade, depth in zip(grades, depths):
            if int(grade.get("turn") or 0) != depth:
                grade["turn"] = depth
                changed = True
        if not changed:
            continue
        record["conversation_turns"] = depths[-1]
        path = result_path(run_dir, str(record.get("id")))
        if path.is_file():
            stored = load_json(path) or record
            stored["grades"] = grades
            stored["conversation_turns"] = depths[-1]
            write_result(path, stored)


def load_run_records(run_dir: Path) -> tuple[list[dict], list[str]]:
    agent = run_dir / "agent"
    complete = []
    incomplete = []
    if not agent.is_dir():
        return complete, incomplete
    for task_dir in sorted(path for path in agent.iterdir() if path.is_dir()):
        record = load_json(task_dir / "result.json")
        if not record:
            incomplete.append(task_dir.name)
            continue
        if record.get("status") == _COMPLETE:
            complete.append(record)
        else:
            incomplete.append(task_dir.name)
    return complete, incomplete


def write_summary(run_dir: Path, summary_path: Path, report_path: Path, turns: list[int], benchmark: str, run: str) -> dict:
    complete, incomplete = load_run_records(run_dir)
    attach_conversation_turns(run_dir, complete)
    attach_skill_library(run_dir, complete)
    summary = summarize_records(complete, turns, benchmark=benchmark, run=run)
    meta = load_json(run_dir / "run.json") or {}
    summary["model"] = meta.get("model")
    summary["provider"] = meta.get("provider")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report(summary, incomplete), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"), end="")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return summary


def summary_paths(run_dir: Path, summary: str | None, report: str | None) -> tuple[Path, Path]:
    summary_path = Path(summary).expanduser() if summary else run_dir / "summary.json"
    report_path = Path(report).expanduser() if report else run_dir / "summary.txt"
    if not summary_path.is_absolute():
        summary_path = Path.cwd() / summary_path
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    return summary_path, report_path


def stable_case_id(classname: str, name: str) -> str:
    if classname and "/" not in classname and "\\" not in classname:
        return f"{classname}::{name}"
    return name


def parse_junit(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    root = ET.parse(path).getroot()
    cases = []
    for node in root.iter("testcase"):
        name = node.attrib.get("name") or "test"
        failure = node.find("failure")
        error = node.find("error")
        skipped = node.find("skipped") is not None
        detail = ""
        problem = failure if failure is not None else error
        if problem is not None:
            detail = (problem.attrib.get("message") or problem.text or "")[:400]
        cases.append({
            "id": stable_case_id(node.attrib.get("classname") or "", name),
            "passed": problem is None and not skipped,
            "detail": detail,
            "graded": True,
        })
    return cases


def verifier_finished(_replica: Path, _workspace: Path, cases: list[dict]) -> bool:
    """Stop when every graded check passed.

    Ungraded checks, such as PaperBench rubric leaves before ``judge.json``,
    do not keep the conversation going.
    """
    graded = [case for case in cases if case.get("graded") is not False]
    return bool(graded) and all(case.get("passed") for case in graded)


def followup_prompt(cases: list[dict]) -> str:
    """Tell the agent the workspace is not accepted yet.

    Check ids and assertion text stay with the grader. ``cases`` is unused so a
    later turn cannot quote them. The message also tells the agent to stop when
    its own check of the task statement already passes, so a hidden failure does
    not start a new design.
    """
    del cases
    return CONTINUE_PROMPT


def uv_executable() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def pytest_command(test_path: Path, report: Path, root: Path) -> list[str]:
    """Run the benchmark's pytest file. Use uv when this interpreter has no pytest."""
    import importlib.util

    args = [
        "-m", "pytest", str(test_path), "-q",
        "--confcutdir", str(root), "--rootdir", str(root),
        f"--junitxml={report}", "-p", "no:cacheprovider",
    ]
    if importlib.util.find_spec("pytest") is not None:
        return [sys.executable, *args]
    uv = uv_executable()
    if not uv:
        return []
    return [uv, "run", "--no-project", "--python", sys.executable, "--with", "pytest>=8,<9", "python", *args]


def grade_with_pytest(test_file: Path, workspace: Path, mounts: list[tuple[str, Path]]) -> list[dict]:
    """Run a copied verifier with mount paths rewritten to this run's workspace.

    SkillsBench and Terminal-Bench decide a task by this pytest file. Their
    ``test.sh`` scripts also install packages, so those scripts are not run.
    """
    if not test_file.is_file():
        return [{"id": "verifier", "passed": False, "detail": f"missing {test_file}", "graded": True, "fatal": True}]
    tmp = Path(tempfile.mkdtemp(prefix="bench-grade-"))
    try:
        text = test_file.read_text(encoding="utf-8")
        for mount, dest in mounts:
            text = relocate(text, mount, dest)
        copied = tmp / test_file.name
        copied.write_text(text, encoding="utf-8")
        # A verifier may consist of several sibling modules: the test file
        # imports helpers that live next to it (in the task's Docker image
        # the whole verifier dir is present and test.sh runs it in place).
        # Copy the siblings, with the same mount relocations applied, so the
        # relocated test can import them from its new location.
        for sibling in sorted(test_file.parent.iterdir()):
            if not sibling.is_file() or sibling.name == test_file.name:
                continue
            try:
                sib_text = sibling.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                shutil.copy2(sibling, tmp / sibling.name)
                continue
            for mount, dest in mounts:
                sib_text = relocate(sib_text, mount, dest)
            (tmp / sibling.name).write_text(sib_text, encoding="utf-8")
        report = tmp / "junit.xml"
        command = pytest_command(copied, report, tmp)
        if not command:
            return [{
                "id": "verifier",
                "passed": False,
                "detail": "pytest is not installed and uv is not available to run the verifier",
                "graded": True,
                "fatal": True,
            }]
        try:
            proc = subprocess.run(
                command,
                cwd=str(tmp),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return [{"id": "pytest", "passed": False, "detail": "pytest timed out", "graded": True}]
        cases = parse_junit(report)
        if cases:
            return cases
        detail = (proc.stdout or proc.stderr or "pytest produced no tests")[-400:]
        return [{"id": "pytest", "passed": False, "detail": detail, "graded": True, "fatal": proc.returncode != 0 and not report.is_file()}]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rubric_leaves(node: dict) -> list[dict]:
    leaves = []

    def walk(item: dict) -> None:
        children = item.get("sub_tasks") or []
        if children:
            for child in children:
                if isinstance(child, dict):
                    walk(child)
            return
        leaves.append({
            "id": str(item.get("id") or item.get("requirements") or "leaf"),
            "passed": False,
            "detail": str(item.get("requirements") or "")[:240],
            "graded": False,
        })

    if isinstance(node, dict):
        walk(node)
    return leaves


def apply_judge(cases: list[dict], judge: dict) -> list[dict]:
    leaves = judge.get("leaves") if isinstance(judge.get("leaves"), dict) else judge
    graded = []
    for case in cases:
        if case["id"] in leaves:
            graded.append({**case, "passed": bool(leaves[case["id"]]), "graded": True, "detail": case.get("detail") or ""})
        else:
            graded.append(case)
    return graded


def list_session_ids(home: Path) -> list[str]:
    """Session ids in this experiment profile, oldest first."""
    db = home / "state.db"
    if not db.is_file():
        return []
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return [row[0] for row in con.execute("SELECT id FROM sessions ORDER BY started_at, rowid")]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def sessions_started_since(before: list[str], after: list[str]) -> list[str]:
    seen = set(before)
    return [session_id for session_id in after if session_id not in seen]


def experiment_trajectory(run_dir: Path) -> Path:
    return run_dir / "trajectory.jsonl"


def _trajectory_message(task_id: str, turn: int, session_id: str, message: dict) -> dict:
    line = {"task": task_id, "turn": turn, "session_id": session_id, "role": message.get("role")}
    for key in ("content", "tool_calls", "tool_call_id", "name", "reasoning"):
        value = message.get(key)
        if value not in (None, "", []):
            line[key] = value
    return line


def load_trajectory_messages(home: Path, session_ids: list[str], task_id: str) -> list[dict]:
    """Read the messages Hermes stored for the given sessions."""
    if not session_ids:
        return []
    from agent.trace_upload import load_session_messages

    db = home / "state.db"
    lines = []
    for session_id in session_ids:
        messages, _meta = load_session_messages(session_id, db_path=db)
        for message in messages:
            if isinstance(message, dict):
                lines.append(_trajectory_message(task_id, 0, session_id, message))
    return lines


def _logged_message_count(path: Path, task_id: str, session_id: str) -> int:
    if not path.is_file():
        return 0
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(item, dict)
            and item.get("task") == task_id
            and item.get("session_id") == session_id
            and item.get("role") in {"user", "assistant", "tool"}
        ):
            count += 1
    return count


def append_turn_trajectory(path: Path, home: Path, task_id: str, runner_turn: int, before: list[str], exit_code, stdout: str, stderr: str, *, prior_depth: int = 0, resume_session: str | None = None) -> list[dict]:
    after = list_session_ids(home)
    new_ids = sessions_started_since(before, after)
    session_ids = new_ids or ([resume_session] if resume_session else [])
    try:
        messages = load_trajectory_messages(home, session_ids, task_id)
    except Exception as exc:
        messages = [{
            "task": task_id,
            "session_id": None,
            "role": "runner",
            "content": f"could not read session: {exc}",
        }]
    else:
        if resume_session and resume_session not in new_ids:
            already = _logged_message_count(path, task_id, resume_session)
            messages = [line for line in messages if line.get("session_id") == resume_session][already:]
    depth = stamp_conversation_turns(messages, prior_depth)
    if depth <= prior_depth:
        depth = prior_depth + 1
    for message in messages:
        if message.get("role") == "runner":
            message["turn"] = depth
    lines: list[dict] = [{
        "task": task_id,
        "turn": depth,
        "runner_turn": runner_turn,
        "type": "turn",
        "exit_code": exit_code,
        "session_ids": session_ids,
    }]
    lines.extend(messages)
    if len(lines) == 1:
        note = (stdout or stderr or "").strip()
        if note:
            lines.append({
                "task": task_id,
                "turn": depth,
                "session_id": None,
                "role": "runner",
                "content": note[-4000:],
            })
    append_trajectory_lines(path, lines)
    return lines


def append_trajectory_lines(path: Path, lines: list[dict]) -> None:
    if not lines:
        return
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    _append(path, text)


def forget_task_trajectory(path: Path, task_id: str) -> None:
    """Drop one task's lines so a rerun replaces them in the shared transcript."""
    if not path.is_file():
        return
    kept = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            kept.append(raw)
            continue
        if isinstance(item, dict) and item.get("task") == task_id:
            continue
        kept.append(raw)
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def invoke_worker(*, workspace, prompt, usage, hermes_home, resume, model, provider, skill_mode, timeout):
    prompt_path = Path(usage).parent / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "_worker",
        "--workspace", str(workspace),
        "--prompt", str(prompt_path),
        "--usage", str(usage),
        "--hermes-home", str(hermes_home),
    ]
    if resume:
        cmd.extend(["--resume", resume])
    if model:
        cmd.extend(["--model", model])
    if provider:
        cmd.extend(["--provider", provider])
    cmd.extend(["--skill-mode", skill_mode])
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    agent_dir = Path(usage).parents[2]
    scratch = Path(tempfile.mkdtemp(prefix="bench-jail-"))
    env_file = scratch / "env"
    blank_file = scratch / "blank"
    blank_file.write_text("", encoding="utf-8")
    jail.write_env_file(env_file, env)
    try:
        argv = jail.command(
            cmd,
            workspace=Path(workspace),
            hermes_home=Path(hermes_home),
            agent_dir=agent_dir,
            env_file=env_file,
            blank_file=blank_file,
        )
        proc = subprocess.run(
            argv, cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or "", exc.stderr or ""
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _run_task_turns(spec, record, agent_dir, replica, workspace, prompt, home, *, trajectory_path, task_id, skill_mode, turns_max, model, provider, timeout, started, deadline, session_id):
    forget_task_trajectory(trajectory_path, task_id)
    runner_turn = 0
    while runner_turn < turns_max:
        runner_turn += 1
        remaining = None if deadline is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0:
            record["status"] = "timeout"
            record["error"] = f"timed out after {timeout}s"
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        turn_dir = agent_dir / "turns" / str(runner_turn)
        turn_dir.mkdir(parents=True, exist_ok=True)
        usage_path = turn_dir / "usage.json"
        previous = record["grades"][-1]["cases"] if record["grades"] else []
        message = prompt if runner_turn == 1 else followup_prompt(previous)
        before = list_session_ids(home)
        code, stdout, stderr = invoke_worker(
            workspace=workspace,
            prompt=message,
            usage=usage_path,
            hermes_home=home,
            resume=session_id,
            model=model,
            provider=provider,
            skill_mode=skill_mode,
            timeout=remaining,
        )
        _append(agent_dir / "agent.stdout", f"\n--- runner turn {runner_turn} ---\n{stdout}")
        _append(agent_dir / "agent.stderr", f"\n--- runner turn {runner_turn} ---\n{stderr}")
        try:
            transcript = append_turn_trajectory(
                trajectory_path, home, task_id, runner_turn, before, code, stdout, stderr,
                prior_depth=int(record.get("conversation_turns") or 0),
                resume_session=session_id,
            )
        except OSError as exc:
            transcript = []
            print(f"trajectory turn {runner_turn}: {exc}", file=sys.stderr)
        depth = next((int(line["turn"]) for line in transcript if line.get("type") == "turn"), int(record.get("conversation_turns") or 0) + 1)
        record["conversation_turns"] = depth
        observe_skill_library(record.setdefault("skill_library", empty_skill_library_usage()), transcript)
        usage = load_json(usage_path) or {}
        incoming, outgoing, total = token_counts(usage)
        record["input_tokens"] += incoming
        record["output_tokens"] += outgoing
        record["total_tokens"] += total
        if usage.get("session_id"):
            session_id = str(usage["session_id"])
        if code is None:
            record["status"] = "timeout"
            record["error"] = f"timed out after {timeout}s"
            record["turns_run"] = depth
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        try:
            cases = spec.grade(replica, workspace, agent_dir)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record["turns_run"] = depth
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        record["grades"].append({"turn": depth, "cases": cases})
        record["turns_run"] = depth
        record["duration_sec"] = time.perf_counter() - started
        write_result(agent_dir / "result.json", record)
        if any(case.get("fatal") for case in cases):
            record["status"] = "error"
            record["error"] = next(case.get("detail") for case in cases if case.get("fatal"))
            write_result(agent_dir / "result.json", record)
            return record
        if spec.finished(replica, workspace, cases) or depth >= turns_max:
            break
    record["status"] = _COMPLETE
    record["duration_sec"] = time.perf_counter() - started
    write_result(agent_dir / "result.json", record)
    return record


def run_task(spec: Spec, run_dir: Path, task: dict, *, skill_mode: str, turns: list[int], model, provider, timeout) -> dict:
    task_id = task["id"]
    agent_dir = run_dir / "agent" / task_id
    replica = run_dir / "tasks" / task_id
    for path in (agent_dir, replica):
        if path.exists():
            shutil.rmtree(path)
    spec.replicate(benchmark_root(), task_id, replica)
    workspace = agent_dir / "workspace"
    workspace.mkdir(parents=True)
    prompt, error = spec.prepare(replica, workspace, skill_mode)
    (agent_dir / "prompt.txt").write_text((prompt or "") + ("\n" if prompt else ""), encoding="utf-8")
    record = {
        "bench": spec.name,
        "id": task_id,
        "skill_mode": skill_mode,
        "model": model,
        "provider": provider,
        "workspace": str(workspace),
        "task_replica": str(replica),
        "hermes_home": "",
        "status": "running",
        "grades": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "duration_sec": 0.0,
        "turns_run": 0,
        "skill_library": empty_skill_library_usage(),
    }
    home = run_dir / ".hermes"
    record["hermes_home"] = str(home)
    try:
        if error:
            record["status"] = "error"
            record["error"] = error
            write_result(agent_dir / "result.json", record)
            return record
        write_result(agent_dir / "result.json", record)
        started = time.perf_counter()
        deadline = None if timeout is None else started + timeout
        max_k = max(turns)
        trajectory_path = experiment_trajectory(run_dir)
        record["trajectory"] = str(trajectory_path)
        return _run_task_turns(
            spec, record, agent_dir, replica, workspace, prompt, home,
            trajectory_path=trajectory_path, task_id=task_id,
            skill_mode=skill_mode, turns_max=max_k, model=model, provider=provider,
            timeout=timeout, started=started, deadline=deadline, session_id=None,
        )
    finally:
        try:
            record["trajectory"] = str(experiment_trajectory(run_dir))
            write_result(agent_dir / "result.json", record)
        except OSError as exc:
            print(f"trajectory: {exc}", file=sys.stderr)


def regrade_task(spec: Spec, run_dir: Path, task_id: str) -> dict | None:
    record = load_json(result_path(run_dir, task_id))
    if not record or not record.get("grades"):
        return record
    replica = Path(record["task_replica"])
    workspace = Path(record["workspace"])
    cases = spec.grade(replica, workspace, result_path(run_dir, task_id).parent)
    record["grades"][-1]["cases"] = cases
    if record.get("status") == "running":
        record["status"] = _COMPLETE
    write_result(result_path(run_dir, task_id), record)
    return record


def _checkout_present(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    try:
        return any(path.name != ".git" for path in dest.iterdir())
    except OSError:
        return False


def clone_repo(spec: Spec, root: Path) -> str:
    """Check out one benchmark. The recorded path is a submodule of this repo."""
    dest = root / spec.checkout
    if _checkout_present(dest):
        return f"{spec.name}: already at {dest}"
    repo = repo_root()
    rel = dest.relative_to(repo)
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"
    initialized = subprocess.run(
        ["git", "-C", str(repo), "submodule", "update", "--init", "--", str(rel)],
        env=env,
    )
    if initialized.returncode == 0 and _checkout_present(dest):
        if spec.after_clone:
            spec.after_clone(dest)
        return f"{spec.name}: initialized submodule {dest}"
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", spec.repo_url, str(dest)], check=True, env=env)
    if spec.after_clone:
        spec.after_clone(dest)
    return f"{spec.name}: cloned {dest}"


def load_mapping(text: str) -> dict:
    """Read a config document. The project venv has ruamel; tests may have PyYAML."""
    try:
        from ruamel.yaml import YAML
        data = YAML(typ="safe").load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def read_profile_model(config_path: Path) -> tuple[str, str, dict[str, str]]:
    """Return ``(model.default, model.provider, {provider name: default_model})``."""
    data = load_mapping(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    model_cfg = data.get("model")
    default_model = ""
    default_provider = ""
    if isinstance(model_cfg, str):
        default_model = model_cfg.strip()
    elif isinstance(model_cfg, dict):
        raw = model_cfg.get("default") or model_cfg.get("model") or ""
        default_model = str(raw).strip() if not isinstance(raw, dict) else ""
        default_provider = str(model_cfg.get("provider") or "").strip()
    provider_models: dict[str, str] = {}
    providers = data.get("providers")
    if isinstance(providers, dict):
        for name, info in providers.items():
            if isinstance(info, dict) and info.get("default_model"):
                provider_models[str(name)] = str(info["default_model"]).strip()
    return default_model, default_provider, provider_models


def resolve_agent_model(config_path: Path, model: str | None, provider: str | None) -> tuple[str | None, str | None]:
    """Pair a requested model with a provider from the replicated profile.

    Both omitted: return ``(None, None)`` and let Hermes use the profile as written.
    Provider only: use that provider's ``default_model``, then ``model.default``.
    Model only: keep ``model.provider`` from the profile instead of auto-detecting a vendor.
    """
    default_model, default_provider, provider_models = read_profile_model(config_path)
    chosen_provider = (provider or "").strip() or None
    chosen_model = (model or "").strip() or None
    if chosen_provider and not chosen_model:
        chosen_model = provider_models.get(chosen_provider) or default_model or None
    if chosen_model and not chosen_provider:
        chosen_provider = default_provider or None
    return chosen_model, chosen_provider


def _turns_from(args, run_dir: Path) -> list[int]:
    if getattr(args, "turns", None):
        return parse_turns(args.turns)
    meta = load_json(run_dir / "run.json") or {}
    saved = meta.get("turns")
    if isinstance(saved, list) and saved:
        return [int(item) for item in saved]
    return [1, 2, 4, 8]


def cmd_clone(args, spec: Spec) -> int:
    print(clone_repo(spec, benchmark_root()))
    return 0


def cmd_list(args, spec: Spec) -> int:
    chosen = resolve_selection(spec, args)
    split = getattr(args, "split", None) or "all"
    print(f"# {spec.name} split={split} ({len(chosen)})")
    for index, task in chosen:
        print(f"{index}\t{task['id']}")
    return 0 if chosen else 1


def cmd_run(args, spec: Spec) -> int:
    chosen = resolve_selection(spec, args)
    if not chosen:
        print("no tasks selected", file=sys.stderr)
        return 1
    turns = parse_turns(args.turns)
    run_dir = experiment_dir(spec.name, args.run)
    source = source_hermes(args.hermes_home)
    ensure_experiment(run_dir, source, {
        "benchmark": spec.name,
        "run": args.run,
        "turns": turns,
        "skill_mode": args.skill_mode,
        "hermes_source": str(source),
        "split": args.split,
        "start": args.start,
        "end": args.end,
    })
    chosen_model, chosen_provider = resolve_agent_model(
        run_dir / ".hermes" / "config.yaml", args.model, args.provider,
    )
    meta = load_json(run_dir / "run.json") or {}
    meta["model"] = chosen_model
    meta["provider"] = chosen_provider
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if skill_sources(args.skill_mode).hermes:
        ensure_hermes_skills(run_dir / ".hermes")
    print(f"model: {chosen_model or '(profile default)'}  provider: {chosen_provider or '(profile default)'}")
    print(f"workspace: {run_dir}")
    pending_ids = {
        task["id"] for _index, task in chosen
        if args.fresh or not task_is_complete(run_dir, task["id"])
    }
    for index, task in chosen:
        if task["id"] not in pending_ids:
            print(f"{index} {task['id']}: skipped (already complete)")
    failed = 0
    for index, task in chosen:
        if task["id"] not in pending_ids:
            continue
        record = run_task(
            spec, run_dir, task,
            skill_mode=args.skill_mode, turns=turns,
            model=chosen_model, provider=chosen_provider, timeout=args.timeout,
        )
        print(
            f"{index} {task['id']}: {record['status']} turns={record.get('turns_run', 0)} "
            f"model={record.get('model') or '-'} provider={record.get('provider') or '-'} "
            f"trajectory={record.get('trajectory') or '-'}"
        )
        if record["status"] != _COMPLETE:
            failed += 1
    summary, report = summary_paths(run_dir, args.summary, args.report)
    write_summary(run_dir, summary, report, turns, spec.name, args.run)
    return 1 if failed else 0


def cmd_summarize(args, spec: Spec) -> int:
    run_dir = experiment_dir(spec.name, args.run)
    if not run_dir.is_dir():
        print(f"missing experiment {run_dir}", file=sys.stderr)
        return 1
    turns = _turns_from(args, run_dir)
    summary, report = summary_paths(run_dir, args.summary, args.report)
    write_summary(run_dir, summary, report, turns, spec.name, args.run)
    return 0


def cmd_regrade(args, spec: Spec) -> int:
    run_dir = experiment_dir(spec.name, args.run)
    if not run_dir.is_dir():
        print(f"missing experiment {run_dir}", file=sys.stderr)
        return 1
    agent = run_dir / "agent"
    if agent.is_dir():
        for task_dir in sorted(path for path in agent.iterdir() if path.is_dir()):
            if args.task and task_dir.name not in {item.strip() for item in args.task.split(",")}:
                continue
            regrade_task(spec, run_dir, task_dir.name)
            print(f"{task_dir.name}: regraded")
    turns = _turns_from(args, run_dir)
    summary, report = summary_paths(run_dir, args.summary, args.report)
    write_summary(run_dir, summary, report, turns, spec.name, args.run)
    return 0


def _add_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split",
        help="Named task subset. PaperBench: debug, lite, dev, human, testing, all. "
        "ScienceAgentBench: a split stored in tasks.jsonl (verified, validation, train, test, when present). "
        "SkillsBench and Terminal-Bench have no named splits; omit this or pass all.",
    )
    parser.add_argument("--start", type=int, help="First task index in that split, 0-based, inclusive.")
    parser.add_argument("--end", type=int, help="Last task index in that split, 0-based, inclusive.")
    parser.add_argument("--task", help="Task id, or comma-separated ids, inside the index range.")
    parser.add_argument("--limit", type=int, help="Keep at most this many tasks after --start and --end.")


def _add_selection(parser: argparse.ArgumentParser, spec: Spec, *, running: bool) -> None:
    parser.add_argument("--run", required=True, help="Experiment name. One name is one shared Hermes profile and task tree.")
    _add_range(parser)
    if running:
        parser.add_argument(
            "--skill-mode",
            choices=SKILL_MODES,
            default="library",
            help=(
                "task: Hermes skills and this task's skills. "
                "library: this task's skills and SkillLib. "
                "library-compatible: Hermes skills, this task's skills, and SkillLib. "
                "library-only: SkillLib. "
                "none: none of those. Default: library."
            ),
        )
        parser.add_argument(
            "--turns",
            default="1,2,4,8",
            help="User-message budgets, comma-separated. The agent runs every turn up to the largest "
            "budget and stores a grade after each turn.",
        )
        parser.add_argument(
            "--model",
            help="Model id for the agent, such as Qwen/Qwen3.6-27B. "
            "With --provider and no model, that provider's default_model is used. "
            "Default: model.default in the replicated profile.",
        )
        parser.add_argument(
            "--provider",
            help="Provider name from providers: in the replicated profile, such as qwen-local or qwen-31. "
            "A model without this flag keeps model.provider from that profile.",
        )
        parser.add_argument("--timeout", type=float, help="Seconds for the whole task, across turns")
        parser.add_argument("--fresh", action="store_true", help="Run completed tasks again inside this experiment")
        parser.add_argument("--hermes-home", help="Profile to replicate. Default: $HERMES_HOME, else <repo>/.hermes")
    parser.add_argument("--summary", help="JSON summary path. Default: the experiment's summary.json")
    parser.add_argument("--report", help="Text summary path. Default: the experiment's summary.txt")
    if not running:
        parser.add_argument(
            "--turns",
            help="k values to report from the grades stored in each task record. "
            "success@k counts a pass on any conversation turn at or before k, including a k that was not "
            "a budget of the original run. Default: the budgets saved for this experiment.",
        )


def build_parser(spec: Spec, register=None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"benchmark/scripts/{spec.name.replace('-', '_')}.py")
    if spec.name == "terminal-bench":
        parser.prog = "benchmark/scripts/terminal_bench.py"
    sub = parser.add_subparsers(dest="command", required=True)
    clone = sub.add_parser("clone", help=f"Shallow-clone {spec.name} under benchmark/")
    clone.set_defaults(func=cmd_clone)
    listing = sub.add_parser("list", help="List task indexes and ids for a split")
    _add_range(listing)
    listing.set_defaults(func=cmd_list)
    run = sub.add_parser("run", help="Run Hermes in an isolated experiment. Re-running resumes interrupted tasks.")
    _add_selection(run, spec, running=True)
    run.set_defaults(func=cmd_run)
    summarize = sub.add_parser(
        "summarize",
        help="Recompute success@k, AUC, time, and token summaries from stored task grades",
    )
    _add_selection(summarize, spec, running=False)
    summarize.set_defaults(func=cmd_summarize)
    regrade = sub.add_parser("regrade", help="Grade the last turn again and rewrite the summary")
    _add_selection(regrade, spec, running=False)
    regrade.set_defaults(func=cmd_regrade)
    if register:
        register(sub)
    return parser


def cli(spec: Spec, argv=None, register=None) -> int:
    parser = build_parser(spec, register)
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args, spec)


def worker(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="benchmark/scripts/common.py _worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--usage", required=True)
    parser.add_argument("--hermes-home", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--provider", default="")
    parser.add_argument("--skill-mode", choices=SKILL_MODES, default="library")
    args = parser.parse_args(argv)
    os.environ["HERMES_HOME"] = args.hermes_home
    os.chdir(args.workspace)
    sys.path.insert(0, str(repo_root()))
    from hermes_cli.oneshot import run_oneshot

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    disabled = disabled_toolsets(args.skill_mode)
    return run_oneshot(
        prompt,
        model=args.model or None,
        provider=args.provider or None,
        resume=args.resume or None,
        cwd=args.workspace,
        disabled_toolsets=disabled,
        skip_memory=True,
        usage_file=args.usage,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        raise SystemExit(worker(sys.argv[2:]))
    print("Use benchmark/scripts/<benchmark>.py", file=sys.stderr)
    raise SystemExit(2)
