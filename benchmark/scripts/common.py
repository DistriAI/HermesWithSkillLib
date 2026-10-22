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

``--turns`` lists user-message budgets. Turn 1 is the task prompt. Each later
turn resumes the same session and asks the agent to continue. The runner grades
after every turn.

* macro success@k — fraction of tasks that passed every test case on some stored turn at or before k
* micro success@k — fraction of test cases that passed on some stored turn at or before k
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
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

CONTINUE_PROMPT = (
    "Continue the same task from the current workspace. "
    "Finish anything the task still requires. Keep files you already created."
)
_SKIP_FILES = {"Dockerfile", "docker-compose.yml", "docker-compose.yaml", ".dockerignore"}
_LFS_MARK = "version https://git-lfs.github.com/spec/v1"
_SKIP_HERMES_DIRS = {"sessions", "logs", "memories", "cache", "checkpoints", "state-snapshots", "skins"}
_CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml", ".md", ".txt"}
_COMPLETE = "complete"


class Spec:
    def __init__(self, name, checkout, repo_url, list_tasks, replicate, prepare, grade, *, uses_split=False, after_clone=None):
        self.name = name
        self.checkout = checkout
        self.repo_url = repo_url
        self.list_tasks = list_tasks
        self.replicate = replicate
        self.prepare = prepare
        self.grade = grade
        self.uses_split = uses_split
        self.after_clone = after_clone


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


def skill_note(mode: str, workspace: Path, *, has_task_skills: bool) -> str:
    if mode == "library":
        return (
            "\n\nBefore writing the solution, call skill_library_find for this task "
            "and skill_library_load on the best matches (at most three). "
            "Adapt what you load; a skill is a procedure, not the answer."
        )
    if mode == "task" and has_task_skills:
        return (
            f"\n\nThis task ships skills under {workspace / 'skills'}. "
            "Read each SKILL.md and follow those procedures. "
            "Do not call skill_library_find, skill_library_load, or skill_library_visit."
        )
    return (
        "\n\nSolve this from the task statement and the files in the workspace. "
        "Do not call skill_library_find, skill_library_load, or skill_library_visit."
    )


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
        "definitions": {
            "user_turn": "One user message sent by this runner. Turn 1 is the task. Later turns resume that session.",
            "macro_success_at_k": "Fraction of completed tasks that passed every test case on any stored turn at or before k.",
            "micro_success_at_k": "Fraction of test cases that passed on any stored turn at or before k.",
            "auc_at_k": "Mean of success@1 through success@k. Each success@t counts passes at or before t.",
            "final_success": "Pass rate on each completed task's last turn.",
            "successful_tasks": "Completed tasks whose last turn passed every test case.",
            "all_tasks": "Completed tasks. Interrupted tasks are omitted until a rerun finishes them.",
        },
    }


def format_report(summary: dict, incomplete: list[str]) -> str:
    lines = [
        f"benchmark: {summary['benchmark']}",
        f"run: {summary['run']}",
        f"model: {summary.get('model') or ''}",
        f"provider: {summary.get('provider') or ''}",
        "turns: " + ",".join(str(k) for k in summary["turns"]),
        "success@k counts a pass on any stored turn at or before k",
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
    if incomplete:
        lines.extend(["", "incomplete: " + ",".join(incomplete)])
    lines.append("")
    return "\n".join(lines)


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


def grade_with_pytest(test_file: Path, workspace: Path, mounts: list[tuple[str, Path]]) -> list[dict]:
    """Run a copied verifier with mount paths rewritten to this run's workspace."""
    if not test_file.is_file():
        return [{"id": "verifier", "passed": False, "detail": f"missing {test_file}", "graded": True}]
    try:
        import pytest  # noqa: F401
    except ImportError:
        return [{"id": "pytest", "passed": False, "detail": "pytest is not installed", "graded": True}]
    tmp = Path(tempfile.mkdtemp(prefix="bench-grade-"))
    try:
        text = test_file.read_text(encoding="utf-8")
        for mount, dest in mounts:
            text = relocate(text, mount, dest)
        copied = tmp / test_file.name
        copied.write_text(text, encoding="utf-8")
        report = tmp / "junit.xml"
        try:
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", str(copied), "-q",
                    "--confcutdir", str(tmp), "--rootdir", str(tmp),
                    f"--junitxml={report}", "-p", "no:cacheprovider",
                ],
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
        return [{"id": "pytest", "passed": False, "detail": detail, "graded": True}]
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


def invoke_worker(*, workspace, prompt, usage, hermes_home, resume, model, provider, disable_skill_library, timeout):
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
    if disable_skill_library:
        cmd.append("--disable-skill-library")
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    try:
        proc = subprocess.run(
            cmd, cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout, env=env, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return None, exc.stdout or "", exc.stderr or ""
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


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
    }
    if error:
        record["status"] = "error"
        record["error"] = error
        write_result(agent_dir / "result.json", record)
        return record
    home = run_dir / ".hermes"
    record["hermes_home"] = str(home)
    write_result(agent_dir / "result.json", record)
    started = time.perf_counter()
    deadline = None if timeout is None else started + timeout
    session_id = None
    max_k = max(turns)
    disable = skill_mode != "library"
    for turn in range(1, max_k + 1):
        remaining = None if deadline is None else deadline - time.perf_counter()
        if remaining is not None and remaining <= 0:
            record["status"] = "timeout"
            record["error"] = f"timed out after {timeout}s"
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        turn_dir = agent_dir / "turns" / str(turn)
        turn_dir.mkdir(parents=True, exist_ok=True)
        usage_path = turn_dir / "usage.json"
        message = prompt if turn == 1 else CONTINUE_PROMPT
        code, stdout, stderr = invoke_worker(
            workspace=workspace,
            prompt=message,
            usage=usage_path,
            hermes_home=home,
            resume=session_id,
            model=model,
            provider=provider,
            disable_skill_library=disable,
            timeout=remaining,
        )
        _append(agent_dir / "agent.stdout", f"\n--- turn {turn} ---\n{stdout}")
        _append(agent_dir / "agent.stderr", f"\n--- turn {turn} ---\n{stderr}")
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
            record["turns_run"] = turn
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        try:
            cases = spec.grade(replica, workspace, agent_dir)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            record["turns_run"] = turn
            record["duration_sec"] = time.perf_counter() - started
            write_result(agent_dir / "result.json", record)
            return record
        record["grades"].append({"turn": turn, "cases": cases})
        record["turns_run"] = turn
        record["duration_sec"] = time.perf_counter() - started
        write_result(agent_dir / "result.json", record)
        if cases and all(case.get("passed") for case in cases):
            break
    record["status"] = _COMPLETE
    record["duration_sec"] = time.perf_counter() - started
    write_result(agent_dir / "result.json", record)
    return record


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
            f"model={record.get('model') or '-'} provider={record.get('provider') or '-'}"
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
        parser.add_argument("--skill-mode", choices=("library", "task", "none"), default="library")
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
            "success@k counts a pass on any turn at or before k, including a k that was not "
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
    parser.add_argument("--disable-skill-library", action="store_true")
    args = parser.parse_args(argv)
    os.environ["HERMES_HOME"] = args.hermes_home
    os.chdir(args.workspace)
    sys.path.insert(0, str(repo_root()))
    from hermes_cli.oneshot import run_oneshot

    prompt = Path(args.prompt).read_text(encoding="utf-8")
    disabled = ["skill_library"] if args.disable_skill_library else None
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
