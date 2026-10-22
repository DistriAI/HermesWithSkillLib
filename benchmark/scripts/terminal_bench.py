#!/usr/bin/env python3
"""Terminal-Bench: clone, list, run, summarize.

    python benchmark/scripts/terminal_bench.py clone
    python benchmark/scripts/terminal_bench.py list
    python benchmark/scripts/terminal_bench.py run --run exp1 --task data-anonymization --turns 1,2,4,8
    python benchmark/scripts/terminal_bench.py summarize --run exp1 --summary /tmp/tb.json --report /tmp/tb.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

CHECKOUT = "terminal-bench"


def list_tasks(root: Path, split: str | None = None) -> tuple[list[dict], str]:
    unnamed = common.unnamed_split_warning("Terminal-Bench", split)
    if unnamed:
        return [], unnamed
    tasks = common.task_dirs(root / CHECKOUT / "tasks")
    found = [{"id": path.name} for path in tasks if (path / "instruction.md").is_file()]
    warning = "" if found or (root / CHECKOUT / "tasks").is_dir() else f"missing {root / CHECKOUT / 'tasks'}"
    return found, warning


def replicate(root: Path, task_id: str, dest: Path) -> None:
    common.copy_tree(root / CHECKOUT / "tasks" / task_id, dest, {"solution"})


def prepare(replica: Path, workspace: Path, skill_mode: str) -> tuple[str, str]:
    instruction = replica / "instruction.md"
    if not instruction.is_file():
        return "", f"missing {instruction}"
    common.copy_inputs(replica / "environment", workspace, include_skills=False)
    prompt = common.relocate(instruction.read_text(encoding="utf-8").strip(), "/app", workspace)
    prompt = prompt.rstrip() + common.layout_note(workspace)
    return prompt.rstrip() + common.skill_note(skill_mode, workspace, has_task_skills=False), ""


def grade(replica: Path, workspace: Path, task_dir: Path) -> list[dict]:
    tests = replica / "tests"
    return common.grade_with_pytest(
        tests / "test_outputs.py",
        workspace,
        [("/app", workspace), ("/tests", tests)],
    )


SPEC = common.Spec(
    name="terminal-bench",
    checkout=CHECKOUT,
    repo_url="https://github.com/harbor-framework/terminal-bench.git",
    list_tasks=list_tasks,
    replicate=replicate,
    prepare=prepare,
    grade=grade,
)


if __name__ == "__main__":
    raise SystemExit(common.cli(SPEC))
