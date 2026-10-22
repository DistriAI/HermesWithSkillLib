#!/usr/bin/env python3
"""SkillsBench: clone, list, run, summarize.

    python benchmark/scripts/skillsbench.py clone
    python benchmark/scripts/skillsbench.py list
    python benchmark/scripts/skillsbench.py run --run exp1 --task citation-check --turns 1,2,4,8
    python benchmark/scripts/skillsbench.py summarize --run exp1 --summary /tmp/skills.json --report /tmp/skills.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

CHECKOUT = "skillsbench"


def list_tasks(root: Path, split: str | None = None) -> tuple[list[dict], str]:
    unnamed = common.unnamed_split_warning("SkillsBench", split)
    if unnamed:
        return [], unnamed
    tasks = common.task_dirs(root / CHECKOUT / "tasks")
    found = [{"id": path.name} for path in tasks if (path / "task.md").is_file()]
    warning = "" if found or (root / CHECKOUT / "tasks").is_dir() else f"missing {root / CHECKOUT / 'tasks'}"
    return found, warning


def replicate(root: Path, task_id: str, dest: Path) -> None:
    common.copy_tree(root / CHECKOUT / "tasks" / task_id, dest, {"oracle"})


def prepare(replica: Path, workspace: Path, skill_mode: str) -> tuple[str, str]:
    task_md = replica / "task.md"
    if not task_md.is_file():
        return "", f"missing {task_md}"
    include = skill_mode == "task"
    common.copy_inputs(replica / "environment", workspace, include_skills=include)
    prompt = common.relocate(common.instruction_body(task_md.read_text(encoding="utf-8")), "/root", workspace)
    prompt = prompt.rstrip() + common.layout_note(workspace)
    has_skills = include and (replica / "environment" / "skills").is_dir()
    return prompt.rstrip() + common.skill_note(skill_mode, workspace, has_task_skills=has_skills), ""


def grade(replica: Path, workspace: Path, task_dir: Path) -> list[dict]:
    return common.grade_with_pytest(
        replica / "verifier" / "test_outputs.py",
        workspace,
        [("/root", workspace)],
    )


SPEC = common.Spec(
    name="skillsbench",
    checkout=CHECKOUT,
    repo_url="https://github.com/benchflow-ai/skillsbench.git",
    list_tasks=list_tasks,
    replicate=replicate,
    prepare=prepare,
    grade=grade,
)


if __name__ == "__main__":
    raise SystemExit(common.cli(SPEC))
