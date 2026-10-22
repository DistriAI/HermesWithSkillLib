#!/usr/bin/env python3
"""PaperBench: clone, pull paper files, list, run, regrade, summarize.

Rubric leaves are the test cases. A leaf counts as passed only when the task
directory contains ``judge.json`` of the form ``{"leaves": {"<leaf-id>": true}}``.
``regrade`` reads that file after the official judge and refreshes the last turn.

    python benchmark/scripts/paperbench.py clone
    python benchmark/scripts/paperbench.py pull
    python benchmark/scripts/paperbench.py list --split lite
    python benchmark/scripts/paperbench.py run --run exp1 --split lite --turns 1,4,8
    python benchmark/scripts/paperbench.py regrade --run exp1 --summary /tmp/pb.json --report /tmp/pb.txt
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

CHECKOUT = "frontier-evals"


def _papers(root: Path) -> Path:
    return root / CHECKOUT / "project" / "paperbench" / "data" / "papers"


def list_tasks(root: Path, split: str | None = None) -> tuple[list[dict], str]:
    papers = _papers(root)
    if not papers.is_dir():
        return [], f"PaperBench papers directory is missing: {papers}"
    present = {path.name for path in common.task_dirs(papers)}
    if not split:
        return [{"id": name} for name in sorted(present)], ""
    path = root / CHECKOUT / "project" / "paperbench" / "experiments" / "splits" / f"{split}.txt"
    if not path.is_file():
        return [], (
            f"PaperBench split not found: {path}. "
            "Available names: debug, lite, dev, human, testing, all."
        )
    ordered = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ordered:
        return [], f"PaperBench split {split} is empty: {path}"
    missing = [name for name in ordered if name not in present]
    warning = f"PaperBench split {split} is missing directories: {', '.join(missing)}" if missing else ""
    return [{"id": name} for name in ordered if name in present], warning


def replicate(root: Path, task_id: str, dest: Path) -> None:
    common.copy_tree(_papers(root) / task_id, dest, set())


def _title(paper_dir: Path) -> str:
    config = paper_dir / "config.yaml"
    if not config.is_file():
        return paper_dir.name
    for line in config.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("title:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return paper_dir.name


def prepare(replica: Path, workspace: Path, skill_mode: str) -> tuple[str, str]:
    paper_md = replica / "paper.md"
    if not paper_md.is_file():
        return "", f"missing {paper_md}"
    if common.is_lfs_pointer(paper_md):
        return "", (
            "paper.md is still a Git LFS pointer. Run: "
            "python benchmark/scripts/paperbench.py pull"
        )
    dest = workspace / "paper"
    dest.mkdir(parents=True, exist_ok=True)
    for path in replica.iterdir():
        if path.name == "rubric.json" or common.is_lfs_pointer(path):
            continue
        target = dest / path.name
        if path.is_dir():
            common.copy_tree(path, target, set())
        elif path.is_file():
            shutil.copy2(path, target)
    (workspace / "submission").mkdir(exist_ok=True)
    prompt = (
        f'Reproduce the research paper "{_title(replica)}".\n\n'
        f"Read {dest / 'paper.md'} fully, then read {dest / 'addendum.md'}. "
        "Appendix-only experiments are out of scope. Experiments described in the main text "
        "stay in scope even when some details are in the appendix.\n\n"
        f"Do not look up or reuse the paper's own codebase or any URL in {dest / 'blacklist.txt'}.\n\n"
        f"Write the replication as a git repository at {workspace / 'submission'}. "
        "Include README.md describing what you reproduced and how the code maps to the paper, "
        "and reproduce.sh when a single command can rerun the experiments."
    )
    return prompt.rstrip() + common.skill_note(skill_mode, workspace, has_task_skills=False), ""


def grade(replica: Path, workspace: Path, task_dir: Path) -> list[dict]:
    rubric_path = replica / "rubric.json"
    if not rubric_path.is_file():
        return [{"id": "rubric", "passed": False, "detail": f"missing {rubric_path}", "graded": False}]
    cases = common.rubric_leaves(json.loads(rubric_path.read_text(encoding="utf-8")))
    judge = common.load_json(task_dir / "judge.json")
    if judge:
        cases = common.apply_judge(cases, judge)
    return cases


def pull_lfs(dest: Path) -> None:
    subprocess.run(
        ["git", "-C", str(dest), "-c", "lfs.fetchexclude=", "lfs", "pull", "-I", "project/paperbench/data/papers/**"],
        check=False,
    )


def cmd_pull(_args, _spec) -> int:
    dest = common.benchmark_root() / CHECKOUT
    if not dest.is_dir():
        print(f"missing {dest}; run: git submodule update --init -- benchmark/frontier-evals", file=sys.stderr)
        return 1
    pull_lfs(dest)
    print(f"pulled paper files into {dest}")
    return 0


def register(sub) -> None:
    pull = sub.add_parser("pull", help="Download PaperBench paper files through Git LFS")
    pull.set_defaults(func=cmd_pull)


SPEC = common.Spec(
    name="paperbench",
    checkout=CHECKOUT,
    repo_url="https://github.com/openai/frontier-evals.git",
    list_tasks=list_tasks,
    replicate=replicate,
    prepare=prepare,
    grade=grade,
    uses_split=True,
    after_clone=pull_lfs,
)


if __name__ == "__main__":
    raise SystemExit(common.cli(SPEC, register=register))
