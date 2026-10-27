#!/usr/bin/env python3
"""ScienceAgentBench: clone, fetch the task sheet, unpack the zip, run, summarize.

Micro cases are ``program_saved``, ``program_ran``, and ``output_saved``. When an
eval module's ``eval()`` returns a mapping of criterion to bool, each criterion
is its own case. A single bool becomes the ``eval_passed`` case.

    python benchmark/scripts/scienceagentbench.py clone
    python benchmark/scripts/scienceagentbench.py fetch
    python benchmark/scripts/scienceagentbench.py unpack --zip benchmark/scienceagentbench/benchmark_verified.zip
    python benchmark/scripts/scienceagentbench.py run --run exp1 --limit 3 --turns 1,2,4
    python benchmark/scripts/scienceagentbench.py summarize --run exp1 --summary /tmp/sab.json --report /tmp/sab.txt
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

CHECKOUT = "scienceagentbench"
DATASET = "osunlp/ScienceAgentBench"
SPLIT_PREFERENCE = ("verified", "validation")
ZIP_PASSWORD = "scienceagentbench"


def checkout(root: Path) -> Path:
    return root / CHECKOUT


def tasks_path(root: Path) -> Path:
    return checkout(root) / "tasks.jsonl"


def list_tasks(root: Path, split: str | None = None) -> tuple[list[dict], str]:
    path = tasks_path(root)
    if not path.is_file():
        return [], (
            "ScienceAgentBench tasks are not in the git repo. "
            "Run: python benchmark/scripts/scienceagentbench.py fetch"
        )
    rows = []
    available: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        name = row.get("split")
        if name and name not in available:
            available.append(str(name))
    if split and split != "all":
        matched = [row for row in rows if row.get("split") == split]
        if not matched:
            listed = ", ".join(available) or "none"
            return [], f"ScienceAgentBench split {split!r} is not in {path}. Available: {listed}."
        rows = matched
    return [{"id": str(row["id"])} for row in rows], ""


def load_row(root: Path, task_id: str) -> dict | None:
    path = tasks_path(root)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row["id"]) == task_id:
            return row
    return None


def dataset_name(folder: str) -> str:
    lines = [line for line in folder.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[0]
    if len(first) > 4 and first[:4].isspace():
        return first[4:].strip().strip("/")
    return first.strip().strip("/")


def science_dataset_dir(root: Path) -> Path:
    base = checkout(root)
    official = base / "benchmark" / "datasets"
    legacy = base / "datasets"
    if official.is_dir() and any(official.iterdir()):
        return official
    if legacy.is_dir() and any(legacy.iterdir()):
        return legacy
    return official


def choose_science_split(available: list[str], requested: str | None = None) -> str:
    if requested:
        if requested not in available:
            raise ValueError(f"split {requested!r} is not one of {available}")
        return requested
    for name in SPLIT_PREFERENCE:
        if name in available:
            return name
    if not available:
        raise ValueError(f"{DATASET} published no splits")
    return available[0]


def _copy_matching(folder: Path, stem: str, dest: Path) -> None:
    if not folder.is_dir():
        return
    matches = [path for path in folder.iterdir() if path.is_file() and stem in path.name]
    if not matches:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in matches:
        shutil.copy2(path, dest / path.name)


def replicate(root: Path, task_id: str, dest: Path) -> None:
    row = load_row(root, task_id)
    if row is None:
        raise FileNotFoundError(f"ScienceAgentBench task {task_id}")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "task.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    name = dataset_name(row.get("dataset_folder_tree") or "")
    source = science_dataset_dir(root) / name if name else None
    if source and source.is_dir():
        common.copy_tree(source, dest / "datasets" / name, set())
    stem = Path(row.get("gold_program_name") or "").stem
    base = checkout(root) / "benchmark"
    if stem:
        _copy_matching(base / "eval_programs", stem, dest / "eval_programs")
        _copy_matching(base / "gold_programs", stem, dest / "gold_programs")
        _copy_matching(base / "scoring_rubrics", stem, dest / "scoring_rubrics")


def prepare(replica: Path, workspace: Path, skill_mode: str) -> tuple[str, str]:
    task_path = replica / "task.json"
    if not task_path.is_file():
        return "", f"missing {task_path}"
    row = json.loads(task_path.read_text(encoding="utf-8"))
    program = workspace / f"pred_{row['gold_program_name']}"
    name = dataset_name(row.get("dataset_folder_tree") or "")
    dataset_path = replica / "datasets" / name if name else replica / "datasets"
    parts = [
        "Write a complete Python program that accomplishes the task and save its outputs "
        "in the format the task asks for. Put the program in a single ```python fence and "
        f"also save that program to {program}.",
        "",
        row.get("task_inst") or "",
    ]
    if common.skill_sources(skill_mode).benchmark and row.get("domain_knowledge"):
        parts.extend(["", "Domain knowledge:", str(row["domain_knowledge"])])
    parts.extend([
        "",
        f"Dataset directory: {dataset_path}",
        "Directory structure:",
        row.get("dataset_folder_tree") or "",
        "",
        "Dataset preview:",
        row.get("dataset_preview") or "",
        "",
        f"Expected output file name: {row.get('output_fname') or ''}",
    ])
    if not dataset_path.is_dir():
        parts.append(f"\nDataset files are not unpacked at {dataset_path} yet. Use the directory tree and preview below.")
    prompt = "\n".join(parts).strip() + common.skill_note(skill_mode, workspace, has_task_skills=False)
    return prompt, ""


def _eval_cases(replica: Path, row: dict) -> list[dict]:
    stem = Path(row.get("gold_program_name") or "").stem
    folder = replica / "eval_programs"
    if not stem or not folder.is_dir():
        return []
    matches = [path for path in folder.iterdir() if path.is_file() and stem in path.name and path.suffix == ".py"]
    if len(matches) != 1:
        return []
    spec = importlib.util.spec_from_file_location("sab_eval_case", matches[0])
    if spec is None or spec.loader is None:
        return [{"id": "eval_passed", "passed": False, "detail": f"cannot load {matches[0]}", "graded": True}]
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        result = module.eval()
    except Exception as exc:
        return [{"id": "eval_passed", "passed": False, "detail": repr(exc)[:400], "graded": True}]
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], dict):
        if result[1] and all(isinstance(value, bool) for value in result[1].values()):
            return [
                {"id": str(key), "passed": bool(value), "detail": "", "graded": True}
                for key, value in result[1].items()
            ]
    success = result[0] if isinstance(result, tuple) and result else result
    detail = ""
    if isinstance(result, tuple) and len(result) > 1:
        detail = str(result[1])[:400]
    return [{"id": "eval_passed", "passed": bool(success), "detail": detail, "graded": True}]


def grade_row(replica: Path, workspace: Path, row: dict) -> list[dict]:
    program = workspace / f"pred_{row.get('gold_program_name') or 'program.py'}"
    detail = ""
    parses = False
    if program.is_file():
        try:
            ast.parse(program.read_text(encoding="utf-8"))
            parses = True
        except SyntaxError as exc:
            detail = str(exc)
    else:
        detail = "program missing"
    cases = [{"id": "program_saved", "passed": parses, "detail": detail, "graded": True}]
    ran = False
    run_detail = ""
    if parses:
        try:
            proc = subprocess.run(
                [sys.executable, str(program)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            ran = proc.returncode == 0
            run_detail = (proc.stderr or "")[-400:]
        except subprocess.TimeoutExpired:
            run_detail = "timed out"
    cases.append({"id": "program_ran", "passed": ran, "detail": run_detail, "graded": True})
    output_name = row.get("output_fname") or ""
    if output_name:
        output = Path(output_name)
        if not output.is_absolute():
            output = workspace / output
        saved = output.is_file() and output.stat().st_size > 0
        cases.append({"id": "output_saved", "passed": saved, "detail": str(output), "graded": True})
    cases.extend(_eval_cases(replica, row))
    return cases


def grade(replica: Path, workspace: Path, task_dir: Path) -> list[dict]:
    task_path = replica / "task.json"
    if not task_path.is_file():
        return [{"id": "task", "passed": False, "detail": f"missing {task_path}", "graded": True}]
    row = json.loads(task_path.read_text(encoding="utf-8"))
    return grade_row(replica, workspace, row)


def merge_split_rows(existing: str, split: str, rows: list[dict]) -> str:
    """Replace rows of ``split`` and keep every other split already in the file."""
    kept = []
    for line in existing.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") != split:
            kept.append(row)
    kept.extend(rows)
    if not kept:
        return ""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in kept) + "\n"


def fetch_science(root: Path, split: str | None = None) -> str:
    dest = tasks_path(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise SystemExit(
            f"ScienceAgentBench tasks come from Hugging Face dataset {DATASET}. "
            "Install the datasets package and rerun fetch."
        ) from exc
    available = list(get_dataset_split_names(DATASET))
    try:
        chosen = choose_science_split(available, split)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    dataset = load_dataset(DATASET, split=chosen)
    rows = []
    for index, row in enumerate(dataset):
        rows.append({
            "id": f"{chosen}-{index}",
            "split": chosen,
            "task_inst": row.get("task_inst") or "",
            "dataset_folder_tree": row.get("dataset_folder_tree") or "",
            "dataset_preview": row.get("dataset_preview") or "",
            "output_fname": row.get("output_fname") or "",
            "gold_program_name": row.get("gold_program_name") or f"task_{index}.py",
            "domain_knowledge": row.get("domain_knowledge") or "",
        })
    existing = dest.read_text(encoding="utf-8") if dest.is_file() else ""
    dest.write_text(merge_split_rows(existing, chosen, rows), encoding="utf-8")
    return f"wrote {len(rows)} {chosen} tasks to {dest}"


def _zip_members(zip_path: Path) -> list[str]:
    proc = subprocess.run(["unzip", "-Z1", str(zip_path)], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        proc = subprocess.run(
            ["unzip", "-Z1", "-P", ZIP_PASSWORD, str(zip_path)],
            capture_output=True, text=True, check=False,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        raise SystemExit(detail or f"cannot list {zip_path}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip() and not line.endswith("/")]


def unpack_science(root: Path, zip_path: Path) -> str:
    base = checkout(root)
    if not base.is_dir():
        raise SystemExit(f"missing checkout {base}")
    if not zip_path.is_file():
        raise SystemExit(f"missing zip {zip_path}")
    members = _zip_members(zip_path)
    if not members:
        raise SystemExit(f"{zip_path} is empty")
    into_repo = any(name.startswith("benchmark/") for name in members)
    dest = base if into_repo else base / "benchmark"
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["unzip", "-o", "-P", ZIP_PASSWORD, str(zip_path), "-d", str(dest)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode > 1:
        detail = (proc.stderr or proc.stdout).strip()
        raise SystemExit(detail or f"unzip failed with status {proc.returncode}")
    return f"unpacked {zip_path.name} into {dest}"


def cmd_fetch(args, _spec) -> int:
    print(fetch_science(common.benchmark_root(), split=args.split))
    return 0


def cmd_unpack(args, _spec) -> int:
    zip_path = Path(args.zip_path) if args.zip_path else checkout(common.benchmark_root()) / "benchmark_verified.zip"
    print(unpack_science(common.benchmark_root(), zip_path))
    return 0


def register(sub) -> None:
    fetch = sub.add_parser("fetch", help="Download the Hugging Face task sheet to tasks.jsonl")
    fetch.add_argument("--split", help="Dataset split to write, such as verified, validation, train, or test. Other splits already in tasks.jsonl are kept.")
    fetch.set_defaults(func=cmd_fetch)
    unpack = sub.add_parser("unpack", help="Unzip the benchmark bundle into the ScienceAgentBench checkout")
    unpack.add_argument("--zip", dest="zip_path", help="Path to benchmark_verified.zip")
    unpack.set_defaults(func=cmd_unpack)


SPEC = common.Spec(
    name="scienceagentbench",
    checkout=CHECKOUT,
    repo_url="https://github.com/OSU-NLP-Group/ScienceAgentBench.git",
    list_tasks=list_tasks,
    replicate=replicate,
    prepare=prepare,
    grade=grade,
)


if __name__ == "__main__":
    raise SystemExit(common.cli(SPEC, register=register))
