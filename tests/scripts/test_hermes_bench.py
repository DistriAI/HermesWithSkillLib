"""Benchmark runners: isolation, grading, and turn-budget metrics. No model calls."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "benchmark" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common
import jail
import paperbench
import scienceagentbench
import skillsbench
import terminal_bench


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_relocate_rewrites_only_path_boundaries():
    text = "read /root/data and /root, not /rooted or /appetizer, then /app/input"
    once = common.relocate(text, "/root", Path("/work"))
    twice = common.relocate(once, "/app", Path("/work"))
    assert twice == "read /work/data and /work, not /rooted or /appetizer, then /work/input"


def test_replicate_hermes_keeps_sessions_behind(tmp_path: Path):
    source = tmp_path / "src"
    _write(source / "config.yaml", "model: m\n")
    _write(source / ".env", "API=1\n")
    _write(source / "state.db", "db")
    _write(source / "sessions" / "a.json", "{}")
    dest = tmp_path / "dest"
    common.replicate_hermes_config(source, dest)
    assert (dest / "config.yaml").read_text(encoding="utf-8") == "model: m\n"
    assert (dest / ".env").read_text(encoding="utf-8") == "API=1\n"
    assert not (dest / "state.db").exists()
    assert not (dest / "sessions").exists()


def test_skillsbench_hides_oracle_and_skills(tmp_path: Path):
    root = tmp_path / "benchmark"
    task = root / "skillsbench" / "tasks" / "demo"
    _write(task / "task.md", "---\nname: demo\n---\nUse /root/input.txt\n")
    _write(task / "environment" / "input.txt", "data")
    _write(task / "environment" / "skills" / "helper" / "SKILL.md", "secret")
    _write(task / "oracle" / "solve.sh", "echo leak")
    _write(task / "verifier" / "test_outputs.py", "def test_ok():\n    assert True\n")
    dest = tmp_path / "replica"
    skillsbench.replicate(root, "demo", dest)
    assert not (dest / "oracle").exists()
    assert (dest / "verifier" / "test_outputs.py").is_file()
    workspace = tmp_path / "workspace"
    prompt, error = skillsbench.prepare(dest, workspace, "library")
    assert error == ""
    assert (workspace / "input.txt").is_file()
    assert (workspace / "skills" / "helper" / "SKILL.md").read_text(encoding="utf-8") == "secret"
    assert "Read each SKILL.md" in prompt
    assert "skill_library_find" in prompt
    assert "Do not call" not in prompt
    assert "/root/input.txt" not in prompt
    only_ws = tmp_path / "only-ws"
    only_prompt, _ = skillsbench.prepare(dest, only_ws, "library-only")
    assert not (only_ws / "skills").exists()
    assert "Read each SKILL.md" not in only_prompt
    assert "skill_library_find" in only_prompt
    task_prompt, _ = skillsbench.prepare(dest, tmp_path / "task-ws", "task")
    assert (tmp_path / "task-ws" / "skills" / "helper" / "SKILL.md").is_file()
    assert "Do not call skill_library_find" in task_prompt


def test_terminal_bench_skips_solution(tmp_path: Path):
    root = tmp_path / "benchmark"
    task = root / "terminal-bench" / "tasks" / "demo"
    _write(task / "instruction.md", "Edit /app/input.txt\n")
    _write(task / "environment" / "input.txt", "x")
    _write(task / "solution" / "solve.sh", "echo leak")
    dest = tmp_path / "replica"
    terminal_bench.replicate(root, "demo", dest)
    assert not (dest / "solution").exists()
    workspace = tmp_path / "workspace"
    prompt, error = terminal_bench.prepare(dest, workspace, "none")
    assert error == ""
    assert str(workspace / "input.txt") in prompt
    assert "/app/input.txt" not in prompt


def test_science_domain_knowledge_and_cases(tmp_path: Path):
    root = tmp_path / "benchmark"
    row = {
        "id": "0",
        "task_inst": "plot the curve",
        "dataset_folder_tree": "    example_data/\n",
        "dataset_preview": "a,b",
        "output_fname": "out.csv",
        "gold_program_name": "plot.py",
        "domain_knowledge": "use scipy",
    }
    _write(root / "scienceagentbench" / "tasks.jsonl", json.dumps(row) + "\n")
    official = root / "scienceagentbench" / "benchmark" / "datasets" / "example_data"
    _write(official / "train.csv", "a,b\n")
    legacy = root / "scienceagentbench" / "datasets"
    legacy.mkdir(parents=True)
    assert scienceagentbench.science_dataset_dir(root) == official.parent
    dest = tmp_path / "replica"
    scienceagentbench.replicate(root, "0", dest)
    assert (dest / "datasets" / "example_data" / "train.csv").is_file()
    library, _ = scienceagentbench.prepare(dest, tmp_path / "lib", "library")
    task, _ = scienceagentbench.prepare(dest, tmp_path / "task", "task")
    only, _ = scienceagentbench.prepare(dest, tmp_path / "only", "library-only")
    assert "use scipy" in library
    assert "use scipy" in task
    assert "use scipy" not in only
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write(workspace / "pred_plot.py", "from pathlib import Path\nPath('out.csv').write_text('a\\n')\n")
    flags = {case["id"]: case["passed"] for case in scienceagentbench.grade_row(dest, workspace, row)}
    assert flags == {"program_saved": True, "program_ran": True, "output_saved": True}


def test_choose_science_split_and_unpack(tmp_path: Path, monkeypatch):
    assert scienceagentbench.choose_science_split(["validation", "verified"]) == "verified"
    root = tmp_path / "benchmark"
    base = root / "scienceagentbench"
    base.mkdir(parents=True)
    archive = base / "benchmark_verified.zip"
    archive.write_bytes(b"not-a-real-zip")
    monkeypatch.setattr(scienceagentbench, "_zip_members", lambda path: ["benchmark/datasets/example/a.csv"])

    def fake_run(cmd, **kwargs):
        dest = Path(cmd[cmd.index("-d") + 1])
        assert dest == base
        return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(scienceagentbench.subprocess, "run", fake_run)
    assert "unpacked" in scienceagentbench.unpack_science(root, archive)


def test_pytest_grades_rewritten_workspace(tmp_path: Path):
    test_file = tmp_path / "test_outputs.py"
    test_file.write_text(
        "from pathlib import Path\n\ndef test_ok():\n    assert Path('/root/answer.txt').read_text() == 'yes'\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    _write(workspace / "answer.txt", "yes")
    cases = common.grade_with_pytest(test_file, workspace, [("/root", workspace)])
    assert any(case["passed"] and "test_ok" in case["id"] for case in cases)


def test_conversation_stops_on_the_benchmark_verifier(tmp_path: Path):
    passed = [{"id": "test_ok", "passed": True, "graded": True}]
    failed = [{"id": "test_ok", "passed": False, "detail": "missing mass_report.json", "graded": True}]
    unjudged = [{"id": "leaf-1", "passed": False, "graded": False}]
    assert common.verifier_finished(tmp_path, tmp_path, passed)
    assert not common.verifier_finished(tmp_path, tmp_path, failed)
    assert not common.verifier_finished(tmp_path, tmp_path, unjudged)
    followup = common.followup_prompt(failed)
    assert "test_ok" not in followup
    assert "mass_report.json" not in followup
    assert "not accepted" in followup
    assert "task statement is the only spec" in followup
    assert "stop without changing the design" in followup
    assert "change only that part" in followup
    workspace = tmp_path / "workspace"
    assert not paperbench.finished(tmp_path, workspace, unjudged)
    readme = workspace / "submission" / "README.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("reproduced the main experiment\n", encoding="utf-8")
    assert paperbench.finished(tmp_path, workspace, unjudged)


def test_paper_leaves_follow_judge_file():
    rubric = {
        "id": "root",
        "requirements": "all",
        "sub_tasks": [
            {"id": "leaf-a", "requirements": "A", "sub_tasks": []},
            {"id": "leaf-b", "requirements": "B", "sub_tasks": []},
        ],
    }
    cases = common.rubric_leaves(rubric)
    assert [case["id"] for case in cases] == ["leaf-a", "leaf-b"]
    assert all(case["graded"] is False for case in cases)
    judged = common.apply_judge(cases, {"leaves": {"leaf-a": True}})
    flags = {case["id"]: case for case in judged}
    assert flags["leaf-a"]["passed"] is True and flags["leaf-a"]["graded"] is True
    assert flags["leaf-b"]["graded"] is False
    replica = paperbench  # module import keeps the grader available
    assert replica.grade.__name__ == "grade"


def test_index_range_follows_split_order(tmp_path: Path):
    tasks = [{"id": str(index)} for index in range(5)]
    window = common.window_tasks(tasks, start=1, end=3)
    assert [(index, item["id"]) for index, item in window] == [(1, "1"), (2, "2"), (3, "3")]
    assert common.window_tasks(tasks, start=1, end=3, task="2,9")[0][1]["id"] == "2"
    try:
        common.window_tasks(tasks, start=4, end=1)
    except ValueError as exc:
        assert "invalid" in str(exc)
    else:
        raise AssertionError("expected an invalid range")

    root = tmp_path / "benchmark"
    papers = root / "frontier-evals" / "project" / "paperbench" / "data" / "papers"
    for name in ("a-paper", "b-paper"):
        (papers / name).mkdir(parents=True)
    _write(
        root / "frontier-evals" / "project" / "paperbench" / "experiments" / "splits" / "test.txt",
        "b-paper\na-paper\n",
    )
    listed, warning = paperbench.list_tasks(root, "test")
    assert warning == ""
    assert [item["id"] for item in listed] == ["b-paper", "a-paper"]

    rows = [
        {"id": "train-0", "split": "train", "task_inst": "a", "gold_program_name": "a.py", "output_fname": "a.csv", "dataset_folder_tree": "", "dataset_preview": "", "domain_knowledge": ""},
        {"id": "train-1", "split": "train", "task_inst": "b", "gold_program_name": "b.py", "output_fname": "b.csv", "dataset_folder_tree": "", "dataset_preview": "", "domain_knowledge": ""},
        {"id": "test-0", "split": "test", "task_inst": "c", "gold_program_name": "c.py", "output_fname": "c.csv", "dataset_folder_tree": "", "dataset_preview": "", "domain_knowledge": ""},
    ]
    _write(root / "scienceagentbench" / "tasks.jsonl", "\n".join(json.dumps(row) for row in rows) + "\n")
    test_rows, warning = scienceagentbench.list_tasks(root, "test")
    assert warning == ""
    assert [item["id"] for item in test_rows] == ["test-0"]
    missing, warning = scienceagentbench.list_tasks(root, "validation")
    assert missing == []
    assert "validation" in warning
    merged = scienceagentbench.merge_split_rows(
        "\n".join(json.dumps(row) for row in rows),
        "test",
        [{"id": "test-0", "split": "test"}],
    )
    kept = [json.loads(line)["id"] for line in merged.splitlines()]
    assert kept == ["train-0", "train-1", "test-0"]
    assert "named split" in skillsbench.list_tasks(root, "train")[1]


def test_resolve_agent_model_uses_profile_providers(tmp_path: Path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join([
            "model:",
            "  default: Qwen/Qwen3.8-27B",
            "  provider: qwen-local",
            "providers:",
            "  qwen-31:",
            "    default_model: Qwen/Qwen3.6-27B",
            "  qwen-local:",
            "    default_model: Qwen/Qwen3.8-27B",
            "",
        ]),
        encoding="utf-8",
    )
    assert common.resolve_agent_model(config, None, None) == (None, None)
    assert common.resolve_agent_model(config, "Qwen/Qwen3.5-9B", None) == ("Qwen/Qwen3.5-9B", "qwen-local")
    assert common.resolve_agent_model(config, None, "qwen-31") == ("Qwen/Qwen3.6-27B", "qwen-31")
    assert common.resolve_agent_model(config, "Qwen/Qwen3.6-27B", "qwen-31") == ("Qwen/Qwen3.6-27B", "qwen-31")


def test_success_curves_and_cost_stats():
    records = [
        {
            "status": "complete", "id": "A", "duration_sec": 10,
            "input_tokens": 100, "output_tokens": 40, "total_tokens": 140,
            "grades": [
                {"turn": 1, "cases": [{"id": "a1", "passed": False}, {"id": "a2", "passed": False}]},
                {"turn": 2, "cases": [{"id": "a1", "passed": True}, {"id": "a2", "passed": True}]},
            ],
        },
        {
            "status": "complete", "id": "B", "duration_sec": 30,
            "input_tokens": 10, "output_tokens": 10, "total_tokens": 20,
            "grades": [
                {"turn": 1, "cases": [{"id": "b1", "passed": True}]},
                {"turn": 2, "cases": [{"id": "b1", "passed": False}]},
            ],
        },
        {"status": "running", "id": "C", "grades": [], "duration_sec": 99, "input_tokens": 9, "output_tokens": 9, "total_tokens": 18},
    ]
    summary = common.summarize_records(records, [1, 2], benchmark="skillsbench", run="r")
    assert summary["n_complete"] == 2
    assert summary["macro_success_at_k"] == {"1": 0.5, "2": 1.0}
    assert abs(summary["micro_success_at_k"]["1"] - (1 / 3)) < 1e-9
    assert summary["micro_success_at_k"]["2"] == 1.0
    assert abs(summary["macro_auc_at_k"]["2"] - 0.75) < 1e-9
    assert abs(summary["micro_auc_at_k"]["2"] - ((1 / 3) + 1) / 2) < 1e-9
    assert summary["final_macro_success"] == 0.5
    assert abs(summary["final_micro_success"] - (2 / 3)) < 1e-9
    assert summary["execution_time_sec"]["successful_tasks"]["mean"] == 10
    assert summary["execution_time_sec"]["successful_tasks"]["n"] == 1
    assert summary["execution_time_sec"]["all_tasks"]["mean"] == 20
    assert summary["input_tokens"]["successful_tasks"]["mean"] == 100
    assert summary["total_tokens"]["all_tasks"]["n"] == 2


def test_skill_library_usage_splits_success_by_calls():
    find_call = {
        "role": "assistant",
        "turn": 1,
        "tool_calls": [{
            "id": "c1",
            "function": {"name": "skill_library_find", "arguments": json.dumps({"task": "parse stl"})},
        }],
    }
    find_result = {
        "role": "tool",
        "turn": 1,
        "tool_call_id": "c1",
        "content": json.dumps({"success": True, "cards": [{"id": "mesh-volume"}, {"id": "unrelated"}]}),
    }
    load_call = {
        "role": "assistant",
        "turn": 1,
        "tool_calls": [{
            "id": "c2",
            "function": {"name": "skill_library_load", "arguments": json.dumps({"skill_id": "mesh-volume"})},
        }],
    }
    load_result = {
        "role": "tool",
        "turn": 1,
        "tool_call_id": "c2",
        "content": json.dumps({"success": True, "gap": "need checkout"}),
    }
    other = {"role": "assistant", "turn": 1, "tool_calls": [{"id": "t", "function": {"name": "terminal", "arguments": "{}"}}]}
    used = common.observe_skill_library(common.empty_skill_library_usage(), [find_call, find_result, load_call, load_result, other])
    assert used["calls"] == {"skill_library_find": 1, "skill_library_load": 1, "skill_library_visit": 0}
    assert used["cards_offered"] == ["mesh-volume", "unrelated"]
    assert used["skills_loaded"] == ["mesh-volume"]
    assert used["loads_with_gap"] == 1
    assert used["involved"] is True
    find_only = common.observe_skill_library(common.empty_skill_library_usage(), [find_call, find_result])
    records = [
        {"status": "complete", "id": "loaded", "grades": [{"turn": 1, "cases": [{"id": "a", "passed": True}]}], "skill_library": used},
        {"status": "complete", "id": "find-only", "grades": [{"turn": 1, "cases": [{"id": "a", "passed": False}]}], "skill_library": find_only},
        {"status": "complete", "id": "plain", "grades": [{"turn": 1, "cases": [{"id": "a", "passed": True}]}]},
    ]
    summary = common.summarize_records(records, [1], benchmark="skillsbench", run="r")
    library = summary["skill_library"]
    assert library["tasks_involved"] == 2
    assert library["tasks_not_involved"] == 1
    assert library["tasks_loaded"] == 1
    assert library["tasks_find_only"] == 1
    assert library["calls_total"]["skill_library_find"] == 2
    assert library["calls_total"]["skill_library_load"] == 1
    assert library["final_macro_success"]["loaded"] == 1.0
    assert library["final_macro_success"]["find_only"] == 0.0
    assert library["final_macro_success"]["not_involved"] == 1.0
    assert "mesh-volume" in library["skills_loaded"]
    report = common.format_report(summary, [])
    assert "skill_library involved=2/3" in report


def test_success_at_k_follows_tool_call_turns_inside_the_conversation():
    user = {"role": "user", "content": "task"}
    call = {"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "terminal", "arguments": "{}"}}]}
    tool = {"role": "tool", "tool_call_id": "c", "content": "{}"}
    done = {"role": "assistant", "content": "done"}
    stamped = [dict(item) for item in (user, call, tool, call, tool, call, tool, done)]
    depth = common.stamp_conversation_turns(stamped)
    assert depth == 4
    assert [item["turn"] for item in stamped] == [1, 2, 2, 3, 3, 4, 4, 4]
    record = {
        "status": "complete",
        "id": "deep",
        "grades": [{"turn": depth, "cases": [{"id": "a", "passed": True}]}],
    }
    summary = common.summarize_records([record], [1, 4], benchmark="skillsbench", run="r")
    assert summary["macro_success_at_k"]["1"] == 0.0
    assert summary["macro_success_at_k"]["4"] == 1.0


def test_summarize_rewrites_a_grade_labeled_turn_1_from_the_transcript(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "benchmark_root", lambda: tmp_path / "benchmark")
    run_dir = tmp_path / "benchmark" / "experiments" / "skillsbench" / "exp"
    record = {
        "status": "complete",
        "id": "deep",
        "grades": [{"turn": 1, "cases": [{"id": "a", "passed": True}]}],
    }
    _write(run_dir / "agent" / "deep" / "result.json", json.dumps(record))
    lines = [
        {"task": "deep", "turn": 1, "type": "turn", "exit_code": 0, "session_ids": ["s"]},
        {"task": "deep", "turn": 1, "role": "user", "content": "task"},
        {"task": "deep", "turn": 1, "role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"task": "deep", "turn": 1, "role": "tool", "tool_call_id": "c1", "content": "{}"},
        {"task": "deep", "turn": 1, "role": "assistant", "tool_calls": [{"id": "c2", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"task": "deep", "turn": 1, "role": "tool", "tool_call_id": "c2", "content": "{}"},
        {"task": "deep", "turn": 1, "role": "assistant", "content": "done"},
    ]
    _write(run_dir / "trajectory.jsonl", "".join(json.dumps(line) + "\n" for line in lines))
    spec = common.Spec(
        "skillsbench", "skillsbench", "https://example.invalid/bench.git",
        lambda *_args: ([], ""), lambda *_args: None, lambda *_args: ("", ""), lambda *_args: [],
    )
    summary_path = tmp_path / "summary.json"
    code = common.cli(spec, ["summarize", "--run", "exp", "--turns", "1,3", "--summary", str(summary_path), "--report", str(tmp_path / "summary.txt")])
    assert code == 0
    body = json.loads(summary_path.read_text(encoding="utf-8"))
    assert body["macro_success_at_k"]["1"] == 0.0
    assert body["macro_success_at_k"]["3"] == 1.0
    stored = json.loads((run_dir / "agent" / "deep" / "result.json").read_text(encoding="utf-8"))
    assert stored["grades"][0]["turn"] == 3


def test_experiment_trajectory_is_one_shared_file(tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    import sqlite3
    con = sqlite3.connect(home / "state.db")
    con.execute("CREATE TABLE sessions (id TEXT, started_at TEXT)")
    con.executemany("INSERT INTO sessions VALUES (?, ?)", [("s1", "1"), ("s2", "2"), ("s3", "3")])
    con.commit()
    con.close()
    assert common.list_session_ids(home) == ["s1", "s2", "s3"]
    assert common.sessions_started_since(["s1"], ["s1", "s2", "s3"]) == ["s2", "s3"]
    path = tmp_path / "experiments" / "skillsbench" / "exp" / "trajectory.jsonl"
    common.append_trajectory_lines(path, [
        {"task": "a", "turn": 1, "role": "user", "content": "go"},
        {"task": "b", "turn": 1, "role": "user", "content": "other"},
    ])
    common.forget_task_trajectory(path, "a")
    common.append_trajectory_lines(path, [
        {"task": "a", "turn": 2, "role": "assistant", "content": "again"},
    ])
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(line["task"], line.get("content")) for line in lines] == [("b", "other"), ("a", "again")]
    kept = common._trajectory_message("a", 2, "s2", {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]})
    assert kept["task"] == "a"
    assert kept["tool_calls"] == [{"id": "c"}]
    assert "content" not in kept


def test_summarize_any_k_counts_passes_at_or_before_k(tmp_path: Path, monkeypatch):
    """A later summarize reads stored grades. A k absent from the original run still works."""
    monkeypatch.setattr(common, "benchmark_root", lambda: tmp_path / "benchmark")
    run_dir = tmp_path / "benchmark" / "experiments" / "skillsbench" / "exp"

    def grade(turn, cases):
        return {"turn": turn, "cases": [{"id": case_id, "passed": passed} for case_id, passed in cases]}

    records = {
        "early": {
            "status": "complete",
            "id": "early",
            "grades": [grade(1, [("a", True)])],
        },
        "later": {
            "status": "complete",
            "id": "later",
            "grades": [
                grade(1, [("b", False)]),
                grade(2, [("b", False)]),
                grade(3, [("b", False)]),
                grade(4, [("b", True)]),
            ],
        },
        "regressed": {
            "status": "complete",
            "id": "regressed",
            "grades": [
                grade(1, [("c1", False), ("c2", True)]),
                grade(2, [("c1", True), ("c2", True)]),
                grade(3, [("c1", False), ("c2", False)]),
            ],
        },
    }
    for task_id, record in records.items():
        _write(run_dir / "agent" / task_id / "result.json", json.dumps(record))
    _write(run_dir / "run.json", json.dumps({"turns": [1, 2, 4, 8], "run": "exp"}))
    summary_path = tmp_path / "summary.json"
    spec = common.Spec(
        "skillsbench", "skillsbench", "https://example.invalid/bench.git",
        lambda *_args: ([], ""), lambda *_args: None, lambda *_args: ("", ""), lambda *_args: [],
    )
    code = common.cli(spec, [
        "summarize", "--run", "exp", "--turns", "3",
        "--summary", str(summary_path), "--report", str(tmp_path / "summary.txt"),
    ])
    assert code == 0
    body = json.loads(summary_path.read_text(encoding="utf-8"))
    assert list(body["macro_success_at_k"]) == ["3"]
    assert abs(body["macro_success_at_k"]["3"] - (2 / 3)) < 1e-9
    assert abs(body["micro_success_at_k"]["3"] - 0.75) < 1e-9
    assert abs(body["macro_auc_at_k"]["3"] - (5 / 9)) < 1e-9
    report = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "at or before k" in report


def test_run_is_isolated_and_resumes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "benchmark_root", lambda: tmp_path / "benchmark")
    calls = {"worker": 0}

    def list_tasks(_root, _split=None):
        return [{"id": "t"}], ""

    def replicate(_root, _task_id, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)

    def prepare(_replica, workspace: Path, _skill_mode):
        workspace.mkdir(parents=True, exist_ok=True)
        return "do the task", ""

    def grade(_replica, _workspace, _task_dir):
        return [{"id": "c", "passed": True, "detail": "", "graded": True}]

    def fake_worker(**kwargs):
        calls["worker"] += 1
        home = Path(kwargs["hermes_home"])
        assert home.name == ".hermes"
        assert home.parent.name in {"r1", "r2"}
        assert not (home.parent / "agent" / "t" / ".hermes").exists()
        assert kwargs["skill_mode"] == "library"
        assert kwargs["model"] == "Qwen/Qwen3.6-27B"
        assert kwargs["provider"] == "qwen-31"
        Path(kwargs["usage"]).write_text(
            json.dumps({"session_id": "s", "input_tokens": 3, "output_tokens": 4, "total_tokens": 7}),
            encoding="utf-8",
        )
        return 0, "ok", ""

    monkeypatch.setattr(common, "invoke_worker", fake_worker)
    spec = common.Spec("skillsbench", "skillsbench", "https://example.invalid/bench.git", list_tasks, replicate, prepare, grade)
    source = tmp_path / "hermes"
    _write(source / "config.yaml", "model: m\n")
    _write(source / "state.db", "secret-session")
    summary = tmp_path / "summary.json"
    report = tmp_path / "summary.txt"
    argv = ["run", "--run", "r1", "--turns", "1,2", "--model", "Qwen/Qwen3.6-27B", "--provider", "qwen-31", "--hermes-home", str(source), "--summary", str(summary), "--report", str(report)]
    assert common.cli(spec, argv) == 0
    assert calls["worker"] == 1
    run_dir = tmp_path / "benchmark" / "experiments" / "skillsbench" / "r1"
    assert (run_dir / ".hermes" / "config.yaml").is_file()
    assert not (run_dir / ".hermes" / "state.db").exists()
    assert common.task_is_complete(run_dir, "t")
    calls["worker"] = 0
    assert common.cli(spec, argv) == 0
    assert calls["worker"] == 0
    record = json.loads((run_dir / "agent" / "t" / "result.json").read_text(encoding="utf-8"))
    record["status"] = "running"
    (run_dir / "agent" / "t" / "result.json").write_text(json.dumps(record), encoding="utf-8")
    assert common.cli(spec, argv) == 0
    assert calls["worker"] == 1
    other = tmp_path / "benchmark" / "experiments" / "skillsbench" / "r2"
    (run_dir / ".hermes" / "config.yaml").write_text("model: changed\n", encoding="utf-8")
    argv[2] = "r2"
    calls["worker"] = 0
    assert common.cli(spec, argv) == 0
    assert (other / ".hermes" / "config.yaml").read_text(encoding="utf-8") == "model: m\n"
    body = json.loads(summary.read_text(encoding="utf-8"))
    assert body["run"] == "r2"
    assert body["model"] == "Qwen/Qwen3.6-27B"
    assert body["provider"] == "qwen-31"
    assert body["final_macro_success"] == 1.0
    assert body["input_tokens"]["all_tasks"]["mean"] == 3


def test_skill_modes_choose_hermes_benchmark_and_library():
    assert common.skill_sources("task") == common.SkillSources(True, True, False)
    assert common.skill_sources("library") == common.SkillSources(False, True, True)
    assert common.skill_sources("library-compatible") == common.SkillSources(True, True, True)
    assert common.skill_sources("library-only") == common.SkillSources(False, False, True)
    assert common.skill_sources("none") == common.SkillSources(False, False, False)
    assert common.disabled_toolsets("task") == ["skill_library"]
    assert common.disabled_toolsets("library") == ["skills"]
    assert common.disabled_toolsets("library-compatible") == []
    assert common.disabled_toolsets("library-only") == ["skills"]
    assert common.disabled_toolsets("none") == ["skills", "skill_library"]


def test_skill_note_names_the_sources_a_mode_offers(tmp_path: Path):
    workspace = tmp_path / "workspace"
    task = common.skill_note("task", workspace, has_task_skills=True)
    assert "Read each SKILL.md" in task
    assert "Do not call" in task and "skill_library_find" in task
    library = common.skill_note("library", workspace, has_task_skills=True)
    assert "Read each SKILL.md" in library and "skill_library_find" in library
    assert "Do not call" not in library
    compatible = common.skill_note("library-compatible", workspace, has_task_skills=True)
    assert "Read each SKILL.md" in compatible and "skill_library_load" in compatible
    assert "Do not call" not in compatible
    only = common.skill_note("library-only", workspace, has_task_skills=False)
    assert "Read each SKILL.md" not in only
    assert "skill_library_find" in only and "Do not call" not in only
    none = common.skill_note("none", workspace, has_task_skills=False)
    assert "Read each SKILL.md" not in none
    assert "Do not call" in none


def test_prepare_copies_task_skills_only_when_the_mode_includes_them(tmp_path: Path):
    replica = tmp_path / "replica"
    _write(replica / "environment" / "skills" / "pid" / "SKILL.md", "pid procedure\n")
    _write(replica / "environment" / "sensor_data.csv", "t\n")
    _write(replica / "task.md", "Implement it under /root.\n")
    for mode, present in (
        ("task", True),
        ("library", True),
        ("library-compatible", True),
        ("library-only", False),
        ("none", False),
    ):
        workspace = tmp_path / mode
        prompt, error = skillsbench.prepare(replica, workspace, mode)
        assert error == ""
        assert (workspace / "sensor_data.csv").is_file()
        assert (workspace / "skills" / "pid" / "SKILL.md").is_file() is present
        assert ("Read each SKILL.md" in prompt) is present


def test_science_domain_knowledge_follows_the_benchmark_source(tmp_path: Path):
    replica = tmp_path / "replica"
    _write(replica / "task.json", json.dumps({
        "gold_program_name": "run.py",
        "task_inst": "fit the curve",
        "domain_knowledge": "use a PID loop",
        "dataset_folder_tree": "",
        "dataset_preview": "",
        "output_fname": "out.csv",
    }))
    for mode, present in (("library", True), ("library-only", False), ("task", True), ("none", False)):
        prompt, error = scienceagentbench.prepare(replica, tmp_path / mode, mode)
        assert error == ""
        assert ("use a PID loop" in prompt) is present


def test_ensure_hermes_skills_syncs_into_the_experiment_home(tmp_path: Path, monkeypatch):
    import tools.skills_sync as sync

    seen = {}

    def fake_sync(quiet=False):
        seen["home"] = os.environ["HERMES_HOME"]
        seen["quiet"] = quiet
        return {}

    monkeypatch.setattr(sync, "sync_skills", fake_sync)
    home = tmp_path / "exp" / ".hermes"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "source"))
    common.ensure_hermes_skills(home)
    assert seen == {"home": str(home), "quiet": True}
    assert os.environ["HERMES_HOME"] == str(tmp_path / "source")


def test_run_seeds_hermes_skills_only_for_modes_that_include_them(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(common, "benchmark_root", lambda: tmp_path / "benchmark")
    seeded = []
    monkeypatch.setattr(common, "ensure_hermes_skills", lambda home: seeded.append(home))

    def fake_worker(**kwargs):
        Path(kwargs["usage"]).write_text(
            json.dumps({"session_id": "s", "input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),
            encoding="utf-8",
        )
        return 0, "", ""

    monkeypatch.setattr(common, "invoke_worker", fake_worker)

    def list_tasks(_root, _split=None):
        return [{"id": "t"}], ""

    def replicate(_root, _task_id, dest: Path):
        dest.mkdir(parents=True, exist_ok=True)

    def prepare(_replica, workspace: Path, _skill_mode):
        workspace.mkdir(parents=True, exist_ok=True)
        return "do the task", ""

    def grade(_replica, _workspace, _task_dir):
        return [{"id": "c", "passed": True, "detail": "", "graded": True}]

    spec = common.Spec(
        "skillsbench", "skillsbench", "https://example.invalid/bench.git",
        list_tasks, replicate, prepare, grade,
    )
    source = tmp_path / "hermes"
    _write(source / "config.yaml", "model: m\n")
    argv = ["run", "--run", "seed", "--turns", "1", "--skill-mode", "library", "--hermes-home", str(source)]
    assert common.cli(spec, argv) == 0
    assert seeded == []
    argv[argv.index("library")] = "task"
    argv[argv.index("seed")] = "seed-task"
    assert common.cli(spec, argv) == 0
    assert len(seeded) == 1
    assert seeded[0].name == ".hermes"
    assert seeded[0].parent.name == "seed-task"


def test_grader_paths_cover_verifier_oracle_and_rubric(tmp_path: Path):
    bench = tmp_path / "benchmark"
    _write(bench / "skillsbench" / "tasks" / "demo" / "verifier" / "test_outputs.py", "SECRET\n")
    _write(bench / "skillsbench" / "tasks" / "demo" / "oracle" / "solve.sh", "ORACLE\n")
    _write(bench / "skillsbench" / "tasks" / "demo" / "environment" / "input.txt", "data\n")
    _write(bench / "terminal-bench" / "tasks" / "demo" / "tests" / "test_outputs.py", "SECRET\n")
    _write(bench / "frontier-evals" / "paper" / "rubric.json", "{}\n")
    _write(bench / "experiments" / "run" / "agent" / "demo" / "workspace" / "answer.txt", "yes\n")
    directories, rubrics = jail.grader_paths(bench)
    names = {path.name for path in directories}
    assert {"verifier", "oracle", "tests"} <= names
    assert rubrics == [bench / "frontier-evals" / "paper" / "rubric.json"]
    assert all("workspace" not in path.parts for path in directories)


def test_jail_mounts_hide_graders_and_keep_the_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    bench = repo / "benchmark"
    verifier = bench / "skillsbench" / "tasks" / "demo" / "verifier" / "test_outputs.py"
    _write(verifier, "SECRET\n")
    agent = bench / "experiments" / "run" / "agent" / "demo"
    _write(agent / "workspace" / "answer.txt", "yes\n")
    home = bench / "experiments" / "run" / ".hermes"
    home.mkdir(parents=True)
    blank = tmp_path / "blank"
    blank.write_text("", encoding="utf-8")
    args = jail.mount_args(repo=repo, agent_dir=agent, hermes_home=home, blank_file=blank)
    assert f"{verifier.parent}:ro" in args
    assert f"{agent}:{agent}" in args
    assert str(verifier) not in args


def test_container_cannot_read_the_adaptive_cruise_verifier():
    if shutil.which("docker") is None:
        return
    repo = Path(__file__).resolve().parents[2]
    verifier = repo / "benchmark/skillsbench/tasks/adaptive-cruise-control/verifier/test_outputs.py"
    if not verifier.is_file():
        return
    jail.ensure_image()
    agent = Path("/tmp/bench-jail-test-agent")
    home = Path("/tmp/bench-jail-test-home")
    agent.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    blank = agent / "blank"
    blank.write_text("", encoding="utf-8")
    script = (
        "import os,sys\n"
        f"sys.exit(0 if not os.path.isfile({str(verifier)!r}) else 2)\n"
    )
    proc = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "host",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--entrypoint", sys.executable,
            *jail.mount_args(repo=repo, agent_dir=agent, hermes_home=home, blank_file=blank),
            jail.JAIL_IMAGE, "-c", script,
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
