"""Run the benchmark worker where grader files are not readable.

The parent process still grades from the replica. The worker runs in a
container whose view of ``benchmark/`` has the grader directories and
``rubric.json`` files covered. Its workspace and the run's ``.hermes`` stay
writable. A prompt that says "do not open the verifier" is not this boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

JAIL_IMAGE = "bench-agent-jail"
_GRADER_DIRS = {
    "verifier",
    "oracle",
    "eval_programs",
    "gold_programs",
    "scoring_rubrics",
}
_SKIP_WALK = {".git", ".hermes", "agent", "datasets", "node_modules", "workspace", "__pycache__"}


def grader_paths(benchmark: Path) -> tuple[list[Path], list[Path]]:
    """Return ``(directories, rubric files)`` the worker must not read."""
    directories: list[Path] = []
    rubrics: list[Path] = []
    if not benchmark.is_dir():
        return directories, rubrics
    for current, dirnames, filenames in os.walk(benchmark):
        held: list[str] = []
        for name in dirnames:
            path = Path(current) / name
            if name in _GRADER_DIRS or (name == "tests" and (path / "test_outputs.py").is_file()):
                directories.append(path)
                continue
            if name not in _SKIP_WALK:
                held.append(name)
        dirnames[:] = held
        if "rubric.json" in filenames:
            rubrics.append(Path(current) / "rubric.json")
    return directories, rubrics


def skill_cache(repo: Path) -> Path | None:
    """The cache named in ``skill-lib/skilllib.json``, when that directory exists."""
    config = repo / "skill-lib" / "skilllib.json"
    if not config.is_file():
        return None
    try:
        raw = json.loads(config.read_text(encoding="utf-8")).get("cache")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo / "skill-lib" / path
    return path if path.is_dir() else None


def mount_args(
    *,
    repo: Path,
    agent_dir: Path,
    hermes_home: Path,
    blank_file: Path,
) -> list[str]:
    """Docker mount flags: repo read-only, graders covered, workspace writable."""
    args = [
        "-v", f"{repo}:{repo}:ro",
        "-v", "/lib:/lib:ro",
        "-v", "/lib64:/lib64:ro",
        "-v", "/usr:/usr:ro",
        "-v", "/etc:/etc:ro",
        "--tmpfs", "/tmp:rw,mode=1777",
    ]
    python_home = Path(sys.executable).resolve().parents[1]
    if not _is_relative_to(python_home, repo):
        args.extend(["-v", f"{python_home}:{python_home}:ro"])
    directories, rubrics = grader_paths(repo / "benchmark")
    for path in directories:
        args.extend(["--tmpfs", f"{path}:ro"])
    for path in rubrics:
        args.extend(["-v", f"{blank_file}:{path}:ro"])
    cache = skill_cache(repo)
    if cache is not None and not _is_relative_to(cache, repo):
        args.extend(["-v", f"{cache}:{cache}:ro"])
    project_cache = repo / "skill-lib" / "cache"
    if project_cache.is_dir():
        args.extend(["-v", f"{project_cache}:{project_cache}"])
    args.extend([
        "-v", f"{agent_dir}:{agent_dir}",
        "-v", f"{hermes_home}:{hermes_home}",
    ])
    return args


def command(
    cmd: list[str],
    *,
    workspace: Path,
    hermes_home: Path,
    agent_dir: Path,
    env_file: Path,
    blank_file: Path,
) -> list[str]:
    """Wrap ``cmd`` so the worker cannot read grader files."""
    repo = Path(__file__).resolve().parents[2]
    ensure_image()
    return [
        "docker", "run", "--rm", "--network", "host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--entrypoint", cmd[0],
        "--env-file", str(env_file),
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-w", str(workspace),
        *mount_args(repo=repo, agent_dir=agent_dir, hermes_home=hermes_home, blank_file=blank_file),
        JAIL_IMAGE,
        *cmd[1:],
    ]


def ensure_image() -> None:
    probe = subprocess.run(
        ["docker", "image", "inspect", JAIL_IMAGE],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0:
        return
    build = subprocess.run(
        ["docker", "build", "-t", JAIL_IMAGE, "-"],
        input="FROM scratch\n",
        capture_output=True, text=True, check=False,
    )
    if build.returncode != 0:
        detail = (build.stderr or build.stdout or "docker build failed").strip()
        raise SystemExit(f"the agent jail image could not be built: {detail}")


def write_env_file(path: Path, env: dict[str, str]) -> None:
    lines = []
    for key, value in env.items():
        if not key or "=" in key or "\n" in key or "\n" in value or "\x00" in value:
            continue
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
