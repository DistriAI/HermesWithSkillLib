"""Adapter for the SkillNet baseline (zjunlp/SkillNet).

Hermes stays the agent. This module locates the checkout's analysis index
and calls that checkout's ``route``. It does not read or write SkillLib
routing cards.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "SkillNet"
_PACKAGE = _PACKAGE_ROOT / "skillnet-ai" / "src"
_INDEX = _PACKAGE_ROOT / "hermes_index"
_SKILLS = _PACKAGE_ROOT / "hermes_lib" / "skills"
TOOL_NAMES = ("skill_net_route", "skill_net_load")
_TOP = 5
_TASK_INSTRUCTION = (
    "Before writing the solution, call skill_net_route for this task and "
    "skill_net_load on the skills it returns (at most 3). The choice is SkillNet's route. "
    "Adapt what you load; a skill is a procedure, not the answer."
)


def package_root() -> Path | None:
    if (_PACKAGE / "skillnet_ai" / "__init__.py").is_file():
        return _PACKAGE_ROOT
    return None


def index_dir() -> Path | None:
    if (_INDEX / "CURRENT").is_file():
        return _INDEX
    return None


def skillnet_available() -> bool:
    return package_root() is not None and index_dir() is not None and _SKILLS.is_dir()


def unavailable_reason() -> str:
    if package_root() is None:
        return "SkillNet checkout is missing at baselines/SkillNet."
    if index_dir() is None or not _SKILLS.is_dir():
        return (
            "SkillNet has not published its index. "
            "Run python baselines/SkillNet/hermes_build.py --publish."
        )
    return ""


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def task_instruction() -> str:
    return _TASK_INSTRUCTION


def skillnet_prompt(available_tools: set[str] | None) -> str:
    if not available_tools or any(name not in available_tools for name in TOOL_NAMES):
        return ""
    if not skillnet_available():
        return ""
    return _TASK_INSTRUCTION


def tool_specs() -> list[dict]:
    return [
        {
            "name": "skill_net_route",
            "description": (
                "Choose skills for a task from the SkillNet index built from this project's "
                "skill snapshot. The choice is SkillNet route, over its own graph and embeddings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What the task needs, in its own words."},
                },
                "required": ["task"],
            },
        },
        {
            "name": "skill_net_load",
            "description": "Read one skill returned by skill_net_route. Pass its path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path from skill_net_route."},
                },
                "required": ["path"],
            },
        },
    ]


def _client():
    entry = str(_PACKAGE)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    from skillnet_ai import SkillNetClient

    return SkillNetClient()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def call_tool(name: str, arguments: dict | None = None) -> dict:
    arguments = arguments or {}
    if not skillnet_available():
        return {"success": False, "error": unavailable_reason()}
    if name == "skill_net_route":
        task = str(arguments.get("task") or "").strip()
        if not task:
            return {"success": False, "error": "task is required"}
        try:
            result = _client().route(task, index_dir=index_dir(), k=_TOP)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        skills = []
        for skill in result.skills:
            skills.append({
                "skill_id": skill.skill_id,
                "name": skill.name,
                "path": skill.path,
                "reason": skill.reason,
            })
        return {"success": True, "skills": skills}
    if name == "skill_net_load":
        raw = str(arguments.get("path") or "").strip()
        if not raw:
            return {"success": False, "error": "path is required"}
        target = Path(raw)
        if target.is_dir():
            target = target / "SKILL.md"
        if not _inside(_SKILLS, target) or not target.is_file():
            return {"success": False, "error": "path is not a skill in the SkillNet library"}
        return {"success": True, "path": str(target), "text": target.read_text(encoding="utf-8", errors="replace")}
    return {"success": False, "error": f"unknown tool {name}"}
