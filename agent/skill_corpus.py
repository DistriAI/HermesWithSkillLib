"""Adapter for the SkillCorpus baseline (EverMind-AI/SkillCorpus).

Hermes stays the agent. This module locates the checkout's own library
(``hermes_lib``) and searches it with that checkout's skillsearch engine.
It does not read or write SkillLib routing cards.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "SkillCorpus"
_ENGINE = _PACKAGE_ROOT / "skillcorpus_plugin" / "engine-python"
_LIBRARY = _PACKAGE_ROOT / "hermes_lib"
TOOL_NAMES = ("skill_corpus_find", "skill_corpus_load")
_TOP = 5
_TASK_INSTRUCTION = (
    "Before writing the solution, call skill_corpus_find for this task and "
    "skill_corpus_load on the best matches (at most 3). The search is SkillCorpus's. "
    "Adapt what you load; a skill is a procedure, not the answer."
)


def package_root() -> Path | None:
    if (_PACKAGE_ROOT / "skillcorpus" / "__init__.py").is_file():
        return _PACKAGE_ROOT
    return None


def library_root() -> Path | None:
    if (_LIBRARY / "index.db").is_file() and (_LIBRARY / "skills").is_dir():
        return _LIBRARY
    return None


def corpus_available() -> bool:
    return package_root() is not None and library_root() is not None and _ENGINE.is_dir()


def unavailable_reason() -> str:
    if package_root() is None:
        return "SkillCorpus checkout is missing at baselines/SkillCorpus."
    if library_root() is None:
        return (
            "SkillCorpus has not published its library. "
            "Run python baselines/SkillCorpus/hermes_build.py --publish."
        )
    return "SkillCorpus skillsearch engine is missing."


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def task_instruction() -> str:
    return _TASK_INSTRUCTION


def corpus_prompt(available_tools: set[str] | None) -> str:
    if not available_tools or any(name not in available_tools for name in TOOL_NAMES):
        return ""
    if not corpus_available():
        return ""
    return _TASK_INSTRUCTION


def tool_specs() -> list[dict]:
    return [
        {
            "name": "skill_corpus_find",
            "description": (
                "Search the SkillCorpus library built from this project's skill snapshot. "
                "Ranking is SkillCorpus skillsearch, over the skills its pipeline kept."
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
            "name": "skill_corpus_load",
            "description": "Read one skill returned by skill_corpus_find. Pass its path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The path from skill_corpus_find."},
                },
                "required": ["path"],
            },
        },
    ]


def _engine():
    entry = str(_ENGINE)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    from skillsearch.local_pool import LocalPool
    from skillsearch.local_store import DirectorySkillStore

    return DirectorySkillStore, LocalPool


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def call_tool(name: str, arguments: dict | None = None) -> dict:
    arguments = arguments or {}
    if not corpus_available():
        return {"success": False, "error": unavailable_reason()}
    library = library_root()
    assert library is not None
    if name == "skill_corpus_find":
        task = str(arguments.get("task") or "").strip()
        if not task:
            return {"success": False, "error": "task is required"}
        store_cls, pool_cls = _engine()
        store = store_cls([(library / "skills", "skillcorpus")], max_depth=8)
        hits = pool_cls(store).search(task, top_k=_TOP)
        files = store.list_all()
        cards = []
        for hit in hits:
            match = next((item for item in files if item.name == hit.name), None)
            if match is None:
                continue
            cards.append({
                "name": match.name,
                "score": hit.score,
                "path": str(match.path),
                "description": match.description,
            })
        return {"success": True, "skills": cards}
    if name == "skill_corpus_load":
        raw = str(arguments.get("path") or "").strip()
        if not raw:
            return {"success": False, "error": "path is required"}
        target = Path(raw)
        if target.is_dir():
            target = target / "SKILL.md"
        if not _inside(library / "skills", target) or not target.is_file():
            return {"success": False, "error": "path is not a skill in the SkillCorpus library"}
        return {"success": True, "path": str(target), "text": target.read_text(encoding="utf-8", errors="replace")}
    return {"success": False, "error": f"unknown tool {name}"}
