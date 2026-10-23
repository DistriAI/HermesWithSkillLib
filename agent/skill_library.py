"""Adapter for the skill-lib submodule (DistriAI/SkillLib).

The catalog stays out of ``<available_skills>``. A session that offers the
three host tools receives the prompt, schemas, and tool results defined by
``skilllib.integrate``. This module only locates the checkout and forwards
those calls. Hosts other than Hermes should use ``skilllib.integrate`` the
same way.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "skill-lib"

_import_lock = threading.Lock()
_skilllib: Any = None
_prompt_cache: str | None = None


def package_root() -> Path | None:
    """Submodule checkout that contains ``src/skilllib``, or None."""
    root = _PACKAGE_ROOT
    if (root / "src" / "skilllib" / "__init__.py").is_file():
        return root
    return None


def catalog_root() -> Path | None:
    """Directory whose ``catalog/`` the agent reads. The submodule, unless a test overrides this."""
    return package_root()


def library_available() -> bool:
    """True when the routing-card index is on disk. Does not import skilllib."""
    root = catalog_root()
    return root is not None and (root / "catalog" / "routing_cards.jsonl").is_file()


def skilllib_module():
    """Import ``skilllib`` from the submodule checkout."""
    global _skilllib
    if _skilllib is not None:
        return _skilllib
    with _import_lock:
        if _skilllib is not None:
            return _skilllib
        root = package_root()
        if root is None:
            raise ImportError("skill-lib submodule is not checked out")
        src = str(root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import skilllib
        _skilllib = skilllib
        return skilllib


def _clear_finder_cache() -> None:
    """Test hook. The cache lives in ``skilllib.integrate``."""
    try:
        skilllib_module().integrate.clear_finder_cache()
    except ImportError:
        return


def tool_names() -> tuple[str, ...]:
    try:
        return tuple(skilllib_module().TOOL_NAMES)
    except ImportError:
        return ("skill_library_find", "skill_library_load", "skill_library_visit")


def tool_specs() -> list[dict]:
    """Schemas from SkillLib. Empty when the submodule is not checked out."""
    try:
        return list(skilllib_module().tool_specs())
    except ImportError:
        return []


def library_prompt(available_tools: set[str] | None) -> str:
    """The SkillLib resident prompt, when this session offers every host tool."""
    global _prompt_cache
    names = tool_names()
    if not available_tools or any(name not in available_tools for name in names):
        return ""
    if not library_available():
        return ""
    if _prompt_cache is not None:
        return _prompt_cache
    try:
        text = skilllib_module().resident_prompt()
    except Exception:
        logger.debug("skill library prompt unavailable", exc_info=True)
        return ""
    _prompt_cache = text
    return text


def task_instruction() -> str:
    """The SkillLib task paragraph. Empty when the submodule is not checked out."""
    try:
        return skilllib_module().task_instruction()
    except ImportError:
        return ""


def call_tool(name: str, arguments: dict | None = None, *, root: Path | None = None) -> dict:
    """Forward one tool call. ``root`` defaults to this checkout's catalog."""
    resolved = catalog_root() if root is None else root
    try:
        module = skilllib_module()
    except ImportError:
        return {"success": False, "error": "Skill library package is not checked out."}
    return module.call_tool(name, arguments or {}, root=resolved)


def find_in_library(
    task: str,
    *,
    root: Path | None = None,
    domain: str | None = None,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    return call_tool(
        "skill_library_find",
        {"task": task, "domain": domain, "constraints": constraints},
        root=root,
    )


def load_from_library(skill_id: str, *, root: Path | None = None) -> dict[str, Any]:
    return call_tool("skill_library_load", {"skill_id": skill_id}, root=root)


def visit_in_library(full_name: str, *, root: Path | None = None) -> dict[str, Any]:
    return call_tool("skill_library_visit", {"repository": full_name}, root=root)
