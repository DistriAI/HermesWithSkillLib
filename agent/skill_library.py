"""Adapter for the skill-lib submodule (DistriAI/SkillLib).

The catalog is a domain → scenario → skill index of tens of thousands of
skills. It stays out of ``<available_skills>``: the session sees a short
domain list plus ``skill_library_find`` / ``skill_library_load`` /
``skill_library_visit``. Cards are parsed once per catalog signature.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The library's finder default. Not a tool parameter: instructional tools
# do not paginate, and the resident prompt caps how many bodies are loaded.
_FIND_LIMIT = 5
_LIBRARY_TOOLS = ("skill_library_find", "skill_library_load", "skill_library_visit")

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "skill-lib"

_import_lock = threading.Lock()
_finder_lock = threading.Lock()
_skillrepo: Any = None
_finder_cache: dict[tuple, Any] = {}
_prompt_cache: str | None = None


def package_root() -> Path | None:
    """Submodule checkout that contains ``src/skillrepo``, or None."""
    root = _PACKAGE_ROOT
    if (root / "src" / "skillrepo" / "__init__.py").is_file():
        return root
    return None


def catalog_root() -> Path | None:
    """Directory whose ``catalog/`` the agent reads. The submodule, unless a test overrides this."""
    return package_root()


def library_available() -> bool:
    """True when the routing-card index is on disk. Does not import skillrepo."""
    root = catalog_root()
    return root is not None and (root / "catalog" / "routing_cards.jsonl").is_file()


def _clear_finder_cache() -> None:
    """Test hook."""
    with _finder_lock:
        _finder_cache.clear()


def skillrepo_module():
    """Import ``skillrepo`` from the submodule checkout."""
    global _skillrepo
    if _skillrepo is not None:
        return _skillrepo
    with _import_lock:
        if _skillrepo is not None:
            return _skillrepo
        root = package_root()
        if root is None:
            raise ImportError("skill-lib submodule is not checked out")
        src = str(root / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        import skillrepo
        _skillrepo = skillrepo
        return skillrepo


def _prompt_constants() -> tuple[tuple, int]:
    skillrepo_module()
    from skillrepo.sorting.promptpack import DOMAIN_SENTENCES, LOAD_CAP
    return DOMAIN_SENTENCES, LOAD_CAP


def library_prompt(available_tools: set[str] | None) -> str:
    """Stable system-prompt fragment. Empty unless this session offers the finder.

    The fragment is the library's domain sentences plus the instruction to
    call the Hermes tools. It does not list cards, so it stays byte-stable
    for the life of a conversation.
    """
    global _prompt_cache
    if not available_tools or any(name not in available_tools for name in _LIBRARY_TOOLS):
        return ""
    if not library_available():
        return ""
    if _prompt_cache is not None:
        return _prompt_cache
    try:
        sentences, load_cap = _prompt_constants()
    except Exception:
        logger.debug("skill library prompt unavailable", exc_info=True)
        return ""
    lines = [
        "## Skill library",
        "The skill-lib checkout is a hierarchical catalog: domain, then scenario, then skill.",
        "After the skills listed above, use this catalog when the task still needs a procedure from it.",
        "",
        "Domains:",
    ]
    for name, sentence in sentences:
        lines.append(f"- {name}: {sentence}")
    lines.extend([
        "",
        "Before you decide that no skill applies, call skill_library_find with the user's task.",
        f"Load at most {load_cap} skill bodies with skill_library_load, and only after a card matches.",
        "Follow the loaded procedure. When a card is a combined skill, its conflict_note names the disagreement; follow that note.",
        "When skill_library_load returns a gap, call skill_library_visit with the repository it names, then load the skill again.",
    ])
    text = "\n".join(lines) + "\n"
    _prompt_cache = text
    return text


def _file_sig(path: Path) -> tuple[int, int]:
    try:
        st = path.stat()
    except OSError:
        return (0, 0)
    return (st.st_mtime_ns, st.st_size)


def _finder(root: Path):
    cards_path = root / "catalog" / "routing_cards.jsonl"
    families_path = root / "catalog" / "task_families.json"
    key = (str(root.resolve()), *_file_sig(cards_path), *_file_sig(families_path))
    with _finder_lock:
        cached = _finder_cache.get(key)
        if cached is not None:
            return cached
    skillrepo_module()
    from skillrepo.sorting.cards import CARDS_PATH, load_routing_cards
    from skillrepo.sorting.finder import Finder, default_families_path, load_families
    finder = Finder(load_routing_cards(root / CARDS_PATH), load_families(default_families_path(root)))
    with _finder_lock:
        _finder_cache[key] = finder
    return finder


def _constraint_set(constraints: list[str] | None) -> set[str] | None | str:
    """A constraint set, None when omitted, or an error string."""
    if constraints is None:
        return None
    if isinstance(constraints, str):
        text = constraints.strip()
        return {text} if text else None
    if not isinstance(constraints, (list, tuple, set)):
        return "constraints must be a list of strings"
    return {str(item).strip() for item in constraints if str(item).strip()} or None


def _resolve_root(root: Path | None) -> Path | None:
    if root is not None:
        return root
    return catalog_root()


def find_in_library(
    task: str,
    *,
    root: Path | None = None,
    domain: str | None = None,
    constraints: list[str] | None = None,
) -> dict[str, Any]:
    """Return the library's card views for *task*."""
    query = str(task or "").strip()
    if not query:
        return {"success": False, "error": "task is required"}
    resolved = _resolve_root(root)
    if resolved is None or not (resolved / "catalog" / "routing_cards.jsonl").is_file():
        return {"success": False, "error": "Skill library catalog is not checked out."}
    domain_name = str(domain).strip() if domain else ""
    constraint_set = _constraint_set(constraints)
    if isinstance(constraint_set, str):
        return {"success": False, "error": constraint_set}
    try:
        found = _finder(resolved).find(
            query,
            domain=domain_name or None,
            constraints=constraint_set,
            limit=_FIND_LIMIT,
        )
    except ImportError:
        return {"success": False, "error": "Skill library package is not checked out."}
    except Exception as exc:
        logger.debug("skill library find failed", exc_info=True)
        return {"success": False, "error": f"Skill library search failed: {exc}"}
    return {"success": True, "cards": found.get("cards") or [], "pairs": found.get("pairs") or []}


def load_from_library(skill_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Return one cached skill body, or the checkout gap the library recorded."""
    skill_id = str(skill_id or "").strip()
    if not skill_id:
        return {"success": False, "error": "skill id is required"}
    resolved = _resolve_root(root)
    if resolved is None or not (resolved / "catalog" / "routing_cards.jsonl").is_file():
        return {"success": False, "error": "Skill library catalog is not checked out."}
    try:
        loaded = _finder(resolved).load(resolved, skill_id)
    except KeyError:
        return {"success": False, "error": f"Unknown skill id: {skill_id}"}
    except ImportError:
        return {"success": False, "error": "Skill library package is not checked out."}
    except Exception as exc:
        logger.debug("skill library load failed", exc_info=True)
        return {"success": False, "error": f"Skill library load failed: {exc}"}
    result = {"success": True, **loaded}
    if result.get("gap"):
        result["next"] = (
            "Call skill_library_visit once for each owner/name in gap, then call skill_library_load again."
        )
    return result


def visit_in_library(full_name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Check out one catalog repository so a loaded skill's scripts are on disk."""
    full_name = str(full_name or "").strip()
    if not full_name:
        return {"success": False, "error": "repository name is required"}
    resolved = _resolve_root(root)
    if resolved is None:
        return {"success": False, "error": "Skill library catalog is not checked out."}
    try:
        module = skillrepo_module()
        dest = module.visit_repository(full_name, resolved)
    except ImportError:
        return {"success": False, "error": "Skill library package is not checked out."}
    except Exception as exc:
        logger.debug("skill library visit failed", exc_info=True)
        return {"success": False, "error": str(exc)}
    return {
        "success": True,
        "repository": full_name,
        "path": str(dest),
        "note": (
            "Read the skill files under path. The stored catalog gap stays until skill-lib "
            "rewrites checkout state, so a later skill_library_load can still report that gap."
        ),
    }
