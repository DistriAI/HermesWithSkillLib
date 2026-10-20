"""Tools for the hierarchical skill-lib catalog.

Registered on the ``skill_library`` toolset and listed in the core tool
list, so a normal session can search the catalog. ``check_fn`` is the
checkout: with no ``catalog/routing_cards.jsonl`` the tools stay off the
schema. Background skill review enables only the ``skills`` toolset, so
it does not receive these tools.
"""

import json

from agent.skill_library import (
    find_in_library,
    library_available,
    load_from_library,
    visit_in_library,
)
from tools.registry import registry


def check_skill_library() -> bool:
    return library_available()


def _dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def skill_library_find(task: str, domain: str = "", constraints: list | None = None) -> str:
    return _dump(find_in_library(task, domain=domain or None, constraints=constraints))


def skill_library_load(skill_id: str) -> str:
    return _dump(load_from_library(skill_id))


def skill_library_visit(repository: str) -> str:
    return _dump(visit_in_library(repository))


FIND_SCHEMA = {
    "name": "skill_library_find",
    "description": (
        "Search the hierarchical skill library (domain, then scenario, then skill) "
        "and return a few matching cards. Call this before deciding that no skill applies. "
        "Each card includes id, name, description, scopes, family, and conflict_note. "
        "Load a matching card with skill_library_load."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "The user's task, in their words.",
            },
            "domain": {
                "type": "string",
                "description": "Optional domain name from the skill library prompt. Omit to search every domain.",
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tool names already available, such as python or git. A card whose precondition is missing is ranked lower.",
            },
        },
        "required": ["task"],
    },
}

LOAD_SCHEMA = {
    "name": "skill_library_load",
    "description": (
        "Load one skill body from the hierarchical skill library by the card id "
        "returned from skill_library_find. When bundled scripts are not checked out, "
        "text is empty and gap names the repository to pass to skill_library_visit."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "skill_id": {
                "type": "string",
                "description": "Card id from skill_library_find.",
            },
        },
        "required": ["skill_id"],
    },
}

VISIT_SCHEMA = {
    "name": "skill_library_visit",
    "description": (
        "Check out a skill repository named by a skill_library_load gap, so the "
        "skill's bundled scripts and references are on disk. Pass owner/name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "repository": {
                "type": "string",
                "description": "GitHub repository id from the load gap, as owner/name.",
            },
        },
        "required": ["repository"],
    },
}

registry.register(
    name="skill_library_find",
    toolset="skill_library",
    schema=FIND_SCHEMA,
    handler=lambda args, **kw: skill_library_find(
        task=args.get("task", ""),
        domain=args.get("domain", "") or "",
        constraints=args.get("constraints"),
    ),
    check_fn=check_skill_library,
    emoji="🗂️",
)

registry.register(
    name="skill_library_load",
    toolset="skill_library",
    schema=LOAD_SCHEMA,
    handler=lambda args, **kw: skill_library_load(skill_id=args.get("skill_id", "")),
    check_fn=check_skill_library,
    emoji="🗂️",
)

registry.register(
    name="skill_library_visit",
    toolset="skill_library",
    schema=VISIT_SCHEMA,
    handler=lambda args, **kw: skill_library_visit(repository=args.get("repository", "")),
    check_fn=check_skill_library,
    emoji="🗂️",
)
