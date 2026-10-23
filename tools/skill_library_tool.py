"""Tools for the hierarchical skill-lib catalog.

The names, schemas, and result objects come from ``skilllib.integrate``.
This file only registers them on the ``skill_library`` toolset. ``check_fn``
is the checkout: with no ``catalog/routing_cards.jsonl`` the tools stay off
the schema. Background skill review enables only the ``skills`` toolset, so
it does not receive these tools.
"""

import json

from agent.skill_library import call_tool, library_available, tool_specs
from tools.registry import registry


def check_skill_library() -> bool:
    return library_available()


def _handle(name: str):
    def handle(args, **_kwargs):
        return json.dumps(call_tool(name, args or {}), ensure_ascii=False)

    return handle


for _spec in tool_specs():
    registry.register(
        name=_spec["name"],
        toolset="skill_library",
        schema=_spec,
        handler=_handle(_spec["name"]),
        check_fn=check_skill_library,
        emoji="🗂️",
    )
