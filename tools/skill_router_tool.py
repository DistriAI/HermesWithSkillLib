"""Tools for the SkillRouter baseline pool.

Schemas live in ``agent.skill_router``. This file only registers them on the
``skill_router`` toolset. ``check_fn`` is the checkout plus the downloaded
skill pool and the torch runtime: without those the tools stay off the schema.
"""

import json

from agent.skill_router import call_tool, router_available, tool_specs
from tools.registry import registry


def check_skill_router() -> bool:
    return router_available()


def _handle(name: str):
    def handle(args, **_kwargs):
        return json.dumps(call_tool(name, args or {}), ensure_ascii=False)

    return handle


for _spec in tool_specs():
    registry.register(
        name=_spec["name"],
        toolset="skill_router",
        schema=_spec,
        handler=_handle(_spec["name"]),
        check_fn=check_skill_router,
        emoji="🧭",
    )
