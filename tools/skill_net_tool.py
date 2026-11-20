"""Tools for the SkillNet baseline index.

Schemas live in ``agent.skill_net``. This file only registers them on the
``skill_net`` toolset. ``check_fn`` is the checkout plus the index that
checkout built. Without those, the tools stay off the schema.
"""

import json

from agent.skill_net import call_tool, skillnet_available, tool_specs
from tools.registry import registry


def check_skill_net() -> bool:
    return skillnet_available()


def _handle(name: str):
    def handle(args, **_kwargs):
        return json.dumps(call_tool(name, args or {}), ensure_ascii=False)

    return handle


for _spec in tool_specs():
    registry.register(
        name=_spec["name"],
        toolset="skill_net",
        schema=_spec,
        handler=_handle(_spec["name"]),
        check_fn=check_skill_net,
        emoji="🕸️",
    )
