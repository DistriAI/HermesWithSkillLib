"""Tools for the SkillCorpus baseline library.

Schemas live in ``agent.skill_corpus``. This file only registers them on the
``skill_corpus`` toolset. ``check_fn`` is the checkout plus the library that
checkout built. Without those, the tools stay off the schema.
"""

import json

from agent.skill_corpus import call_tool, corpus_available, tool_specs
from tools.registry import registry


def check_skill_corpus() -> bool:
    return corpus_available()


def _handle(name: str):
    def handle(args, **_kwargs):
        return json.dumps(call_tool(name, args or {}), ensure_ascii=False)

    return handle


for _spec in tool_specs():
    registry.register(
        name=_spec["name"],
        toolset="skill_corpus",
        schema=_spec,
        handler=_handle(_spec["name"]),
        check_fn=check_skill_corpus,
        emoji="📚",
    )
