"""Registry dispatch for the hierarchical skill library tools."""

import json
from pathlib import Path

import pytest

from agent.skill_library import _clear_finder_cache, package_root
from tools.registry import discover_builtin_tools, registry


def _library(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    card = {
        "id": "demo::gsea",
        "name": "gsea",
        "aliases": ["gsea"],
        "kind": "separate",
        "description": "Run GSEA on a ranked gene list.",
        "positive_scope": ["ranked gene list"],
        "negative_scope": [],
        "inputs": [],
        "outputs": [],
        "domains": ["science"],
        "preconditions": [],
        "provenance": {"conflict_note": ""},
        "body": {},
        "sources": [],
        "links": {},
        "applicability": {"gap": ""},
    }
    (catalog / "routing_cards.jsonl").write_text(json.dumps(card) + "\n", encoding="utf-8")
    (catalog / "task_families.json").write_text("{}\n", encoding="utf-8")


def test_skill_library_toolset_is_separate_from_profile_skills():
    from toolsets import _HERMES_CORE_TOOLS, resolve_toolset

    names = set(resolve_toolset("skill_library", include_registry=False))
    assert names == {"skill_library_find", "skill_library_load", "skill_library_visit"}
    assert names <= set(_HERMES_CORE_TOOLS)
    assert names.isdisjoint(resolve_toolset("skills", include_registry=False))


def test_dispatch_find_reads_the_catalog_root(monkeypatch, tmp_path):
    if package_root() is None:
        pytest.skip("skill-lib submodule is not checked out")
    _library(tmp_path)
    _clear_finder_cache()
    monkeypatch.setattr("agent.skill_library.catalog_root", lambda: tmp_path)
    discover_builtin_tools()
    raw = registry.dispatch("skill_library_find", {"task": "Run GSEA on a ranked gene list"})
    payload = json.loads(raw)
    assert payload["success"] is True
    assert payload["cards"][0]["id"] == "demo::gsea"
    _clear_finder_cache()


def test_dispatch_find_reports_a_missing_catalog(monkeypatch, tmp_path):
    monkeypatch.setattr("agent.skill_library.catalog_root", lambda: tmp_path)
    discover_builtin_tools()
    raw = registry.dispatch("skill_library_find", {"task": "rank genes"})
    payload = json.loads(raw)
    assert payload["success"] is False
    assert "catalog" in payload["error"]
