"""Contracts for the skill-lib adapter: prompt gating and catalog lookup."""

import json
from pathlib import Path

import pytest

from agent.skill_library import (
    _LIBRARY_TOOLS,
    _clear_finder_cache,
    find_in_library,
    library_available,
    library_prompt,
    load_from_library,
    package_root,
    visit_in_library,
)


def _library(tmp_path: Path) -> Path:
    skill = tmp_path / "cache" / "skills" / "demo" / "gsea" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: gsea\n---\n# GSEA\nRun the enrichment procedure.\n",
        encoding="utf-8",
    )
    card = {
        "id": "demo::gsea",
        "name": "gsea",
        "aliases": ["gsea"],
        "kind": "authentic",
        "description": "Run GSEA on a ranked gene list.",
        "positive_scope": ["ranked gene list"],
        "negative_scope": [],
        "inputs": [],
        "outputs": [],
        "domains": ["science"],
        "preconditions": [],
        "provenance": {"conflict_note": "Prefer the ranked-list method."},
        "body": {"scripts_checked_out": False},
        "sources": [{
            "repository": "demo/gsea",
            "cache_path": "cache/skills/demo/gsea/SKILL.md",
            "present": True,
        }],
        "links": {"requires": [], "alternative": [], "follows": []},
        "applicability": {"gap": ""},
    }
    gap_card = {
        **card,
        "id": "demo::pdf",
        "name": "pdf",
        "description": "Work with PDF files.",
        "positive_scope": ["pdf"],
        "sources": [],
        "provenance": {"conflict_note": ""},
        "applicability": {"gap": "Run visit for demo/pdf before using bundled scripts."},
    }
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    lines = [json.dumps(card), json.dumps(gap_card)]
    (catalog / "routing_cards.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (catalog / "task_families.json").write_text("{}\n", encoding="utf-8")
    return tmp_path


def test_prompt_is_omitted_unless_the_session_offers_the_library_tools():
    assert library_prompt(None) == ""
    assert library_prompt({"skill_view", "skills_list"}) == ""
    assert library_prompt({"skill_library_find"}) == ""


def test_find_requires_a_task_and_a_catalog(tmp_path):
    assert find_in_library("   ", root=tmp_path)["success"] is False
    assert find_in_library("rank genes", root=tmp_path)["error"].startswith("Skill library catalog")


def test_find_rejects_a_constraints_mapping(tmp_path):
    _library(tmp_path)
    result = find_in_library("rank genes", root=tmp_path, constraints={"python": True})  # type: ignore[arg-type]
    assert result["success"] is False
    assert "constraints" in result["error"]


@pytest.fixture
def library_tree(tmp_path):
    _clear_finder_cache()
    if package_root() is None:
        pytest.skip("skill-lib submodule is not checked out")
    root = _library(tmp_path)
    yield root
    _clear_finder_cache()


def test_find_and_load_round_trip(library_tree):
    found = find_in_library("Run GSEA on a ranked gene list", root=library_tree)
    assert found["success"] is True
    cards = found["cards"]
    assert cards and cards[0]["name"] == "gsea"
    assert cards[0]["id"] == "demo::gsea"
    loaded = load_from_library(cards[0]["id"], root=library_tree)
    assert loaded["success"] is True
    assert "enrichment procedure" in loaded["text"]
    assert "Prefer the ranked-list method." in loaded["text"]


def test_load_reports_a_checkout_gap_instead_of_a_procedure(library_tree):
    loaded = load_from_library("demo::pdf", root=library_tree)
    assert loaded["success"] is True
    assert loaded["text"] == ""
    assert "demo/pdf" in loaded["gap"]
    assert "skill_library_visit" in loaded["next"]


def test_load_unknown_id(library_tree):
    loaded = load_from_library("missing::skill", root=library_tree)
    assert loaded["success"] is False
    assert "Unknown skill id" in loaded["error"]


def test_visit_rejects_a_malformed_repository_id(tmp_path):
    if package_root() is None:
        pytest.skip("skill-lib submodule is not checked out")
    result = visit_in_library("not a repository", root=tmp_path)
    assert result["success"] is False


def test_prompt_uses_the_library_domain_sentences():
    if package_root() is None or not library_available():
        pytest.skip("skill-lib catalog is not checked out")
    from agent.skill_library import _prompt_constants
    text = library_prompt(set(_LIBRARY_TOOLS))
    sentences, load_cap = _prompt_constants()
    for name, sentence in sentences:
        assert f"- {name}: {sentence}" in text
    assert f"at most {load_cap}" in text
    assert "skill_library_find" in text
    assert "skill_library_visit" in text


def test_skills_system_prompt_appends_the_fragment_only_when_tools_are_offered(monkeypatch, tmp_path):
    from agent.prompt_builder import build_skills_system_prompt, clear_skills_system_prompt_cache

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    clear_skills_system_prompt_cache(clear_snapshot=True)
    try:
        withheld = build_skills_system_prompt(available_tools={"skill_view", "skills_list"})
        offered = build_skills_system_prompt(available_tools=set(_LIBRARY_TOOLS))
    finally:
        clear_skills_system_prompt_cache(clear_snapshot=True)
    assert "## Skill library" not in withheld
    if library_available():
        assert "## Skill library" in offered
        assert offered.index("## Skill library") == offered.rfind("## Skill library")
    else:
        assert "## Skill library" not in offered
