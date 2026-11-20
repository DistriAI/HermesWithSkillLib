"""SkillCorpus and SkillNet keep their own stores and their own search."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from agent.skill_corpus import call_tool as corpus_call
from agent.skill_library import catalog_root, package_root
from agent.skill_net import call_tool as skillnet_call

REPO = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


corpus = _load(REPO / "baselines" / "SkillCorpus" / "hermes_build.py", "skillcorpus_hermes_build")
skillnet = _load(REPO / "baselines" / "SkillNet" / "hermes_build.py", "skillnet_hermes_build")


def test_publish_refuses_a_partial_library(tmp_path: Path):
    with pytest.raises(SystemExit, match="partial library"):
        corpus.build(tmp_path, limit=2, publish=True)
    with pytest.raises(SystemExit, match="partial index"):
        skillnet.build(tmp_path, limit=2, publish=True)


def test_skill_library_catalog_stays_the_skilllib_checkout(monkeypatch):
    monkeypatch.setenv("SKILL_LIBRARY_VARIANT", "skillcorpus")
    assert catalog_root() == package_root()


def test_skill_corpus_search_uses_its_own_engine(tmp_path: Path, monkeypatch):
    if not (REPO / "baselines" / "SkillCorpus" / "skillcorpus" / "__init__.py").is_file():
        pytest.skip("SkillCorpus checkout is missing")
    skill = tmp_path / "skills" / "demo"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text(
        "---\nname: gsea\ndescription: Rank genes with enrichment\n---\nRun the enrichment.\n",
        encoding="utf-8",
    )
    (tmp_path / "index.db").write_text("", encoding="utf-8")
    monkeypatch.setattr("agent.skill_corpus._LIBRARY", tmp_path)
    found = corpus_call("skill_corpus_find", {"task": "rank genes with enrichment"})
    assert found["success"] is True
    assert found["skills"][0]["name"] == "gsea"
    loaded = corpus_call("skill_corpus_load", {"path": found["skills"][0]["path"]})
    assert "Run the enrichment." in loaded["text"]
    blocked = corpus_call("skill_corpus_load", {"path": str(tmp_path / "index.db")})
    assert blocked["success"] is False


def test_skill_net_load_reads_only_its_skill_tree(tmp_path: Path, monkeypatch):
    if not (REPO / "baselines" / "SkillNet" / "skillnet-ai" / "src" / "skillnet_ai" / "__init__.py").is_file():
        pytest.skip("SkillNet checkout is missing")
    index = tmp_path / "index"
    index.mkdir()
    (index / "CURRENT").write_text("snapshot\n", encoding="utf-8")
    skill = tmp_path / "skills" / "s000000"
    skill.mkdir(parents=True)
    skill.joinpath("SKILL.md").write_text("Follow the graph.\n", encoding="utf-8")
    monkeypatch.setattr("agent.skill_net._INDEX", index)
    monkeypatch.setattr("agent.skill_net._SKILLS", tmp_path / "skills")
    loaded = skillnet_call("skill_net_load", {"path": str(skill)})
    assert loaded["success"] is True
    assert "Follow the graph." in loaded["text"]
    blocked = skillnet_call("skill_net_load", {"path": str(index / "CURRENT")})
    assert blocked["success"] is False


def test_runner_requires_each_systems_own_store(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "benchmark" / "scripts"))
    import common

    monkeypatch.setattr(common, "repo_root", lambda: REPO)
    with pytest.raises(SystemExit, match="hermes_build.py --publish"):
        common.require_native_library("corpus-only")
    with pytest.raises(SystemExit, match="hermes_index"):
        common.require_native_library("skillnet-only")
    common.require_native_library("library-only")
