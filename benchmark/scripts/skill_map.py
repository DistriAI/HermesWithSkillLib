"""Map each SkillsBench task skill onto the SkillLib cards that are that skill.

A retrieved SkillLib id counts as a hit only when it is listed on the task
skill. Byte identity is decisive. When the bytes differ, a card with the same
name counts only if its text contains the task skill (token containment) or
the two texts are nearly the same set of tokens (Jaccard). Same-name skills
below those cuts stay unmapped, and the closest candidate is recorded so the
decision can be checked.

Rebuild with ``python benchmark/scripts/skill_map.py`` after the catalog or
the SkillsBench tasks change. Scoring reads the written map and does not walk
the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

CONTAINMENT_MIN = 0.95
JACCARD_MIN = 0.75
MAP_PATH = Path(__file__).resolve().parent / "data" / "skillsbench_skilllib_map.json"

_TOKEN = re.compile(r"[a-z0-9]+")
_FRONT_NAME = re.compile(r"(?m)^name:\s*[\"']?([^\"'\n]+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _skill_name(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part and part != "::"]
    # Occurrence ids look like ``owner/repo::skills/name/SKILL.md``.
    flattened: list[str] = []
    for part in parts:
        flattened.extend(piece for piece in part.split("::") if piece)
    if flattened and flattened[-1].lower() == "skill.md":
        flattened = flattened[:-1]
    return flattened[-1].lower() if flattened else ""


def _front_name(text: str) -> str:
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    head = text[3:end] if end > 0 else text[:800]
    match = _FRONT_NAME.search(head)
    if not match:
        return ""
    return match.group(1).strip().strip("\"'").lower()


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_roots(repo: Path) -> list[Path]:
    roots: list[Path] = []
    config = repo / "skill-lib" / "skilllib.json"
    if config.is_file():
        try:
            raw = json.loads(config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        cache = raw.get("cache") if isinstance(raw, dict) else None
        if cache:
            roots.append(Path(cache))
    roots.append(repo / "skill-lib" / "cache")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _locate(rel: str, roots: list[Path], repo: Path) -> Path | None:
    if not rel:
        return None
    candidates = [repo / "skill-lib" / rel]
    suffix = rel[len("cache/"):] if rel.startswith("cache/") else rel
    for root in roots:
        candidates.append(root / suffix)
        candidates.append(root / rel)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _card_names(card: dict) -> set[str]:
    names: set[str] = set()

    def add(value: object) -> None:
        raw = str(value or "").strip()
        if not raw:
            return
        text = _skill_name(raw) if "/" in raw or "::" in raw else raw.lower()
        if text:
            names.add(text)

    add(card.get("name"))
    for alias in card.get("aliases") or []:
        add(alias)
    provenance = card.get("provenance") if isinstance(card.get("provenance"), dict) else {}
    add(_skill_name(str(provenance.get("path") or "")))
    for source in card.get("sources") or []:
        if not isinstance(source, dict):
            continue
        add(_skill_name(str(source.get("path") or "")))
        add(_skill_name(str(source.get("occurrence_id") or "")))
    names.discard("")
    return names


def _source_rows(card: dict) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    body = card.get("body") if isinstance(card.get("body"), dict) else {}
    candidates = []
    if body.get("cache_path"):
        candidates.append({"cache_path": body.get("cache_path"), "occurrence_id": card.get("id")})
    for source in card.get("sources") or []:
        if isinstance(source, dict) and source.get("cache_path"):
            candidates.append(source)
    for source in candidates:
        rel = str(source.get("cache_path") or "")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        occurrence = str(source.get("occurrence_id") or card.get("id") or "")
        rows.append({"cache_path": rel, "occurrence_id": occurrence, "card_id": str(card.get("id") or "")})
    return rows


def _index_catalog(repo: Path) -> tuple[dict[str, list[dict]], dict[str, list[dict]], dict[str, dict]]:
    """Hash index, same-name index, and card-id index for files that are on disk."""
    roots = _cache_roots(repo)
    cards_path = repo / "skill-lib" / "catalog" / "routing_cards.jsonl"
    by_hash: dict[str, list[dict]] = {}
    by_name: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    with cards_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            names = _card_names(card)
            sources = _source_rows(card)
            entry = {"id": str(card.get("id") or ""), "names": names, "sources": sources}
            if entry["id"]:
                by_id[entry["id"]] = entry
            for name in names:
                by_name.setdefault(name, []).append(entry)
            for source in sources:
                path = _locate(source["cache_path"], roots, repo)
                if path is None:
                    continue
                source["path"] = path
                source["sha256"] = _sha256(path)
                by_hash.setdefault(source["sha256"], []).append(source)
    return by_hash, by_name, by_id


def _task_skills(repo: Path) -> list[tuple[str, str, Path]]:
    root = repo / "benchmark" / "skillsbench" / "tasks"
    rows: list[tuple[str, str, Path]] = []
    if not root.is_dir():
        return rows
    for task in sorted(path for path in root.iterdir() if path.is_dir()):
        skills = task / "environment" / "skills"
        if not skills.is_dir():
            continue
        for skill in sorted(path for path in skills.iterdir() if path.is_dir()):
            document = skill / "SKILL.md"
            if document.is_file():
                rows.append((task.name, skill.name, document))
    return rows


def _overlap(bench_tokens: set[str], lib_tokens: set[str]) -> tuple[float, float]:
    union = bench_tokens | lib_tokens
    overlap = len(bench_tokens & lib_tokens)
    jaccard = overlap / len(union) if union else 0.0
    containment = overlap / len(bench_tokens) if bench_tokens else 0.0
    return jaccard, containment


def _prepare_sources(text: str, cards: list[dict]) -> list[tuple[dict, float, float]]:
    """Token overlap of every on-disk source. The strongest Jaccard is last."""
    bench_tokens = _tokens(text)
    scored: list[tuple[dict, float, float]] = []
    for card in cards:
        for source in card["sources"]:
            path = source.get("path")
            if path is None:
                continue
            lib_tokens = source.get("tokens")
            if lib_tokens is None:
                lib_tokens = _tokens(path.read_text(encoding="utf-8", errors="replace"))
                source["tokens"] = lib_tokens
            jaccard, containment = _overlap(bench_tokens, lib_tokens)
            source["jaccard"] = jaccard
            source["containment"] = containment
            scored.append((source, jaccard, containment))
    return scored


def _accepted(source: dict) -> bool:
    return source.get("containment", 0.0) >= CONTAINMENT_MIN or source.get("jaccard", 0.0) >= JACCARD_MIN


def _accept_ids(cards: list[dict], predicate) -> list[str]:
    accepted: set[str] = set()
    for card in cards:
        matched_sources = [source for source in card["sources"] if predicate(source)]
        if not matched_sources:
            continue
        if card["id"]:
            accepted.add(card["id"])
        for source in matched_sources:
            if source.get("occurrence_id"):
                accepted.add(str(source["occurrence_id"]))
    return sorted(accepted)


def _match_skill(task: str, skill: str, document: Path, by_hash: dict, by_name: dict, by_id: dict) -> dict:
    text = document.read_text(encoding="utf-8", errors="replace")
    digest = hashlib.sha256(document.read_bytes()).hexdigest()
    names = {skill.lower(), _front_name(text)}
    names.discard("")
    same_name: list[dict] = []
    seen_cards: set[str] = set()
    for name in names:
        for card in by_name.get(name, []):
            if card["id"] in seen_cards:
                continue
            seen_cards.add(card["id"])
            same_name.append(card)
    hash_hits = by_hash.get(digest, [])
    gold_id = f"{task}/{skill}"
    if hash_hits:
        cards = [by_id[hit["card_id"]] for hit in hash_hits if hit.get("card_id") in by_id]
        ids = _accept_ids(cards, lambda source: source.get("sha256") == digest)
        return _entry(gold_id, skill, digest, True, "sha256", 1.0, ids, None)
    scored = _prepare_sources(text, same_name)
    passing = [item for item in scored if _accepted(item[0])]
    if passing:
        # Prefer a source that contains the task skill. Otherwise take the closest token set.
        chosen = max(passing, key=lambda item: (item[2] >= CONTAINMENT_MIN, item[1]))
        _source, jaccard, containment = chosen
        method = "contained" if containment >= CONTAINMENT_MIN else "near_copy"
        score = 1.0 if method == "contained" else jaccard
        ids = _accept_ids(same_name, _accepted)
        return _entry(gold_id, skill, digest, True, method, round(score, 4), ids, None)
    candidate = None
    if scored:
        source, jaccard, containment = max(scored, key=lambda item: item[1])
        candidate = {
            "skilllib_id": source.get("card_id") or source.get("occurrence_id") or "",
            "jaccard": round(jaccard, 4),
            "containment": round(containment, 4),
        }
    return _entry(gold_id, skill, digest, False, "", 0.0, [], candidate)


def _entry(gold_id, skill, digest, covered, method, score, ids, candidate) -> dict:
    row = {
        "skill": skill,
        "gold_id": gold_id,
        "sha256": digest,
        "covered": covered,
        "method": method,
        "score": score,
        "skilllib_ids": ids,
    }
    if candidate is not None:
        row["best_candidate"] = candidate
    return row


def build(repo: Path | None = None) -> dict:
    """Match every SkillsBench ``environment/skills/*/SKILL.md`` to SkillLib."""
    root = repo or repo_root()
    by_hash, by_name, by_id = _index_catalog(root)
    tasks: dict[str, dict] = {}
    for task, skill, document in _task_skills(root):
        bucket = tasks.setdefault(task, {"skills": []})
        bucket["skills"].append(_match_skill(task, skill, document, by_hash, by_name, by_id))
    for bucket in tasks.values():
        skills = bucket["skills"]
        covered = [item for item in skills if item["covered"]]
        bucket["n_gold"] = len(skills)
        bucket["n_covered"] = len(covered)
        bucket["coverage"] = round(len(covered) / len(skills), 4) if skills else 0.0
    n_skills = sum(bucket["n_gold"] for bucket in tasks.values())
    n_covered = sum(bucket["n_covered"] for bucket in tasks.values())
    return {
        "schema_version": 1,
        "benchmark": "skillsbench",
        "containment_min": CONTAINMENT_MIN,
        "jaccard_min": JACCARD_MIN,
        "n_tasks": len(tasks),
        "n_skills": n_skills,
        "n_covered_skills": n_covered,
        "coverage": round(n_covered / n_skills, 4) if n_skills else 0.0,
        "tasks": tasks,
    }


_LOADED: dict[str, dict] = {}


def load(path: Path | None = None) -> dict:
    chosen = MAP_PATH if path is None else path
    key = str(chosen)
    cached = _LOADED.get(key)
    if cached is not None:
        return cached
    raw = json.loads(chosen.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "tasks" not in raw:
        raise ValueError(f"skill map at {chosen} has no tasks")
    _LOADED[key] = raw
    return raw


def task_entry(skill_map: dict, task_id: str) -> dict | None:
    tasks = skill_map.get("tasks") if isinstance(skill_map, dict) else None
    entry = tasks.get(task_id) if isinstance(tasks, dict) else None
    return entry if isinstance(entry, dict) else None


def write(path: Path | None = None, repo: Path | None = None) -> dict:
    document = build(repo)
    _LOADED.clear()
    chosen = MAP_PATH if path is None else path
    chosen.parent.mkdir(parents=True, exist_ok=True)
    chosen.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return document


def _report(document: dict) -> str:
    uncovered = []
    full = partial = none = 0
    for task_id, entry in sorted((document.get("tasks") or {}).items()):
        skills = entry.get("skills") or []
        missed = [item["skill"] for item in skills if not item.get("covered")]
        if not missed:
            full += 1
        elif len(missed) == len(skills):
            none += 1
        else:
            partial += 1
        for item in skills:
            if item.get("covered"):
                continue
            candidate = item.get("best_candidate") or {}
            uncovered.append(
                f"{task_id}/{item['skill']} nearest={candidate.get('skilllib_id', '')} "
                f"jaccard={candidate.get('jaccard', 0)} containment={candidate.get('containment', 0)}"
            )
    lines = [
        f"tasks={document.get('n_tasks')} skills={document.get('n_skills')} "
        f"covered_skills={document.get('n_covered_skills')} coverage={document.get('coverage')}",
        f"tasks_full={full} tasks_partial={partial} tasks_uncovered={none}",
        "uncovered:",
        *uncovered,
    ]
    return "\n".join(lines)


def main() -> None:
    document = write()
    print(_report(document))
    print(f"wrote {MAP_PATH}")


if __name__ == "__main__":
    sys.exit(main())
