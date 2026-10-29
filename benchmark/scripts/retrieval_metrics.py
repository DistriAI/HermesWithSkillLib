"""Skill-retrieval metrics for one experiment.

The definitions follow SkillRouter's public scorer: Hit@1, Precision@3,
MRR@10, Recall@10/20/50, FullCoverage@3/5/10, and nDCG@1/3/10. On a
SkillRouter ranking, a retrieved id matches a gold id when the strings are
equal or their skill-name component is equal. Graded nDCG reads SkillRouter's
``relevance`` map when this task is in that file.

A SkillLib ranking is scored through ``skill_map`` instead. A retrieved id
hits a task skill only when the map lists that id for the skill. Metrics on
the full gold set count a skill SkillLib does not contain as a miss.
``FullCoverageCovered@k`` is 1 when every skill the map does link is inside
the top k, including the case where the map links nothing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

_METRIC_KEYS = (
    "nDCG@1", "nDCG@3", "nDCG@10",
    "Hit@1", "Precision@3", "MRR@10",
    "Recall@10", "Recall@20", "Recall@50",
    "FullCoverage@3", "FullCoverage@5", "FullCoverage@10",
)
_COVERED_KEYS = ("FullCoverageCovered@3", "FullCoverageCovered@5", "FullCoverageCovered@10")
_CATALOG_MEAN_KEYS = _METRIC_KEYS + _COVERED_KEYS + ("coverage",)
_FIND_TOOLS = ("skill_router_find", "skill_library_find")


def relevance_path() -> Path:
    return Path(__file__).resolve().parents[2] / "baselines" / "SkillRouter" / "data" / "eval_core" / "relevance.json"


def skill_name(skill_id: str) -> str:
    """Last path component, ignoring a trailing ``SKILL.md``."""
    parts = [part for part in str(skill_id).replace("\\", "/").split("/") if part]
    if parts and parts[-1].lower() == "skill.md":
        parts = parts[:-1]
    return parts[-1].lower() if parts else str(skill_id).lower()


def _matches(ranked_id: str, gold_id: str) -> bool:
    return ranked_id == gold_id or skill_name(ranked_id) == skill_name(gold_id)


def _grade(ranked_id: str, gold_ids: list[str], relevance: dict) -> float:
    if ranked_id in relevance:
        return float(relevance[ranked_id])
    name = skill_name(ranked_id)
    grades = [
        float(score) for gold_id, score in relevance.items()
        if skill_name(str(gold_id)) == name
    ]
    if grades:
        return max(grades)
    if any(_matches(ranked_id, gold_id) for gold_id in gold_ids):
        return 1.0
    return 0.0


def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(index + 2) for index, rel in enumerate(relevances[:k]))


def score(ranked_ids: list[str], gold_ids: list[str], relevance: dict | None = None) -> dict[str, float]:
    """One task, one ranked list. Empty gold scores as a miss on every hit metric except coverage."""
    gold = [str(item) for item in gold_ids]
    ranked = [str(item) for item in ranked_ids]
    grades = relevance or {}

    def hit(k: int) -> float:
        return 1.0 if any(any(_matches(ranked_id, gold_id) for gold_id in gold) for ranked_id in ranked[:k]) else 0.0

    def precision(k: int) -> float:
        if k <= 0:
            return 0.0
        hits = sum(1 for ranked_id in ranked[:k] if any(_matches(ranked_id, gold_id) for gold_id in gold))
        return hits / k

    def recall(k: int) -> float:
        if not gold:
            return 0.0
        found = sum(1 for gold_id in gold if any(_matches(ranked_id, gold_id) for ranked_id in ranked[:k]))
        return found / len(gold)

    def coverage(k: int) -> float:
        if not gold:
            return 1.0
        return 1.0 if all(any(_matches(ranked_id, gold_id) for ranked_id in ranked[:k]) for gold_id in gold) else 0.0

    mrr = 0.0
    for index, ranked_id in enumerate(ranked[:10]):
        if any(_matches(ranked_id, gold_id) for gold_id in gold):
            mrr = 1.0 / (index + 1)
            break
    relevances = [_grade(ranked_id, gold, grades) for ranked_id in ranked]
    ideal = [float(value) for value in grades.values()] or [1.0] * len(gold)

    def ndcg(k: int) -> float:
        ceiling = _dcg(sorted(ideal, reverse=True), k)
        return _dcg(relevances, k) / ceiling if ceiling > 0 else 0.0

    return {
        "nDCG@1": ndcg(1),
        "nDCG@3": ndcg(3),
        "nDCG@10": ndcg(10),
        "Hit@1": hit(1),
        "Precision@3": precision(3),
        "MRR@10": mrr,
        "Recall@10": recall(10),
        "Recall@20": recall(20),
        "Recall@50": recall(50),
        "FullCoverage@3": coverage(3),
        "FullCoverage@5": coverage(5),
        "FullCoverage@10": coverage(10),
    }


def _links(skills: list[dict]) -> dict[str, set[str]]:
    linked: dict[str, set[str]] = {}
    for item in skills:
        gold_id = str(item.get("gold_id") or "")
        if not gold_id:
            continue
        for ident in item.get("skilllib_ids") or []:
            linked.setdefault(str(ident), set()).add(gold_id)
    return linked


def _found(ranked: list[str], linked: dict[str, set[str]], targets: set[str], k: int) -> set[str]:
    hits: set[str] = set()
    for ident in ranked[:k]:
        hits.update(linked.get(str(ident), set()) & targets)
    return hits


def _binary_ndcg(ranked: list[str], linked: dict[str, set[str]], targets: set[str], k: int) -> float:
    """nDCG with one relevant document per gold skill. One id contributes at most 1."""
    if not targets:
        return 0.0
    seen: set[str] = set()
    gains: list[float] = []
    for ident in ranked:
        newly = (linked.get(str(ident), set()) & targets) - seen
        seen.update(newly)
        gains.append(1.0 if newly else 0.0)
    ceiling = _dcg([1.0] * len(targets), k)
    return _dcg(gains, k) / ceiling if ceiling else 0.0


def _skillrouter_keys(ranked: list[str], linked: dict[str, set[str]], targets: list[str]) -> dict[str, float]:
    """SkillRouter's metric set. Membership is the map, not the id string."""
    wanted = set(targets)

    def hit(k: int) -> float:
        if not wanted:
            return 0.0
        return 1.0 if _found(ranked, linked, wanted, k) else 0.0

    def precision(k: int) -> float:
        if k <= 0:
            return 0.0
        hits = sum(1 for ident in ranked[:k] if linked.get(str(ident), set()) & wanted)
        return hits / k

    def recall(k: int) -> float:
        if not wanted:
            return 0.0
        return len(_found(ranked, linked, wanted, k)) / len(wanted)

    def full(k: int) -> float:
        if not wanted:
            return 1.0
        return 1.0 if _found(ranked, linked, wanted, k) == wanted else 0.0

    mrr = 0.0
    for index, ident in enumerate(ranked[:10]):
        if linked.get(str(ident), set()) & wanted:
            mrr = 1.0 / (index + 1)
            break
    return {
        "nDCG@1": _binary_ndcg(ranked, linked, wanted, 1),
        "nDCG@3": _binary_ndcg(ranked, linked, wanted, 3),
        "nDCG@10": _binary_ndcg(ranked, linked, wanted, 10),
        "Hit@1": hit(1),
        "Precision@3": precision(3),
        "MRR@10": mrr,
        "Recall@10": recall(10),
        "Recall@20": recall(20),
        "Recall@50": recall(50),
        "FullCoverage@3": full(3),
        "FullCoverage@5": full(5),
        "FullCoverage@10": full(10),
    }


def catalog_score(ranked_ids: list[str], skills: list[dict]) -> dict:
    """Score one SkillLib ranking against one task's mapped skills.

    The SkillRouter keys use every skill shipped with the task. A skill the
    map does not link can never be hit, so Recall and FullCoverage on that
    set are capped by catalog coverage. ``FullCoverageCovered@k`` uses only
    the linked skills and is 1 when that set is empty. ``on_covered_gold``
    repeats the SkillRouter keys on the linked skills alone, and is omitted
    when the map links nothing.
    """
    ranked = [str(item) for item in ranked_ids]
    rows = [item for item in skills if isinstance(item, dict)]
    linked = _links(rows)
    gold = [str(item["gold_id"]) for item in rows if item.get("gold_id")]
    covered = [str(item["gold_id"]) for item in rows if item.get("covered") and item.get("gold_id")]
    metrics = _skillrouter_keys(ranked, linked, gold)
    covered_targets = set(covered)
    for cutoff in (3, 5, 10):
        if not covered_targets:
            metrics[f"FullCoverageCovered@{cutoff}"] = 1.0
        else:
            found = _found(ranked, linked, covered_targets, cutoff)
            metrics[f"FullCoverageCovered@{cutoff}"] = 1.0 if found == covered_targets else 0.0
    metrics["coverage"] = (len(covered) / len(gold)) if gold else 0.0
    metrics["n_gold"] = len(gold)
    metrics["n_covered"] = len(covered)
    metrics["on_covered_gold"] = _skillrouter_keys(ranked, linked, covered) if covered else None
    return metrics


def evaluate_library(task_id: str, ranked_ids: list[str], skill_map: dict | None = None) -> dict | None:
    """Score ``ranked_ids`` for ``task_id`` using the SkillsBench–SkillLib map."""
    import skill_map as skill_map_mod

    document = skill_map if skill_map is not None else skill_map_mod.load()
    entry = skill_map_mod.task_entry(document, task_id)
    if entry is None:
        return None
    return catalog_score(ranked_ids, entry.get("skills") or [])


def _skill_dir_names(replica: Path | None) -> list[str]:
    if replica is None:
        return []
    skills = replica / "environment" / "skills"
    if not skills.is_dir():
        return []
    return sorted(path.name for path in skills.iterdir() if path.is_dir())


def _load_relevance(path: Path | None) -> dict:
    chosen = relevance_path() if path is None else path
    if not chosen.is_file():
        return {}
    try:
        raw = json.loads(chosen.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def gold_block(task_id: str, replica: Path | None = None, relevance_file: Path | None = None) -> dict:
    """Gold skills for ``task_id``.

    SkillRouter's ``core_gt_ids`` win when this task is in ``relevance.json``
    and is not ``generic_only``. Otherwise the gold set is the task's
    ``environment/skills`` directory names. A task with no gold is not scored.
    """
    entry = _load_relevance(relevance_file).get(task_id) or {}
    if not isinstance(entry, dict):
        entry = {}
    task_type = str(entry.get("task_type") or "")
    catalog = entry.get("core_gt_ids") or entry.get("gt_skill_ids") or []
    if task_type == "generic_only":
        gold: list[str] = []
    elif catalog:
        gold = [str(item) for item in catalog]
    else:
        gold = _skill_dir_names(replica)
    relevance = entry.get("relevance") if isinstance(entry.get("relevance"), dict) else {}
    return {
        "task_id": task_id,
        "scored": bool(gold),
        "task_type": task_type,
        "gold_ids": gold,
        "relevance": {str(key): float(value) for key, value in relevance.items()},
        "ranked_ids": [],
        "retrieval_ids": [],
        "metrics": None,
        "candidate_metrics": None,
    }


def _payload(content) -> dict:
    if isinstance(content, dict):
        return content
    if isinstance(content, str) and content.strip():
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _call_name(call: dict) -> str:
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    return str(function.get("name") or call.get("name") or "")


def ranked_ids_from_lines(lines: list[dict]) -> tuple[list[str], list[str], str]:
    """Ranked ids from the first find result in ``lines``.

    The second list is the cosine candidate ids when the tool returned them.
    The third value is the tool name, empty when no find result is present.
    """
    pending: dict[str, str] = {}
    for line in lines:
        if not isinstance(line, dict) or line.get("type") == "turn":
            continue
        role = line.get("role")
        if role == "assistant":
            for call in line.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                name = _call_name(call)
                if name not in _FIND_TOOLS:
                    continue
                call_id = str(call.get("id") or call.get("call_id") or "")
                if call_id:
                    pending[call_id] = name
        elif role == "tool":
            name = pending.pop(str(line.get("tool_call_id") or ""), "")
            if name not in _FIND_TOOLS:
                continue
            payload = _payload(line.get("content"))
            ranked = [str(item) for item in payload.get("ranked_ids") or []]
            if not ranked and name == "skill_library_find":
                for card in payload.get("cards") or []:
                    if isinstance(card, dict):
                        ident = card.get("id") or card.get("skill_id") or card.get("name")
                        if ident:
                            ranked.append(str(ident))
            if ranked:
                retrieved = [str(item) for item in payload.get("retrieval_ids") or []]
                return ranked, retrieved, name
    return [], [], ""


def apply_ranking(
    block: dict,
    ranked_ids: list[str],
    retrieval_ids: list[str] | None = None,
    tool: str = "",
) -> dict:
    """Score ``block`` from the first non-empty ranking. Later finds stay out.

    ``skill_library_find`` also records ``catalog_metrics`` from the skill map,
    including tasks SkillRouter leaves unscored.
    """
    if not ranked_ids:
        return block
    if block.get("scored") and not block.get("metrics"):
        block["ranked_ids"] = list(ranked_ids)
        block["retrieval_ids"] = list(retrieval_ids or [])
        relevance = block.get("relevance") or None
        block["metrics"] = score(ranked_ids, block.get("gold_ids") or [], relevance)
        if retrieval_ids:
            block["candidate_metrics"] = score(retrieval_ids, block.get("gold_ids") or [], relevance)
    if tool == "skill_library_find" and not isinstance(block.get("catalog_metrics"), dict):
        block["ranked_ids"] = list(block.get("ranked_ids") or ranked_ids)
        block["catalog_metrics"] = evaluate_library(str(block.get("task_id") or ""), ranked_ids)
    return block


def observe(block: dict, lines: list[dict]) -> dict:
    ranked, retrieved, tool = ranked_ids_from_lines(lines)
    return apply_ranking(block, ranked, retrieved, tool)


def attach(records: list[dict], lines_by_task: dict[str, list[dict]]) -> None:
    """Fill gold and, when the transcript has a find result, the metrics."""
    for record in records:
        task_id = str(record.get("id") or "")
        block = record.get("skill_retrieval")
        if not isinstance(block, dict) or "gold_ids" not in block:
            replica = Path(record["task_replica"]) if record.get("task_replica") else None
            block = gold_block(task_id, replica)
            record["skill_retrieval"] = block
        if block.get("metrics"):
            continue
        observe(block, lines_by_task.get(task_id) or [])


def _mean(rows: list[dict], keys: tuple[str, ...] = _METRIC_KEYS) -> dict[str, float | None]:
    if not rows:
        return {key: None for key in keys}
    return {key: sum(float(row.get(key) or 0) for row in rows) / len(rows) for key in keys}


def summarize_catalog(rows: list[dict]) -> dict:
    """Macro-average ``catalog_score`` rows.

    ``pipeline`` scores every skill shipped with the task. ``on_covered_gold``
    scores the skills the SkillLib map links, and leaves out tasks with none.
    ``single`` and ``multi`` follow the number of shipped skills.
    """

    def covered(group: list[dict]) -> list[dict]:
        return [row["on_covered_gold"] for row in group if isinstance(row.get("on_covered_gold"), dict)]

    single = [row for row in rows if row.get("n_gold") == 1]
    multi = [row for row in rows if (row.get("n_gold") or 0) > 1]
    groups = {"all": rows, "single": single, "multi": multi}
    return {
        "n_tasks": len(rows),
        "n_with_covered_gold": len(covered(rows)),
        "coverage": _mean(rows, ("coverage",)).get("coverage"),
        "pipeline": {
            name: _mean(group, _CATALOG_MEAN_KEYS) | {"n": len(group)}
            for name, group in groups.items()
        },
        "on_covered_gold": {
            name: _mean(covered(group)) | {"n": len(covered(group))}
            for name, group in groups.items()
        },
    }


def summary(records: list[dict]) -> dict:
    """Macro-average retrieval metrics over tasks whose find result was scored.

    Tasks with no gold, and tasks that never called find, stay out of the mean.
    ``single`` and ``multi`` follow SkillRouter: one gold skill, or more than one.
    """
    grouped = {"all": [], "single": [], "multi": []}
    candidates = []
    with_gold = 0
    for record in records:
        block = record.get("skill_retrieval") if isinstance(record.get("skill_retrieval"), dict) else {}
        if block.get("scored"):
            with_gold += 1
        metrics = block.get("metrics")
        if not isinstance(metrics, dict):
            continue
        grouped["all"].append(metrics)
        gold = block.get("gold_ids") or []
        if len(gold) == 1:
            grouped["single"].append(metrics)
        elif len(gold) > 1:
            grouped["multi"].append(metrics)
        candidate = block.get("candidate_metrics")
        if isinstance(candidate, dict):
            candidates.append(candidate)
    catalog_rows = []
    covered_rows = []
    for record in records:
        block = record.get("skill_retrieval") if isinstance(record.get("skill_retrieval"), dict) else {}
        catalog = block.get("catalog_metrics")
        if not isinstance(catalog, dict):
            continue
        catalog_rows.append(catalog)
        nested = catalog.get("on_covered_gold")
        if isinstance(nested, dict):
            covered_rows.append(nested)
    return {
        "n_with_gold": with_gold,
        "n_ranked": len(grouped["all"]),
        "pipeline": {name: _mean(rows) | {"n": len(rows)} for name, rows in grouped.items()},
        "candidates": _mean(candidates) | {"n": len(candidates)},
        "catalog": {
            "n_ranked": len(catalog_rows),
            "n_with_covered_gold": len(covered_rows),
            "coverage": _mean(catalog_rows, ("coverage",)).get("coverage"),
            "pipeline": _mean(catalog_rows, _CATALOG_MEAN_KEYS) | {"n": len(catalog_rows)},
            "on_covered_gold": _mean(covered_rows) | {"n": len(covered_rows)},
        },
    }
