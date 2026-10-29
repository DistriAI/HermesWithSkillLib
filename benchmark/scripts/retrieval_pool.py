"""SkillLib documents and SkillsBench queries for retrieval baselines.

Paper baselines score the all-field string ``name | description | body``, with
the description capped at 300 characters and the body at 2,500. The query is
the task statement in ``task.md``, after the frontmatter.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from pathlib import Path

import skill_map

DESC_MAX = 300
BODY_MAX = 2500
QUERY_MAX = 1500
_TOKEN = re.compile(r"[a-z0-9]+")


def repo_root() -> Path:
    return skill_map.repo_root()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def instruction_text(raw: str) -> str:
    """Task statement. The YAML frontmatter is setup metadata, not the query."""
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end > 0:
            raw = raw[end + 4 :]
    return raw.strip()


def all_field_text(name: str, description: str, body: str) -> str:
    return f"{name} | {description[:DESC_MAX]} | {body[:BODY_MAX]}"


def load_queries(skill_map_doc: dict, repo: Path | None = None, start: int | None = None, end: int | None = None) -> list[dict]:
    """One query per mapped SkillsBench task, in task-id order."""
    root = (repo or repo_root()) / "benchmark" / "skillsbench" / "tasks"
    task_ids = sorted((skill_map_doc.get("tasks") or {}))
    if start is not None or end is not None:
        lo = 0 if start is None else start
        hi = len(task_ids) - 1 if end is None else end
        task_ids = task_ids[lo : hi + 1]
    queries = []
    for task_id in task_ids:
        path = root / task_id / "task.md"
        if not path.is_file():
            continue
        text = instruction_text(path.read_text(encoding="utf-8", errors="replace"))
        if text:
            queries.append({"task_id": task_id, "text": text})
    return queries


def _body_rel(card: dict) -> str:
    sources = [item for item in (card.get("sources") or []) if isinstance(item, dict)]
    leaders = [item for item in sources if item.get("role") == "leader" and item.get("cache_path")]
    for source in leaders or sources:
        if source.get("cache_path"):
            return str(source["cache_path"])
    body = card.get("body") if isinstance(card.get("body"), dict) else {}
    return str(body.get("cache_path") or "")


def _read_capped(path: Path, limit: int = 12000) -> str:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def load_documents(repo: Path | None = None, cache: Path | None = None) -> list[dict]:
    """One document per routing card. ``text`` is the all-field string the paper encodes."""
    root = repo or repo_root()
    cards_path = root / "skill-lib" / "catalog" / "routing_cards.jsonl"
    if cache is not None and cache.is_file() and cards_path.is_file() and cache.stat().st_mtime >= cards_path.stat().st_mtime:
        return [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    roots = skill_map._cache_roots(root)
    documents = []
    with cards_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            card = json.loads(line)
            name = str(card.get("name") or skill_map._skill_name(str(card.get("id") or "")))
            description = str(card.get("description") or "").replace("\n", " ").strip()
            rel = _body_rel(card)
            path = skill_map._locate(rel, roots, root) if rel else None
            body = _read_capped(path).replace("\r\n", "\n") if path is not None else ""
            documents.append({
                "id": str(card.get("id") or ""),
                "name": name,
                "description": description[:500],
                "body": body[:BODY_MAX],
                "text": all_field_text(name, description, body),
            })
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in documents), encoding="utf-8")
        temporary.replace(cache)
    return documents


def document_stamp(documents: list[dict], model: str, pool: str, passage_prefix: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"{model}\n{pool}\n{passage_prefix}\n".encode())
    for doc in documents:
        digest.update(doc["id"].encode())
        digest.update(b"\t")
        digest.update(doc["text"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


class BM25Index:
    """Okapi BM25 with the non-negative IDF used by rank_bm25.

    ``k1=1.5`` and ``b=0.75`` are the Robertson and Zaragoza defaults. The IDF
    is ``log(1 + (N - df + 0.5) / (df + 0.5))`` so a term in most documents
    does not receive a negative weight.
    """

    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids = [doc["id"] for doc in documents]
        tokenized = [tokenize(doc["text"]) for doc in documents]
        self.n = len(tokenized)
        self.lengths = [len(tokens) for tokens in tokenized]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        df: dict[str, int] = {}
        for index, tokens in enumerate(tokenized):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, freq in counts.items():
                df[token] = df.get(token, 0) + 1
                self.postings.setdefault(token, []).append((index, freq))
        self.idf = {
            token: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }

    def top(self, query: str, k: int) -> list[str]:
        if self.n == 0 or self.avgdl <= 0 or k <= 0:
            return []
        scores: dict[int, float] = {}
        seen: set[str] = set()
        for token in tokenize(query):
            if token in seen:
                continue
            seen.add(token)
            posting = self.postings.get(token)
            if not posting:
                continue
            weight = self.idf[token]
            for doc_id, freq in posting:
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.lengths[doc_id] / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (freq * (self.k1 + 1.0) / denom)
        chosen = heapq.nlargest(k, scores, key=scores.get)
        return [self.ids[index] for index in chosen]
