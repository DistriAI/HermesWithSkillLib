#!/usr/bin/env python3
"""Score skill retrievers on the SkillLib catalog.

The methods are SkillLib's own finder and the open comparison systems from
SkillRouter (Zheng et al., arXiv:2603.22455): BM25, the encoder families in
their retrieval grid, and the retrieve-and-rerank pairs. Every method ranks
SkillLib cards. Gold comes from the SkillsBench–SkillLib map.

Dense and rerank methods download weights on first use. While another process
holds the GPU, pass ``--device cpu``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import retrieval_metrics
import retrieval_pool
import skill_map

# Appendix G template for the untuned Qwen3 encoders.
_QUERY_INSTRUCTION = (
    "Instruct: Given a task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)
# The released SkillRouter encoder was evaluated with this wording.
_SKILLROUTER_QUERY = (
    "Instruct: Given a coding task description, retrieve the most relevant "
    "skill document that would help an agent complete the task\nQuery:"
)
_BGE_QUERY = "Represent this sentence for searching relevant passages: "

# kind: skilllib | bm25 | dense | pipeline
# pool: last (Qwen3) | cls (BGE, GTE) | mean (E5) | model (NV-Embed encode())
METHODS = {
    "skilllib": {
        "kind": "skilllib",
        "label": "SkillLib finder (routing-card lexical ranker)",
    },
    "bm25": {
        "kind": "bm25",
        "label": "BM25 over name | description | body (Robertson and Zaragoza, 2009)",
    },
    "bge-large": {
        "kind": "dense",
        "model": "BAAI/bge-large-en-v1.5",
        "pool": "cls",
        "query_prefix": _BGE_QUERY,
        "passage_prefix": "",
        "label": "BGE-Large-v1.5",
    },
    "e5-large": {
        "kind": "dense",
        "model": "intfloat/e5-large-v2",
        "pool": "mean",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "label": "E5-Large-v2",
    },
    "gte-large": {
        "kind": "dense",
        "model": "Alibaba-NLP/gte-large-en-v1.5",
        "pool": "cls",
        "query_prefix": "",
        "passage_prefix": "",
        "label": "GTE-Large-v1.5",
    },
    "qwen3-emb-0.6b": {
        "kind": "dense",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "pool": "last",
        "query_prefix": _QUERY_INSTRUCTION,
        "passage_prefix": "",
        "label": "Qwen3-Embedding-0.6B",
    },
    "qwen3-emb-8b": {
        "kind": "dense",
        "model": "Qwen/Qwen3-Embedding-8B",
        "pool": "last",
        "query_prefix": _QUERY_INSTRUCTION,
        "passage_prefix": "",
        "label": "Qwen3-Embedding-8B",
    },
    "nv-embed-v2": {
        "kind": "dense",
        "model": "nvidia/NV-Embed-v2",
        "pool": "model",
        "query_prefix": _QUERY_INSTRUCTION,
        "passage_prefix": "",
        "label": "NV-Embed-v2",
    },
    "skillrouter": {
        "kind": "dense",
        "model": "pipizhao/SkillRouter-Embedding-0.6B",
        "pool": "last",
        "query_prefix": _SKILLROUTER_QUERY,
        "passage_prefix": "",
        "label": "SkillRouter-Embedding-0.6B",
    },
    "qwen3-0.6b-rerank": {
        "kind": "pipeline",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "reranker": "Qwen/Qwen3-Reranker-0.6B",
        "pool": "last",
        "query_prefix": _QUERY_INSTRUCTION,
        "passage_prefix": "",
        "label": "Qwen3-Embedding-0.6B then Qwen3-Reranker-0.6B",
    },
    "skillrouter-rerank": {
        "kind": "pipeline",
        "model": "pipizhao/SkillRouter-Embedding-0.6B",
        "reranker": "pipizhao/SkillRouter-Reranker-0.6B",
        "pool": "last",
        "query_prefix": _SKILLROUTER_QUERY,
        "passage_prefix": "",
        "label": "SkillRouter 0.6B encoder then 0.6B reranker",
    },
}


def _spec(name: str, model: str | None, reranker: str | None, pool: str | None) -> dict:
    """Resolve a preset plus optional encoder, reranker, and pooling overrides."""
    if name not in METHODS:
        known = ", ".join(METHODS)
        raise SystemExit(f"Unknown method {name}. Choose one of: {known}")
    spec = dict(METHODS[name])
    original = spec["kind"]
    if model:
        spec["model"] = model
    if reranker:
        spec["reranker"] = reranker
    if pool:
        spec["pool"] = pool
    if model and original in {"skilllib", "bm25"}:
        spec["kind"] = "dense"
        spec.setdefault("pool", "last")
        spec.setdefault("query_prefix", _QUERY_INSTRUCTION)
        spec.setdefault("passage_prefix", "")
    if spec.get("reranker") and spec["kind"] == "dense":
        spec["kind"] = "pipeline"
    elif spec.get("reranker") and original == "skilllib" and not model:
        spec["kind"] = "skilllib-pipeline"
    if spec["kind"] in {"dense", "pipeline"} and not spec.get("model"):
        raise SystemExit(f"{name} needs an encoder. Pass --model.")
    if spec["kind"] in {"pipeline", "skilllib-pipeline"} and not spec.get("reranker"):
        raise SystemExit(f"{name} needs a reranker. Pass --reranker.")
    return spec


def _local_snapshot(repo: Path, model_id: str) -> str:
    """Use a snapshot already stored under the SkillRouter model cache when one exists."""
    slug = "models--" + model_id.replace("/", "--")
    hub = repo / "baselines" / "SkillRouter" / "models" / "hub" / slug / "snapshots"
    if not hub.is_dir():
        return model_id
    snapshots = [path for path in hub.iterdir() if (path / "config.json").is_file()]
    return str(snapshots[0]) if snapshots else model_id


def rank_skilllib(queries: list[dict], top_k: int, repo: Path) -> dict[str, list[str]]:
    root = repo / "skill-lib"
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from skilllib.sorting.cards import CARDS_PATH, load_routing_cards
    from skilllib.sorting.finder import Finder, default_families_path, load_families

    finder = Finder(
        load_routing_cards(root / CARDS_PATH),
        load_families(default_families_path(root)),
    )
    rankings = {}
    for index, query in enumerate(queries, start=1):
        found = finder.find(query["text"], limit=top_k, root=root)
        rankings[query["task_id"]] = [card["id"] for card in found.get("cards") or [] if card.get("id")]
        if index % 10 == 0 or index == len(queries):
            print(f"skilllib {index}/{len(queries)}", file=sys.stderr)
    return rankings


def rank_bm25(documents: list[dict], queries: list[dict], top_k: int) -> dict[str, list[str]]:
    index = retrieval_pool.BM25Index(documents)
    return {query["task_id"]: index.top(query["text"], top_k) for query in queries}


def _device(name: str):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested, and torch does not see a GPU.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _last_token(hidden, mask):
    import torch

    left = bool((mask[:, -1].sum() == mask.shape[0]).item())
    if left:
        return hidden[:, -1]
    lengths = mask.sum(dim=1) - 1
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch, lengths]


def _encode_batch(model, tokenizer, texts, pool: str, max_length: int, device, instruction: str = ""):
    import torch
    import torch.nn.functional as F

    if pool == "model" and hasattr(model, "encode"):
        encoded = model.encode(texts, instruction=instruction, max_length=max_length)
        tensor = encoded if isinstance(encoded, torch.Tensor) else torch.tensor(encoded)
        return F.normalize(tensor.float().cpu(), p=2, dim=1)
    tokenizer.padding_side = "left" if pool == "last" else "right"
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"]
        if pool == "cls":
            pooled = hidden[:, 0]
        elif pool == "mean":
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        else:
            pooled = _last_token(hidden, mask)
        return F.normalize(pooled.float(), p=2, dim=1).cpu()


def _encode(model, tokenizer, texts, pool, max_length, batch_size, device, label: str, instruction: str = ""):
    import torch

    parts = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        parts.append(_encode_batch(model, tokenizer, batch, pool, max_length, device, instruction))
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 20) == 0:
            print(f"{label} {done}/{len(texts)}", file=sys.stderr)
    return torch.cat(parts, dim=0) if parts else torch.empty(0)


def _load_encoder(model_id: str, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    return model, tokenizer


def _document_matrix(spec: dict, documents: list[dict], repo: Path, device, batch_size: int, max_length: int):
    model_id = _local_snapshot(repo, spec["model"])
    prefix = spec.get("passage_prefix") or ""
    stamp = retrieval_pool.document_stamp(documents, model_id, spec.get("pool") or "last", prefix)
    cache = repo / "benchmark" / "experiments" / "retrieval" / "indexes" / f"{stamp[:16]}.pt"
    import torch

    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("ids") == [doc["id"] for doc in documents]:
            print(f"loaded index {cache}", file=sys.stderr)
            return saved["embeddings"]
    model, tokenizer = _load_encoder(model_id, device)
    texts = [prefix + doc["text"] for doc in documents]
    matrix = _encode(model, tokenizer, texts, spec.get("pool") or "last", max_length, batch_size, device, "documents")
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": [doc["id"] for doc in documents], "embeddings": matrix}, cache)
    print(f"wrote index {cache}", file=sys.stderr)
    del model
    return matrix


def rank_dense(spec, documents, queries, repo, top_k, device_name, batch_size, max_length) -> dict[str, list[str]]:
    import torch

    device = _device(device_name)
    matrix = _document_matrix(spec, documents, repo, device, batch_size, max_length)
    model_id = _local_snapshot(repo, spec["model"])
    model, tokenizer = _load_encoder(model_id, device)
    pool = spec.get("pool") or "last"
    prefix = "" if pool == "model" else (spec.get("query_prefix") or "")
    instruction = (spec.get("query_prefix") or "") if pool == "model" else ""
    texts = [prefix + query["text"][: retrieval_pool.QUERY_MAX] for query in queries]
    query_matrix = _encode(
        model, tokenizer, texts, pool, max_length, batch_size, device, "queries", instruction
    )
    scores = query_matrix @ matrix.T
    k = min(top_k, scores.shape[1])
    chosen = torch.topk(scores, k=k, dim=1).indices.tolist()
    ids = [doc["id"] for doc in documents]
    return {query["task_id"]: [ids[index] for index in row] for query, row in zip(queries, chosen)}


def _rerank_ids(spec, documents, queries, first_pass, repo, rerank_k, device_name, batch_size) -> dict[str, list[str]]:
    """Reorder the first ``rerank_k`` of each list. The tail keeps retrieval order."""
    import torch

    root = repo / "baselines" / "SkillRouter"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.common import load_reranker_model
    from src.run_open_model_eval import score_candidates_with_reranker

    device = _device(device_name)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_id = _local_snapshot(repo, spec["reranker"])
    model, tokenizer = load_reranker_model(model_id, dtype=dtype)
    model.to(device).eval()
    by_id = {doc["id"]: doc for doc in documents}
    rankings = {}
    for index, query in enumerate(queries, start=1):
        ranked = list(first_pass.get(query["task_id"]) or [])
        head = [by_id[item] for item in ranked[:rerank_k] if item in by_id]
        if len(head) > 1:
            scores = score_candidates_with_reranker(
                model, tokenizer, query["text"][: retrieval_pool.QUERY_MAX], head,
                "flat-full", 4096, max(1, batch_size // 2), device,
            )
            order = sorted(range(len(head)), key=lambda item: scores[item], reverse=True)
            head_ids = [head[item]["id"] for item in order]
            tail = [item for item in ranked if item not in set(head_ids)]
            ranked = head_ids + tail
        rankings[query["task_id"]] = ranked
        if index % 10 == 0 or index == len(queries):
            print(f"rerank {index}/{len(queries)}", file=sys.stderr)
    return rankings


def score_rankings(rankings: dict[str, list[str]], skill_map_doc: dict) -> dict:
    rows = []
    for task_id, ranked in rankings.items():
        entry = skill_map.task_entry(skill_map_doc, task_id)
        if entry is None:
            continue
        rows.append(retrieval_metrics.catalog_score(ranked, entry.get("skills") or []))
    return retrieval_metrics.summarize_catalog(rows)


def run(spec: dict, repo: Path, queries: list[dict], top_k: int, rerank_k: int, device: str, batch_size: int, max_length: int) -> dict:
    kind = spec["kind"]
    documents: list[dict] = []
    if kind != "skilllib":
        cache = repo / "benchmark" / "experiments" / "retrieval" / "documents.jsonl"
        documents = retrieval_pool.load_documents(repo, cache)
        print(f"documents {len(documents)}", file=sys.stderr)
    if kind in {"skilllib", "skilllib-pipeline"}:
        rankings = rank_skilllib(queries, top_k, repo)
    elif kind == "bm25":
        rankings = rank_bm25(documents, queries, top_k)
    else:
        rankings = rank_dense(spec, documents, queries, repo, top_k, device, batch_size, max_length)
    if spec.get("reranker"):
        rankings = _rerank_ids(spec, documents or retrieval_pool.load_documents(repo), queries, rankings, repo, rerank_k, device, batch_size)
    return rankings


def format_summary(summary: dict) -> str:
    pipeline = (summary.get("pipeline") or {}).get("all") or {}
    covered = (summary.get("on_covered_gold") or {}).get("all") or {}

    def rate(block, key):
        value = block.get(key)
        return "n/a" if value is None else f"{float(value):.4f}"

    return (
        f"tasks={summary.get('n_tasks', 0)} "
        f"coverage={rate(summary, 'coverage')} "
        f"Hit@1={rate(pipeline, 'Hit@1')} "
        f"MRR@10={rate(pipeline, 'MRR@10')} "
        f"Recall@10={rate(pipeline, 'Recall@10')} "
        f"Recall@20={rate(pipeline, 'Recall@20')} "
        f"Recall@50={rate(pipeline, 'Recall@50')} "
        f"FullCoverage@10={rate(pipeline, 'FullCoverage@10')} "
        f"FullCoverageCovered@5={rate(pipeline, 'FullCoverageCovered@5')} "
        f"FullCoverageCovered@10={rate(pipeline, 'FullCoverageCovered@10')} "
        f"on_covered n={summary.get('n_with_covered_gold', 0)} "
        f"Hit@1={rate(covered, 'Hit@1')} "
        f"Recall@10={rate(covered, 'Recall@10')} "
        f"FullCoverage@10={rate(covered, 'FullCoverage@10')}"
    )


def _parser() -> argparse.ArgumentParser:
    names = ", ".join(f"{name} ({spec['label']})" for name, spec in METHODS.items())
    parser = argparse.ArgumentParser(
        description="Evaluate a skill retriever over the SkillLib catalog on SkillsBench.",
        epilog=names,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--model", help="Hugging Face id or local path. Overrides the method's encoder.")
    parser.add_argument("--reranker", help="Yes/no reranker id. Reorders the first --rerank-k candidates.")
    parser.add_argument("--pool", choices=["last", "cls", "mean", "model"], help="Dense pooling. Overrides the preset.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=50, help="Candidates kept per task. Recall@50 needs 50.")
    parser.add_argument("--rerank-k", type=int, default=20, help="How many of those candidates the reranker reorders.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048, help="Encoder token cap. The paper trains at 2048.")
    parser.add_argument("--start", type=int, help="Inclusive 0-based index into the sorted task ids.")
    parser.add_argument("--end", type=int, help="Inclusive 0-based index into the sorted task ids.")
    parser.add_argument("--output", type=Path, help="Defaults to benchmark/experiments/retrieval/<method>.json")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    repo = retrieval_pool.repo_root()
    spec = _spec(args.method, args.model, args.reranker, args.pool)
    document = skill_map.load()
    queries = retrieval_pool.load_queries(document, repo, args.start, args.end)
    if not queries:
        raise SystemExit("No SkillsBench task statements matched the skill map.")
    print(f"queries {len(queries)} method {args.method}", file=sys.stderr)
    rankings = run(spec, repo, queries, args.top_k, args.rerank_k, args.device, args.batch_size, args.max_length)
    summary = score_rankings(rankings, document)
    payload = {
        "method": args.method,
        "label": spec.get("label") or METHODS[args.method]["label"],
        "model": spec.get("model"),
        "reranker": spec.get("reranker"),
        "top_k": args.top_k,
        "rerank_k": args.rerank_k,
        "n_queries": len(queries),
        "summary": summary,
        "rankings": rankings,
    }
    output = args.output or (repo / "benchmark" / "experiments" / "retrieval" / f"{args.method}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(format_summary(summary))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
