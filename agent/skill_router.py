"""Adapter for the SkillRouter baseline (zhengyanzhao1997/SkillRouter).

Hermes stays the agent. This module locates the checkout, ranks the published
skill pool with the released embedding and reranker, and returns cards the
agent can load. The pool embeddings are cached under the checkout's
``outputs/`` directory after the first query. Model weights download into
``baselines/SkillRouter/models`` and are not part of the git index.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "SkillRouter"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "baselines" / "skillrouter.json"
_DEFAULT_ENCODER = "pipizhao/SkillRouter-Embedding-0.6B"
_DEFAULT_RERANKER = "pipizhao/SkillRouter-Reranker-0.6B"
_TOP_CARDS = 5
_RETRIEVAL_K = 50
_RERANK_K = 20

_lock = threading.Lock()
_pool: list[dict] | None = None
_pool_ids: list[str] | None = None
_embeddings: Any = None
_models: tuple[Any, Any, Any, Any] | None = None

TOOL_NAMES = ("skill_router_find", "skill_router_load")
_TASK_INSTRUCTION = (
    "Before writing the solution, call skill_router_find for this task and "
    "skill_router_load on the best matches (at most 3). Adapt what you load; "
    "a skill is a procedure, not the answer."
)


def package_root() -> Path | None:
    """SkillRouter checkout, or None when ``baselines/SkillRouter`` is absent."""
    if (_PACKAGE_ROOT / "src" / "common.py").is_file():
        return _PACKAGE_ROOT
    return None


def model_cache() -> Path | None:
    """Directory where Hugging Face weights for this baseline are stored."""
    root = package_root()
    if root is None:
        return None
    path = root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config() -> dict:
    if not _CONFIG_PATH.is_file():
        return {}
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def tier() -> str:
    value = str(_config().get("tier") or "hard")
    return value if value in {"easy", "hard"} else "hard"


def pool_dir() -> Path | None:
    root = package_root()
    if root is None:
        return None
    return root / "data" / "eval_core" / tier()


def _deps_ready() -> bool:
    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


def pool_ready() -> bool:
    """True when the selected tier has at least one skill-pool shard."""
    directory = pool_dir()
    if directory is None or not directory.is_dir():
        return False
    return any(directory.glob("*.jsonl")) or any(directory.glob("*.jsonl.gz"))


def router_available() -> bool:
    """True when a session can offer the SkillRouter tools."""
    return package_root() is not None and pool_ready() and _deps_ready()


def unavailable_reason() -> str:
    if package_root() is None:
        return "SkillRouter checkout is missing. Run git submodule update --init baselines/SkillRouter."
    if not pool_ready():
        return (
            f"SkillRouter {tier()} skill pool is not downloaded. "
            "From baselines/SkillRouter, run bash scripts/download_eval_data.sh."
        )
    if not _deps_ready():
        return "SkillRouter needs torch and transformers in this Python environment."
    return ""


def tool_names() -> tuple[str, ...]:
    return TOOL_NAMES


def tool_specs() -> list[dict]:
    return [
        {
            "name": "skill_router_find",
            "description": (
                "Rank skills for a task from the SkillRouter pool. "
                "Returns the highest-ranked skill cards. Load a card before following it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The task to rank skills for."},
                },
                "required": ["task"],
            },
        },
        {
            "name": "skill_router_load",
            "description": "Return the full text of one skill from the SkillRouter pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "A skill_id from skill_router_find."},
                },
                "required": ["skill_id"],
            },
        },
    ]


def task_instruction() -> str:
    return _TASK_INSTRUCTION


def router_prompt(available_tools: set[str] | None) -> str:
    """Resident prompt when this session offers both SkillRouter tools."""
    if not available_tools or any(name not in available_tools for name in TOOL_NAMES):
        return ""
    if not router_available():
        return ""
    return _TASK_INSTRUCTION


def _modules():
    root = package_root()
    if root is None:
        raise ImportError("SkillRouter checkout is missing")
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    import src.common as common
    import src.data_io as data_io
    import src.run_open_model_eval as runner
    return common, data_io, runner


def _device(torch_module):
    choice = str(_config().get("device") or "auto")
    if choice == "cpu":
        return torch_module.device("cpu")
    if choice == "cuda" and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _load_models():
    global _models
    if _models is not None:
        return _models
    cache = model_cache()
    if cache is not None:
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    common, _data_io, _runner = _modules()
    import torch
    device = _device(torch)
    config = _config()
    encoder = str(config.get("encoder") or _DEFAULT_ENCODER)
    reranker = str(config.get("reranker") or _DEFAULT_RERANKER)
    emb_model, emb_tokenizer = common.load_embedding_model(encoder)
    rr_model, rr_tokenizer = common.load_reranker_model(reranker)
    emb_model.to(device).eval()
    rr_model.to(device).eval()
    _models = (emb_model, emb_tokenizer, rr_model, rr_tokenizer)
    return _models


def _stamp(directory: Path) -> str:
    parts = []
    for path in sorted(directory.iterdir()):
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _index_path() -> Path:
    root = package_root()
    assert root is not None
    path = root / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"hermes-index-{tier()}.pt"


def _ensure_pool() -> None:
    """Read the skill-pool shards into memory. Does not load a model."""
    global _pool, _pool_ids
    if _pool is not None and _pool_ids is not None:
        return
    directory = pool_dir()
    if directory is None or not pool_ready():
        raise RuntimeError(unavailable_reason())
    _common, data_io, _runner = _modules()
    _pool = data_io.load_jsonl(directory)
    _pool_ids = [str(row.get("skill_id") or "") for row in _pool]


def _ensure_index():
    """Load the pool and its normalized embeddings, building the cache once."""
    global _embeddings
    if _embeddings is not None and _pool is not None and _pool_ids is not None:
        return
    directory = pool_dir()
    if directory is None or not pool_ready():
        raise RuntimeError(unavailable_reason())
    _ensure_pool()
    common, _data_io, _runner = _modules()
    import torch
    stamp = _stamp(directory)
    cache = _index_path()
    if cache.is_file():
        saved = torch.load(cache, map_location="cpu", weights_only=False)
        if saved.get("stamp") == stamp and list(saved.get("ids") or []) == _pool_ids:
            _embeddings = saved["emb"]
            return
    emb_model, emb_tokenizer, _rr_model, _rr_tokenizer = _load_models()
    device = _device(torch)
    texts = [common.format_skill(row, desc_max=500, body_max=8000) for row in _pool]
    _embeddings = common.encode_texts(emb_model, emb_tokenizer, texts, 4096, 32, device)
    torch.save({"stamp": stamp, "ids": _pool_ids, "emb": _embeddings}, cache)


def _rank(task: str) -> tuple[list[dict], list[str]]:
    """Return reranked skill rows and the cosine top-50 ids."""
    _ensure_index()
    common, _data_io, runner = _modules()
    import torch
    emb_model, emb_tokenizer, rr_model, rr_tokenizer = _load_models()
    device = _device(torch)
    query = common.encode_texts(
        emb_model, emb_tokenizer, [common.format_query(task, max_len=2000)], 4096, 1, device,
    )
    sims = query @ _embeddings.T
    width = min(_RETRIEVAL_K, len(_pool_ids or []))
    _values, indexes = torch.topk(sims[0], width)
    positions = indexes.tolist()
    retrieval_ids = [_pool_ids[index] for index in positions]
    take = positions[:_RERANK_K]
    candidates = [_pool[index] for index in take]
    scores = runner.score_candidates_with_reranker(
        rr_model, rr_tokenizer, task, candidates, "flat-full", 4096, 8, device,
    )
    order = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)
    return [candidates[index] for index in order], retrieval_ids


def find(task: str) -> dict:
    text = task.strip()
    if not text:
        return {"success": False, "error": "task is empty"}
    if not router_available():
        return {"success": False, "error": unavailable_reason()}
    try:
        with _lock:
            ranked, retrieval_ids = _rank(text)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    cards = []
    for row in ranked[:_TOP_CARDS]:
        description = str(row.get("description") or row.get("desc") or "")
        cards.append({
            "skill_id": str(row.get("skill_id") or ""),
            "name": str(row.get("name") or ""),
            "description": description[:400],
        })
    return {
        "success": True,
        "cards": cards,
        "ranked_ids": [str(row.get("skill_id") or "") for row in ranked],
        "retrieval_ids": retrieval_ids,
    }


def load(skill_id: str) -> dict:
    ident = skill_id.strip()
    if not ident:
        return {"success": False, "error": "skill_id is empty"}
    if not pool_ready():
        return {"success": False, "error": unavailable_reason()}
    try:
        with _lock:
            _ensure_pool()
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    for row in _pool or []:
        if str(row.get("skill_id") or "") == ident:
            body = str(row.get("body") or "")
            return {
                "success": bool(body),
                "skill_id": ident,
                "name": str(row.get("name") or ""),
                "description": str(row.get("description") or row.get("desc") or ""),
                "text": body,
                "error": "" if body else "skill body is empty",
            }
    return {"success": False, "error": f"skill {ident} is not in the {tier()} pool"}


def call_tool(name: str, arguments: dict | None = None) -> dict:
    args = arguments or {}
    if name == "skill_router_find":
        return find(str(args.get("task") or ""))
    if name == "skill_router_load":
        return load(str(args.get("skill_id") or ""))
    return {"success": False, "error": f"unknown tool {name}"}
