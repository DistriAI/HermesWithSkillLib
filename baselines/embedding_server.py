"""CPU embedding server for the SkillCorpus and SkillNet builds.

The chat model at 10.50.0.54:8090 does not serve /v1/embeddings, and its GPU
is already full. Hugging Face cannot be reached from this machine. This
process serves the Qwen3-Embedding-8B weights already stored under
baselines/SkillRouter, on CPU, at http://127.0.0.1:8100/v1/embeddings.
SkillCorpus near-duplicate detection and SkillNet analyze both call it.

Run with the Hermes virtualenv, which already has torch and transformers:

    .venv/bin/python baselines/embedding_server.py
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as functional
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    str(Path(__file__).resolve().parent / "SkillRouter" / "models" / "hub"
    / "models--Qwen--Qwen3-Embedding-8B" / "snapshots"
    / "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"),
)
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
HOST = "127.0.0.1"
PORT = 8100
MAX_LENGTH = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left", local_files_only=True)
model = AutoModel.from_pretrained(MODEL, local_files_only=True, dtype=torch.float32)
model.eval()
torch.set_num_threads(int(os.environ.get("EMBEDDING_THREADS", "8")))
_lock = threading.Lock()


def embed(texts: list[str]) -> list[list[float]]:
    batch = tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    with _lock, torch.no_grad():
        hidden = model(**batch).last_hidden_state
    index = batch["attention_mask"].sum(dim=1) - 1
    pooled = hidden[torch.arange(hidden.shape[0]), index]
    pooled = functional.normalize(pooled, p=2, dim=1)
    return pooled.tolist()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/models"}:
            self._json(200, {"ok": True, "model": MODEL_NAME, "dim": model.config.hidden_size})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/embeddings":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or "0")
        raw = json.loads(self.rfile.read(length) or b"{}")
        texts = raw.get("input")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or not texts or not all(isinstance(item, str) for item in texts):
            self._json(400, {"error": "input must be a string or a list of strings"})
            return
        vectors = embed(texts)
        self._json(200, {
            "object": "list",
            "model": MODEL_NAME,
            "data": [
                {"object": "embedding", "index": i, "embedding": vector}
                for i, vector in enumerate(vectors)
            ],
        })


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"embedding server http://{HOST}:{PORT}/v1 model={MODEL_NAME} dim={model.config.hidden_size}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
