#!/usr/bin/env bash
# Rebuild the SkillCorpus library and the SkillNet index from the crawled skills.
# Chat judgments use http://10.50.0.54:8090/v1 (Qwen/Qwen3.8-27B).
# Embeddings use the local CPU server started below.
#
# Manual replay, from the repo root:
#   python3 -m venv baselines/.venv-construct
#   baselines/.venv-construct/bin/pip install -r baselines/SkillCorpus/requirements.txt
#   baselines/.venv-construct/bin/pip install -e 'baselines/SkillNet/skillnet-ai[graph]'
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u baselines/embedding_server.py
#   baselines/.venv-construct/bin/python -u baselines/SkillCorpus/hermes_build.py --publish
#   baselines/.venv-construct/bin/python -u baselines/SkillNet/hermes_build.py --publish
# The embedding server must be listening on http://127.0.0.1:8100/health before the builds.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/baselines/construction-commands.log"
BUILD_PY="$ROOT/baselines/.venv-construct/bin/python"
EMBED_PY="$ROOT/.venv/bin/python"

mkdir -p "$ROOT/baselines"
{
  echo
  echo "===== $(date -Is) construct-repositories.sh ====="
  echo "root=$ROOT"
} >> "$LOG"
exec > >(tee -a "$LOG") 2>&1
set -x

cd "$ROOT"

if [[ ! -x "$BUILD_PY" ]]; then
  python3 -m venv "$ROOT/baselines/.venv-construct"
  "$ROOT/baselines/.venv-construct/bin/pip" install -r "$ROOT/baselines/SkillCorpus/requirements.txt"
  "$ROOT/baselines/.venv-construct/bin/pip" install -e "$ROOT/baselines/SkillNet/skillnet-ai[graph]"
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:8100/health >/dev/null; then
  nohup "$EMBED_PY" "$ROOT/baselines/embedding_server.py" >> "$ROOT/baselines/embedding-server.log" 2>&1 &
  echo $! > "$ROOT/baselines/embedding-server.pid"
  # The embedding weights download on first start. At a slow link that can take hours.
  for _ in $(seq 1 960); do
    if curl -fsS --max-time 2 http://127.0.0.1:8100/health >/dev/null; then
      break
    fi
    sleep 30
  done
  curl -fsS --max-time 5 http://127.0.0.1:8100/health
fi

"$BUILD_PY" -u "$ROOT/baselines/SkillCorpus/hermes_build.py" --publish
"$BUILD_PY" -u "$ROOT/baselines/SkillNet/hermes_build.py" --publish

echo "construction finished $(date -Is)"
