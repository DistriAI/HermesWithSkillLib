# Benchmarks in this checkout

Four benchmarks run from this repository. Each has its own script under `benchmark/scripts/`. Shared code lives in `benchmark/scripts/common.py`. The benchmark trees are submodules of this project:

| Path | Upstream |
|---|---|
| `benchmark/skillsbench` | https://github.com/benchflow-ai/skillsbench.git |
| `benchmark/scienceagentbench` | https://github.com/OSU-NLP-Group/ScienceAgentBench.git |
| `benchmark/frontier-evals` | https://github.com/openai/frontier-evals.git |
| `benchmark/terminal-bench` | https://github.com/harbor-framework/terminal-bench.git |

After the first clone, check the submodules out at the commits this project records. The same command also fetches the SkillLib submodule. Run it again after a later pull moves a submodule commit. Experiment directories, the ScienceAgentBench zip, and fetched task sheets stay untracked. `python benchmark/scripts/<name>.py clone` initializes that one submodule when its directory is missing.

Commands assume the repository root, the project virtualenv, and this checkout's Hermes profile:

```bash
uv venv --python 3.14 .venv          # once
uv sync --frozen --python .venv/bin/python
source .venv/bin/activate
export HERMES_HOME="$PWD/.hermes"
mkdir -p "$HERMES_HOME"
git submodule update --init --recursive
```

`uv sync` stays `--frozen`. A plain `uv sync` re-resolves and can rewrite `uv.lock`. API keys go in `$HERMES_HOME/.env`. Model settings go in `$HERMES_HOME/config.yaml` (`hermes setup` writes that file).

## Isolated experiment

`--run NAME` creates one directory, `benchmark/experiments/<benchmark>/<NAME>/`. Every task in that run uses it. There is not a separate Hermes profile per task.

```
.hermes/                     the run's Hermes profile (config and .env)
tasks/<task-id>/             copy of that task, including grader files the agent cannot read
agent/<task-id>/workspace/   files that task's agent writes
agent/<task-id>/result.json  grades, time, and tokens
agent/<task-id>/turns/<n>/   prompt and usage for the nth time the runner invoked the agent
trajectory.jsonl             every task's conversation, appended as the run proceeds
summary.json
summary.txt
```

The profile copy includes `config.yaml`, `.env`, and other small profile text files. Session databases from the source profile are not copied. Later tasks in the same run share the sessions this run creates under that one `.hermes`. A different `--run` name gets a different directory.

The worker runs in a container. The container can write that task's `agent/` directory, the run's `.hermes`, and, for a SkillRouter run, `baselines/SkillRouter/outputs` and `baselines/SkillRouter/models`. It sees the repository read-only, with grader paths covered: directories named `verifier`, `oracle`, `eval_programs`, `gold_programs`, or `scoring_rubrics`, a Terminal-Bench `tests` directory that holds `test_outputs.py`, and PaperBench `rubric.json`. The checkout and other experiments are not a way to open those files.

`trajectory.jsonl` is one file for the whole run. Each line is one JSON object and carries `task` and `turn`. `turn` is the conversation depth: a user message starts a turn, and an assistant message that calls tools starts the next one. Tool results keep the turn of the call they answer. A turn header (`type`, `exit_code`, `session_ids`, `runner_turn`) marks each time the runner invoked the agent and records the depth reached then. Starting a task again removes that task's previous lines and appends the new ones.

## Options

`list` and `run` select tasks the same way. `list` prints `index<TAB>task-id` so the indexes match `--start` and `--end`. Each example sets `RUN` first. Change that name to keep a new experiment separate from the previous one.

```bash
RUN=exp1
python benchmark/scripts/skillsbench.py list --start 0 --end 4
python benchmark/scripts/skillsbench.py run --run "$RUN" --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary "benchmark/experiments/skillsbench/$RUN/summary.json" \
  --report "benchmark/experiments/skillsbench/$RUN/summary.txt"
```

| Option | Usage |
|---|---|
| `--run NAME` | Experiment directory name. Required for `run`, `summarize`, and `regrade`. |
| `--split NAME` | Subset whose order defines the indexes. PaperBench: `debug`, `lite`, `dev`, `human`, `testing`, `all` (the order in that split file). ScienceAgentBench: a `split` value in `tasks.jsonl`, such as `verified` or `validation`, and `train` or `test` when those rows have been fetched. SkillsBench and Terminal-Bench have no named splits; omit it or pass `all`. |
| `--start N` | First index in that split. 0-based, inclusive. Default: 0 when `--end` is set. |
| `--end N` | Last index in that split. 0-based, inclusive. Default: the last task when `--start` is set. `--start 0 --end 9` is the first 10 tasks. |
| `--task ID` | One id, or a comma-separated list, kept only when it also falls in the index range. |
| `--limit N` | After the index range, keep at most N tasks. |
| `--turns 1,2,4,8` | On `run`, conversation-turn budgets. A turn is a user message or an assistant tool-call message. The agent is graded after each invocation and stops once the conversation reaches the largest budget or the verifier accepts the workspace. Default `1,2,4,8`. On `summarize` and `regrade`, the k values to report from those stored grades. |
| `--skill-mode MODE` | `task`, `library` (default), `library-compatible`, `library-only`, `router`, `router-compatible`, `router-only`, or `none`. See the skill-mode table below. |
| `--model ID` | Model id, such as `Qwen/Qwen3.6-27B`. With `--provider` and no model, that provider's `default_model` is used. |
| `--provider NAME` | A name under `providers:` in the replicated profile, such as `qwen-local` or `qwen-31`. A model without this flag keeps `model.provider` from the profile. |
| `--timeout SECONDS` | Wall-clock limit for one task, across its conversation turns. |
| `--hermes-home PATH` | Profile to copy into the experiment. Default: `$HERMES_HOME`, otherwise `<repo>/.hermes`. |
| `--fresh` | Run tasks again even when their `result.json` status is `complete`. |
| `--summary PATH` | JSON summary. Default: `<experiment>/summary.json`. |
| `--report PATH` | Text summary. Default: `<experiment>/summary.txt`. |

`summarize --run NAME` reads each finished task's `result.json` and recomputes the rates. `--turns 3` or `--turns 3,6` reports those k values even when the original run used a different list, such as `--turns 1,2,4,8`. `regrade --run NAME` grades the last turn again without calling the model. Both accept `--summary` and `--report`.

`--model` and `--provider` are stored in `run.json`, each task's `result.json`, and the summary. Omit both and the agent uses `model.default` and `model.provider` from the copied profile.

| Skill mode | Hermes skills | Task skills | SkillLib | SkillRouter |
|---|---|---|---|---|
| `task` | Synced into the experiment profile. `skills_list`, `skill_view`, and `skill_manage` stay on. | Copied into the workspace. SkillsBench uses `environment/skills/`. ScienceAgentBench includes `domain_knowledge`. The prompt says to read each `SKILL.md`. | Off. | Off. |
| `library` | Off. The `skills` toolset is removed. | Same task skills as `task`. | On. The prompt also appends SkillLib's `task_instruction()`. | Off. |
| `library-compatible` | On, as in `task`. | On, as in `task`. | On, as in `library`. | Off. |
| `library-only` | Off. | Not copied, and `domain_knowledge` is omitted. | On. | Off. |
| `router` | Off. | Same task skills as `task`. | Off. | On. The agent calls `skill_router_find` and `skill_router_load`. |
| `router-compatible` | On, as in `task`. | On, as in `task`. | Off. | On, as in `router`. |
| `router-only` | Off. | Not copied, and `domain_knowledge` is omitted. | Off. | On. |
| `none` | Off. | Off. | Off. | Off. |

Hermes skills are the bundled skills from this checkout. The runner syncs them into the experiment's `.hermes/skills/` when the mode includes them. It does not copy the source profile's `skills/` directory. SkillLib reads `skill-lib/skilllib.json` from the checkout, which is shared by every experiment. SkillRouter reads the pool under `baselines/SkillRouter/data/eval_core/`. Optional `baselines/skillrouter.json` may set `tier` (`hard` or `easy`), `encoder`, `reranker`, and `device` (`auto`, `cpu`, or `cuda`). The default tier is `hard`.

Preparing SkillRouter is separate from the Hermes lock. `torch` and `transformers` are not installed by `uv sync`. The tools stay hidden until the pool shards exist and those packages import in the same Python that runs the worker.

```bash
git submodule update --init baselines/SkillRouter
cd baselines/SkillRouter
uv pip install --python ../../.venv/bin/python -r requirements.txt huggingface_hub
HF_ENDPOINT=https://hf-mirror.com bash scripts/download_eval_data.sh
```

Drop `HF_ENDPOINT` when `huggingface.co` is reachable. The first `skill_router_find` downloads `pipizhao/SkillRouter-Embedding-0.6B` and `pipizhao/SkillRouter-Reranker-0.6B` into `baselines/SkillRouter/models` and writes the pool embedding cache to `baselines/SkillRouter/outputs/`. Those weights, shards, and the cache are not git-indexed. The vLLM server on this machine already fills the GPU; set `"device": "cpu"` in `baselines/skillrouter.json` when the two processes would share that GPU.

## Metrics

`--turns` is a comma-separated list of conversation-turn budgets, default `1,2,4,8`. A turn is one user message or one assistant message that calls tools. Tool results belong to that tool-call turn. An assistant reply with no tool call does not open another turn. The first user message is the task prompt. After each time the runner invokes the agent, it grades the workspace and stores that grade at the conversation depth reached. A task that needed several tool-call rounds therefore does not count at success@1. The runner sends another user message only while checks are still failing and the depth is still under the largest budget. That message says the workspace is not accepted yet and tells the agent to re-check the task statement. If those written checks already pass, the agent is told to stop without changing the design. If a written requirement is still unmet, it is told to change only that part. The message does not name the checks or quote the assertion. The conversation stops when the verifier accepts the workspace, or when the agent stops because its own check of the statement already passes.

| Benchmark | The conversation stops when |
|---|---|
| SkillsBench | Every test in `verifier/test_outputs.py` passes. `/root` is the task workspace. `verifier/test.sh` is not run. |
| Terminal-Bench | Every test in `tests/test_outputs.py` passes. `/app` is the workspace and `/tests` is the replicated tests directory. `tests/test.sh` is not run. |
| ScienceAgentBench | `program_saved`, `program_ran`, `output_saved`, and any unpacked `eval()` criteria all pass. |
| PaperBench | `submission/README.md` exists in the workspace. Rubric leaves are scored later from `judge.json` and do not add more turns. |

| Field | Meaning |
|---|---|
| `macro_success@k` | Fraction of completed tasks that passed every test case on any stored conversation turn at or before k. A pass at conversation turn 4 counts at k = 4 and above. It does not count at k = 1 when the transcript shows the agent used later tool-call turns before the workspace passed. |
| `micro_success@k` | Fraction of test cases that passed on any stored conversation turn at or before k. |
| `macro_auc@k`, `micro_auc@k` | Mean of success@1 through success@k. Each of those rates uses the same at-or-before rule. |
| `final_macro_success`, `final_micro_success` | Pass rate on each task's last turn. |
| `execution_time_sec` | Mean and sample standard deviation of wall time, for tasks that passed on the last turn, and for every completed task. |
| `input_tokens`, `output_tokens`, `total_tokens` | The same two groups. Counts add the main request and auxiliary calls. |
| `skill_library` | Counts of `skill_library_find`, `skill_library_load`, and `skill_library_visit`, the cards offered, the skill ids loaded, visits, gaps, and errors. `final_macro` is also split by calling behavior: any call, no call, a load, or find without a load. |
| `skill_retrieval` | Two retrieval scores on the first find result. The SkillRouter score uses `core_gt_ids` from `baselines/SkillRouter/data/eval_core/relevance.json` when the task is listed there and is not `generic_only`; otherwise the gold ids are the task's `environment/skills` directory names. A retrieved id matches when the strings are equal or their skill-name component is equal. `pipeline` is that score on the reranked list (`Hit@1`, `Precision@3`, `MRR@10`, `Recall@10`, `Recall@20`, `Recall@50`, `FullCoverage@3`, `FullCoverage@5`, `FullCoverage@10`, `nDCG@1`, `nDCG@3`, `nDCG@10`). `candidates` is the same score on the cosine top 50. `single` and `multi` split by how many of those gold skills the task has. `catalog` is the SkillLib score: a `skill_library_find` id hits a task skill only when `benchmark/scripts/data/skillsbench_skilllib_map.json` lists it. Metrics on the full skill set count an unmapped skill as a miss. `FullCoverageCovered@k` is 1 when every skill the map links for that task is inside the top k, and is 1 when the map links none. `on_covered_gold` repeats the SkillRouter metrics using only the linked skills, averaged over tasks that have at least one. |

One completed task has standard deviation 0. Interrupted tasks are left out of the rates until a rerun finishes them.

### Skill retrieval map

SkillsBench is the retrieval benchmark: each task ships the skills a correct retrieval should return. SkillLib cards are not the same files. `benchmark/scripts/data/skillsbench_skilllib_map.json` records, for every `tasks/<id>/environment/skills/<name>/SKILL.md`, the SkillLib card ids that count as that skill, and which tasks ship which skills.

A card matches when its cached file has the same SHA-256 as the task skill, or, failing that, when the card has the same skill name and either contains the task skill's tokens (containment at least 0.95) or shares a token Jaccard of at least 0.75. Same-name cards below both cuts are left unmapped. Their nearest candidate is stored on the skill. The current map covers 229 of 232 task skills. The three left out are `multilingual-video-dubbing/ffmpeg-video-editing`, `sales-pivot-analysis/pdf`, and `sales-pivot-analysis/xlsx`.

`FullCoverage@k` on the catalog score is 1 only when every skill shipped with the task is retrieved. `FullCoverageCovered@k` is 1 when every skill the map links into SkillLib is retrieved. A task whose skills are all absent from SkillLib scores 1 on that variant, because there is no mapped skill left to miss, and it stays out of `on_covered_gold`.

Rebuild the map after the catalog or the SkillsBench tasks change:

```bash
python benchmark/scripts/skill_map.py
```

`retrieval_metrics.evaluate_library(task_id, ranked_ids)` scores a ranking with no agent run. The ids have to be catalog card ids or source occurrence ids from the map.

### Skill retrieval baselines

`benchmark/scripts/retrieval_baselines.py` ranks the SkillLib catalog for every SkillsBench task and scores the ranking with the map. The query is the task statement in `task.md`. BM25 and the dense encoders score `name | description | body`, with the description capped at 300 characters and the body at 2,500, the same field caps as SkillRouter. The SkillLib method is the finder that ranks routing cards. A reranker reorders the first 20 candidates and leaves the rest of the list in retrieval order, so Hit@1 reflects the reranker and Recall@50 still sees the encoder's list.

The method names are the open systems in SkillRouter's retrieval grid and retrieve-and-rerank table: `bm25`, `bge-large`, `e5-large`, `gte-large`, `qwen3-emb-0.6b`, `qwen3-emb-8b`, `nv-embed-v2`, `skillrouter`, `qwen3-0.6b-rerank`, and `skillrouter-rerank`. `skilllib` is this catalog's own ranker. `text-embedding-3-large` and `gemini-embedding-001` are the paper's API baselines; this script does not call them. `--model` and `--reranker` accept another Hugging Face id.

`--start` and `--end` are inclusive 0-based indexes into the sorted task ids. Omit both to score all 87 tasks. The retriever still searches the full catalog. Results go to `benchmark/experiments/retrieval/<method>.json`, which stays untracked. The first dense run writes an embedding cache beside that file and reuses it when the catalog text is unchanged.

```bash
python benchmark/scripts/retrieval_baselines.py --method skilllib
python benchmark/scripts/retrieval_baselines.py --method bm25
python benchmark/scripts/retrieval_baselines.py --method bm25 --start 0 --end 0

# The GPU is often already serving the agent. These download weights on first use.
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method bge-large --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method e5-large --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method gte-large --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method qwen3-emb-0.6b --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method skillrouter --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method skillrouter-rerank --device cpu
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  python benchmark/scripts/retrieval_baselines.py --method qwen3-0.6b-rerank --device cpu
```

`qwen3-emb-8b` and `nv-embed-v2` use the same command with `--device cuda` when a GPU is free. A local checkpoint replaces the preset with `--model /path/to/encoder`. `--reranker Qwen/Qwen3-Reranker-0.6B` reorders the top 20 of whichever method is selected.

Re-running the same `--run` command skips a task whose `result.json` status is `complete`. Status `running`, `timeout`, or `error` starts that task over. `--fresh` runs completed tasks again inside the same experiment.

## SkillsBench

Test cases are the pytest functions in `verifier/test_outputs.py`. The runner copies that file, rewrites `/root` to the task workspace, and runs pytest. It does not run `verifier/test.sh` (that script installs packages on the machine). A task passes when every collected test passes. Oracle solutions are not copied.

```bash
RUN=exp1
git submodule update --init -- benchmark/skillsbench
python benchmark/scripts/skillsbench.py list --start 0 --end 4
python benchmark/scripts/skillsbench.py run --run "$RUN" --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary "benchmark/experiments/skillsbench/$RUN/summary.json" \
  --report "benchmark/experiments/skillsbench/$RUN/summary.txt"
python benchmark/scripts/skillsbench.py summarize --run "$RUN"
python benchmark/scripts/skillsbench.py summarize --run "$RUN" --turns 3,6
```

Change `RUN` to start another experiment. The second `summarize` reports success@3 and success@6 from the grades already stored for that run. Those k values do not have to be members of the `--turns` list used by `run`.

## ScienceAgentBench

The git checkout has no task sheet and no datasets. `fetch --split NAME` writes that Hugging Face split into `benchmark/scienceagentbench/tasks.jsonl` and leaves any other split already in the file. Task ids are `<split>-<index>`. `list --split NAME` then numbers those rows from 0. Use `train` or `test` when the dataset publishes them; this checkout's sheet is the `verified` split until another split is fetched. The dataset zip is published by that benchmark (password `scienceagentbench`, see its Benchmark Access section). Do not commit the zip or the unpacked files.

Micro cases are `program_saved`, `program_ran`, and `output_saved`. If an unpacked eval module's `eval()` returns a map of criterion to bool, each criterion is a case. A single success flag becomes `eval_passed`. The selected dataset directory is copied into the experiment when it has been unpacked. Gold programs stay out of the agent workspace.

```bash
RUN=exp1
git submodule update --init -- benchmark/scienceagentbench
HF_ENDPOINT=https://hf-mirror.com uv run --no-project --python .venv/bin/python --with 'datasets>=2,<5' \
  python benchmark/scripts/scienceagentbench.py fetch --split verified
python benchmark/scripts/scienceagentbench.py unpack \
  --zip benchmark/scienceagentbench/benchmark_verified.zip
python benchmark/scripts/scienceagentbench.py list --split verified --start 0 --end 9
python benchmark/scripts/scienceagentbench.py run --run "$RUN" --split verified --start 0 --end 9 --turns 1,2,4 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary "benchmark/experiments/scienceagentbench/$RUN/summary.json" \
  --report "benchmark/experiments/scienceagentbench/$RUN/summary.txt"
python benchmark/scripts/scienceagentbench.py summarize --run "$RUN" \
  --summary "benchmark/experiments/scienceagentbench/$RUN/summary.json" \
  --report "benchmark/experiments/scienceagentbench/$RUN/summary.txt"
```

`datasets` is not in the frozen Hermes lock, so `fetch` installs it for that command only. Drop `HF_ENDPOINT` when `huggingface.co` is reachable. Fetching `validation`, `train`, or `test` the same way keeps the `verified` rows. `--split` on `fetch` must be a split the dataset publishes.

## PaperBench

Papers are Git LFS objects in `benchmark/frontier-evals`. The upstream LFS config skips `project/paperbench/data/**`, so `pull` downloads the paper trees after the submodule is checked out. `--split` is `debug`, `lite`, `dev`, `human`, `testing`, or `all`. `debug` is only `rice`. `--task` and `--split` intersect. `rubric.json` is copied into the task replica and kept out of the agent workspace.

Micro cases are the rubric leaves. A leaf passes only when `benchmark/experiments/paperbench/<run>/agent/<task-id>/judge.json` contains `{"leaves": {"<leaf-id>": true}}`. Until that file exists the leaf is unjudged and counts as not passed (`unjudged_cases` in the summary). `regrade` reads `judge.json` and refreshes the last turn.

```bash
RUN=exp1
git submodule update --init -- benchmark/frontier-evals
python benchmark/scripts/paperbench.py pull
python benchmark/scripts/paperbench.py list --split lite
python benchmark/scripts/paperbench.py run --run "$RUN" --split lite --start 0 --end 2 --turns 1,4,8 \
  --provider qwen-31 --model Qwen/Qwen3.6-27B \
  --summary "benchmark/experiments/paperbench/$RUN/summary.json" \
  --report "benchmark/experiments/paperbench/$RUN/summary.txt"
# write benchmark/experiments/paperbench/$RUN/agent/<task-id>/judge.json, then:
python benchmark/scripts/paperbench.py regrade --run "$RUN" \
  --summary "benchmark/experiments/paperbench/$RUN/summary.json" \
  --report "benchmark/experiments/paperbench/$RUN/summary.txt"
```

If `paper.md` is still an LFS pointer, `run` records an error for that task and a later `run` retries it after `pull`.

## Terminal-Bench

Test cases are the pytest functions in `tests/test_outputs.py`. The runner rewrites `/app` to the workspace and `/tests` to the replicated tests directory. It does not run `tests/test.sh`. Reference solutions are not copied. Some tasks generate inputs from a setup script that is left in the workspace; the prompt tells the agent to run it.

```bash
RUN=exp1
git submodule update --init -- benchmark/terminal-bench
python benchmark/scripts/terminal_bench.py list --start 0 --end 4
python benchmark/scripts/terminal_bench.py run --run "$RUN" --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.8-27B \
  --summary "benchmark/experiments/terminal-bench/$RUN/summary.json" \
  --report "benchmark/experiments/terminal-bench/$RUN/summary.txt"
python benchmark/scripts/terminal_bench.py summarize --run "$RUN" \
  --summary "benchmark/experiments/terminal-bench/$RUN/summary.json" \
  --report "benchmark/experiments/terminal-bench/$RUN/summary.txt"
```

Official container graders remain available beside these scripts: SkillsBench `verifier/test.sh`, Terminal-Bench Harbor, PaperBench's judge, and ScienceAgentBench's eval programs. The summaries above are what these runners record from the workspace after each user message.
