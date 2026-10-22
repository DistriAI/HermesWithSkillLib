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
tasks/<task-id>/             copy of that task, including grader files
agent/<task-id>/workspace/   files that task's agent writes
agent/<task-id>/result.json  grades, time, and tokens
agent/<task-id>/turns/<n>/   prompt and usage for user-message turn n
summary.json
summary.txt
```

The profile copy includes `config.yaml`, `.env`, and other small profile text files. Session databases from the source profile are not copied. Later tasks in the same run share the sessions this run creates under that one `.hermes`. A different `--run` name gets a different directory.

## Options

`list` and `run` select tasks the same way. `list` prints `index<TAB>task-id` so the indexes match `--start` and `--end`.

```bash
python benchmark/scripts/skillsbench.py list --start 0 --end 4
python benchmark/scripts/skillsbench.py run --run exp1 --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary benchmark/experiments/skillsbench/exp1/summary.json \
  --report benchmark/experiments/skillsbench/exp1/summary.txt
```

| Option | Usage |
|---|---|
| `--run NAME` | Experiment directory name. Required for `run`, `summarize`, and `regrade`. |
| `--split NAME` | Subset whose order defines the indexes. PaperBench: `debug`, `lite`, `dev`, `human`, `testing`, `all` (the order in that split file). ScienceAgentBench: a `split` value in `tasks.jsonl`, such as `verified` or `validation`, and `train` or `test` when those rows have been fetched. SkillsBench and Terminal-Bench have no named splits; omit it or pass `all`. |
| `--start N` | First index in that split. 0-based, inclusive. Default: 0 when `--end` is set. |
| `--end N` | Last index in that split. 0-based, inclusive. Default: the last task when `--start` is set. `--start 0 --end 9` is the first 10 tasks. |
| `--task ID` | One id, or a comma-separated list, kept only when it also falls in the index range. |
| `--limit N` | After the index range, keep at most N tasks. |
| `--turns 1,2,4,8` | On `run`, user-message budgets. The agent runs every turn up to the largest budget and stores a grade after each turn. Default `1,2,4,8`. On `summarize` and `regrade`, the k values to report from those stored grades. |
| `--skill-mode library\|task\|none` | Default `library`. |
| `--model ID` | Model id, such as `Qwen/Qwen3.6-27B`. With `--provider` and no model, that provider's `default_model` is used. |
| `--provider NAME` | A name under `providers:` in the replicated profile, such as `qwen-local` or `qwen-31`. A model without this flag keeps `model.provider` from the profile. |
| `--timeout SECONDS` | Wall-clock limit for one task, across all of its user turns. |
| `--hermes-home PATH` | Profile to copy into the experiment. Default: `$HERMES_HOME`, otherwise `<repo>/.hermes`. |
| `--fresh` | Run tasks again even when their `result.json` status is `complete`. |
| `--summary PATH` | JSON summary. Default: `<experiment>/summary.json`. |
| `--report PATH` | Text summary. Default: `<experiment>/summary.txt`. |

`summarize --run NAME` reads each finished task's `result.json` and recomputes the rates. `--turns 3` or `--turns 3,6` reports those k values even when the original run used a different list, such as `--turns 1,2,4,8`. `regrade --run NAME` grades the last turn again without calling the model. Both accept `--summary` and `--report`.

`--model` and `--provider` are stored in `run.json`, each task's `result.json`, and the summary. Omit both and the agent uses `model.default` and `model.provider` from the copied profile.

| Skill mode | What the agent gets |
|---|---|
| `library` | Task statement and inputs. The prompt tells the agent to call `skill_library_find` and `skill_library_load` (at most three). |
| `task` | SkillsBench `environment/skills/` is copied in. ScienceAgentBench `domain_knowledge` is included. The SkillLib tools are off. |
| `none` | Statement and workspace only. The SkillLib tools are off. |

## Metrics

`--turns` is a comma-separated list of user-message budgets, default `1,2,4,8`. Turn 1 sends the task prompt. Each later turn resumes that session and asks the agent to continue. The script grades the workspace after every turn and stops early when every test case passes. The agent runs up to the largest budget.

| Field | Meaning |
|---|---|
| `macro_success@k` | Fraction of completed tasks that passed every test case on any stored turn at or before k. A pass at turn 2 still counts at k = 4 when the task stopped there, and it still counts when a later turn failed. |
| `micro_success@k` | Fraction of test cases that passed on any stored turn at or before k. |
| `macro_auc@k`, `micro_auc@k` | Mean of success@1 through success@k. Each of those rates uses the same at-or-before rule. |
| `final_macro_success`, `final_micro_success` | Pass rate on each task's last turn. |
| `execution_time_sec` | Mean and sample standard deviation of wall time, for tasks that passed on the last turn, and for every completed task. |
| `input_tokens`, `output_tokens`, `total_tokens` | The same two groups. Counts add the main request and auxiliary calls. |

One completed task has standard deviation 0. Interrupted tasks are left out of the rates until a rerun finishes them.

Re-running the same `--run` command skips a task whose `result.json` status is `complete`. Status `running`, `timeout`, or `error` starts that task over. `--fresh` runs completed tasks again inside the same experiment.

## SkillsBench

Test cases are the pytest functions in `verifier/test_outputs.py`. The runner copies that file, rewrites `/root` to the task workspace, and runs pytest. It does not run `verifier/test.sh` (that script installs packages on the machine). A task passes when every collected test passes. Oracle solutions are not copied.

```bash
git submodule update --init -- benchmark/skillsbench
python benchmark/scripts/skillsbench.py list --start 0 --end 4
python benchmark/scripts/skillsbench.py run --run exp1 --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary benchmark/experiments/skillsbench/exp1/summary.json \
  --report benchmark/experiments/skillsbench/exp1/summary.txt
python benchmark/scripts/skillsbench.py summarize --run exp1
python benchmark/scripts/skillsbench.py summarize --run exp1 --turns 3,6
```

The second `summarize` reports success@3 and success@6 from the grades already stored for `exp1`. Those k values do not have to be members of the `--turns` list used by `run`.

## ScienceAgentBench

The git checkout has no task sheet and no datasets. `fetch --split NAME` writes that Hugging Face split into `benchmark/scienceagentbench/tasks.jsonl` and leaves any other split already in the file. Task ids are `<split>-<index>`. `list --split NAME` then numbers those rows from 0. Use `train` or `test` when the dataset publishes them; this checkout's sheet is the `verified` split until another split is fetched. The dataset zip is published by that benchmark (password `scienceagentbench`, see its Benchmark Access section). Do not commit the zip or the unpacked files.

Micro cases are `program_saved`, `program_ran`, and `output_saved`. If an unpacked eval module's `eval()` returns a map of criterion to bool, each criterion is a case. A single success flag becomes `eval_passed`. The selected dataset directory is copied into the experiment when it has been unpacked. Gold programs stay out of the agent workspace.

```bash
git submodule update --init -- benchmark/scienceagentbench
HF_ENDPOINT=https://hf-mirror.com uv run --no-project --python .venv/bin/python --with 'datasets>=2,<5' \
  python benchmark/scripts/scienceagentbench.py fetch --split verified
python benchmark/scripts/scienceagentbench.py unpack \
  --zip benchmark/scienceagentbench/benchmark_verified.zip
python benchmark/scripts/scienceagentbench.py list --split verified --start 0 --end 9
python benchmark/scripts/scienceagentbench.py run --run exp1 --split verified --start 0 --end 9 --turns 1,2,4 \
  --provider qwen-local --model Qwen/Qwen3.6-27B \
  --summary /tmp/scienceagentbench.json --report /tmp/scienceagentbench.txt
python benchmark/scripts/scienceagentbench.py summarize --run exp1 \
  --summary /tmp/scienceagentbench.json --report /tmp/scienceagentbench.txt
```

`datasets` is not in the frozen Hermes lock, so `fetch` installs it for that command only. Drop `HF_ENDPOINT` when `huggingface.co` is reachable. Fetching `validation`, `train`, or `test` the same way keeps the `verified` rows. `--split` on `fetch` must be a split the dataset publishes.

## PaperBench

Papers are Git LFS objects in `benchmark/frontier-evals`. The upstream LFS config skips `project/paperbench/data/**`, so `pull` downloads the paper trees after the submodule is checked out. `--split` is `debug`, `lite`, `dev`, `human`, `testing`, or `all`. `debug` is only `rice`. `--task` and `--split` intersect. `rubric.json` is copied into the task replica and kept out of the agent workspace.

Micro cases are the rubric leaves. A leaf passes only when `benchmark/experiments/paperbench/<run>/agent/<task-id>/judge.json` contains `{"leaves": {"<leaf-id>": true}}`. Until that file exists the leaf is unjudged and counts as not passed (`unjudged_cases` in the summary). `regrade` reads `judge.json` and refreshes the last turn.

```bash
git submodule update --init -- benchmark/frontier-evals
python benchmark/scripts/paperbench.py pull
python benchmark/scripts/paperbench.py list --split lite
python benchmark/scripts/paperbench.py run --run exp1 --split lite --start 0 --end 2 --turns 1,4,8 \
  --provider qwen-31 --model Qwen/Qwen3.6-27B \
  --summary /tmp/paperbench.json --report /tmp/paperbench.txt
# write agent/<task-id>/judge.json from the PaperBench judge, then:
python benchmark/scripts/paperbench.py regrade --run exp1 \
  --summary /tmp/paperbench.json --report /tmp/paperbench.txt
```

If `paper.md` is still an LFS pointer, `run` records an error for that task and a later `run` retries it after `pull`.

## Terminal-Bench

Test cases are the pytest functions in `tests/test_outputs.py`. The runner rewrites `/app` to the workspace and `/tests` to the replicated tests directory. It does not run `tests/test.sh`. Reference solutions are not copied. Some tasks generate inputs from a setup script that is left in the workspace; the prompt tells the agent to run it.

```bash
git submodule update --init -- benchmark/terminal-bench
python benchmark/scripts/terminal_bench.py list --start 0 --end 4
python benchmark/scripts/terminal_bench.py run --run exp1 --start 0 --end 4 --turns 1,2,4,8 \
  --provider qwen-local --model Qwen/Qwen3.8-27B \
  --summary /tmp/terminal-bench.json --report /tmp/terminal-bench.txt
python benchmark/scripts/terminal_bench.py summarize --run exp1 \
  --summary /tmp/terminal-bench.json --report /tmp/terminal-bench.txt
```

Official container graders remain available beside these scripts: SkillsBench `verifier/test.sh`, Terminal-Bench Harbor, PaperBench's judge, and ScienceAgentBench's eval programs. The summaries above are what these runners record from the workspace after each user message.
