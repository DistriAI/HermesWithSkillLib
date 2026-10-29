# Baselines

Third-party systems compared with SkillLib. Each one is a git submodule. Hermes stays the agent: the baseline exposes tools, and the benchmark runner selects them with `--skill-mode`.

## SkillRouter

[zhengyanzhao1997/SkillRouter](https://github.com/zhengyanzhao1997/SkillRouter). A retrieve-and-rerank skill router. The agent calls `skill_router_find` and `skill_router_load`.

```bash
git submodule update --init baselines/SkillRouter
```

The parent repository indexes the submodule commit. It does not index the skill-pool shards, the model weights, or the embedding cache:

| Path | Indexed |
|---|---|
| `baselines/SkillRouter` gitlink | Yes, the upstream commit |
| `baselines/SkillRouter/data/eval_core/easy/` and `hard/` | No. SkillRouter gitignores the downloaded shards |
| `baselines/SkillRouter/models/` | No. Weights download on first use and stay in the worktree |
| `baselines/SkillRouter/outputs/` | No. SkillRouter gitignores this cache |

Download the pool and the Python packages before a `--skill-mode router` run. The commands are in [`benchmark/README.md`](../benchmark/README.md). Optional `baselines/skillrouter.json` sets `tier`, `encoder`, `reranker`, and `device`. That file is local configuration; leave it untracked when it only repeats the defaults.
