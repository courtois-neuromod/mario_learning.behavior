# mario_learning.behavior

Behavioral analyses for any **mario.scenes-formatted** BIDS dataset (canonical
human-played `mario.scenes`, agent-produced clip sets, simulator runs, …),
with cross-dataset comparison built in.

Every analysis runs through `invoke` tasks. Configuration is centralized in
`config.yaml`; every output file is paired with a JSON sidecar recording the
git commit, parameters, input fingerprints and (when applicable) the random
seed used to produce it.

## Install

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd mario_learning.behavior
./setup.sh                # creates .venv/, installs the package, prompts for
                          # the path to mario.scenes and your SLURM user
source .venv/bin/activate
inv --list                # shows every task
```

## Configure

`setup.sh` generates `config.yaml` from `config.yaml.template` and asks for
your `mario.scenes` path. To add an agent-produced dataset, edit `config.yaml`:

```yaml
datasets:
  humans:
    path: /data/mario.scenes
  agent_ppo:
    path: /data/agent_ppo.scenes      # same BIDS layout
```

The dataset just needs `sub-*/ses-*/gamelogs/*_summary.json` (per-clip metadata
written by `mario.scenes/code/generate_clips`); optional `_variables.json` is
required only for the `traces` task.

## Run

Per-dataset analyses:

```bash
inv load --dataset humans                       # build cached clips parquet
inv descriptive --dataset humans                # per-subject / scene / phase / level tables + QC
inv learning-curves --dataset humans            # moving-average performance vs ClipCode
inv summary --dataset humans                    # per-level grids of Cleared/Duration/Hits
inv scene-performance --dataset humans          # per-scene trajectory grids
inv pattern-difficulty --dataset humans         # success rate by scene pattern + learning metric
inv clustering --dataset humans                 # UMAP scene-space + KMeans clusters
inv traces --dataset humans                     # x/y traces overlaid on level/scene PNGs
inv survival --dataset humans                   # Kaplan-Meier per-scene curves
inv distribution-distances --dataset humans     # Wasserstein-1 on scalars + scene-space
```

Comparison tasks (run the per-dataset task first for each dataset, then):

```bash
inv compare-descriptive --datasets humans,agent_ppo
inv compare-learning-curves --datasets humans,agent_ppo
inv compare-scene-performance --datasets humans,agent_ppo
inv compare-pattern-difficulty --datasets humans,agent_ppo
inv compare-clustering --datasets humans,agent_ppo
inv compare-summary --datasets humans,agent_ppo
inv compare-traces --datasets humans,agent_ppo
inv compare-survival --datasets humans,agent_ppo
inv compare-distribution-distances --datasets humans,agent_ppo
```

Every task accepts `--subject sub-01,sub-02 --session ses-001 --run run-01` to
narrow scope, and `--force` to bypass the sidecar-driven idempotency cache.

See `TASKS.md` for per-task details and `.AGENTS.md` for the project's
operating principles.

## Layout

```
src/mario_learning/        # analysis modules (one file per task)
tasks.py                   # invoke entry points
config.yaml                # paths + per-task parameters (gitignored)
data/processed/            # cached outputs per task per dataset (gitignored)
tests/                     # pytest smoke tests on a fixture dataset
notebooks/archive/         # the original exploratory notebooks
```

## Tests

```bash
pytest -q
ruff check src tests tasks.py
```
