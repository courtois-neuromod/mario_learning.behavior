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

## Compare every agent variant against humans

`config.yaml`'s `agent` dataset group auto-discovers every agent-produced
dataset (`agent_ppo_packnet`, `agent_dqn_vanilla`, …) — `cfg['dataset_groups']['agent']`
lists every currently-registered name. To (re)build the clear-rate / duration
/ score comparison figures and the cross-model ranking tables for **all** of
them against humans:

```bash
# 1. Per-dataset pattern-difficulty analysis (writes pattern_metrics.csv,
#    pattern_metrics_duration.csv, pattern_metrics_score.csv, stage_summary.csv
#    under output/pattern_difficulty/<name>/). Add --force to rebuild ones
#    that already exist but predate the duration/score/stage_summary outputs.
inv pattern-difficulty --dataset humans,agent

# 2. Humans-vs-one-agent comparison figures — has to run once per agent (the
#    figures assume exactly two datasets), so it's a loop. Writes
#    output/compare/<model_label>/pattern_difficulty/all_patterns_aggregate_{clear_rate,duration,score}.png
for a in $(python3 -c "
from mario_learning import utils
cfg = utils.load_config()
print(' '.join(sorted(cfg['dataset_groups']['agent'])))
"); do
  inv compare-pattern-difficulty --datasets humans,"$a"
done

# 3. Montage every model's aggregate figure into one grid per metric, plus
#    rank every model by avg/early_discovery/late_practice/delta.
#    Writes to output/compare/all_models/pattern_difficulty/:
#      all_patterns_aggregate_all_models_{clear_rate,duration,score}.png
#      model_ranking_{clear_rate,duration,score}.csv
#      model_ranking_combined.csv   (all three metrics side by side)
inv compare-all-models --metric all
```

All three steps are idempotent (cache-hit and skip if outputs already match
their sidecars) — add `--force` to any of them to force a rebuild, e.g. after
fixing bad source data for one model. Step 1 only needs to be re-run with
`--force` for datasets that predate a given output file (check
`output/pattern_difficulty/<name>/` for `stage_summary.csv` etc.); step 3
always cheaply recombines whatever step 1/2 outputs already exist, so it's
safe to re-run on its own after adding or fixing one model.

### How the averages are computed

Every "one number per stage" you see — on the aggregate figures and in
`stage_summary.csv` / `model_ranking_*.csv` — is a three-level nested mean,
not a raw per-clip average:

1. **Clip-level, per `(Subject, pattern, Stage)`**: average the raw
   `Cleared`/`Duration`/`ScoreGained` values of every clip in that cell
   (`pattern_metrics*.csv`). A clip belongs to every pattern its scene has
   flagged, so it can contribute to several cells at once.
2. **Average across patterns**, per `(Subject, Stage)`: average that
   subject's ~27 pattern-level cells. Each pattern counts once, regardless
   of how many clips it has — a pattern with 5 clips carries the same weight
   as one with 5,000.
3. **Average across subjects**, per `Stage`: average the 5 subjects'
   numbers from step 2. Each subject counts once, regardless of clip volume.

So the final per-stage number is "the average pattern, averaged over the
average subject" — not "the average clip." `avg` in `stage_summary.csv`
extends step 3 across all 6 stages instead of reporting one; `early_discovery`
and `late_practice` are just that stage's value; `delta` is
`late_practice − early_discovery` (for `duration`, a *negative* delta is the
improvement direction — faster clips — the opposite sign convention from
`clear_rate`/`score`, where positive means improvement).

Step 1 (`stage_summary.csv`, per single dataset) and step 2 (the aggregate
figures, per humans-vs-agent pair) each recompute this independently from
their own merged/unmerged tables — they use the identical formula, so the
numbers match, but neither reads the other's output.

## Layout

```
src/mario_learning/        # analysis modules (one file per task)
tasks.py                   # invoke entry points
config.yaml                # paths + per-task parameters (gitignored)
output/                    # pipeline outputs per task per dataset (gitignored)
data/external/             # auto-downloaded inputs, e.g. scenes_mastersheet.csv (gitignored)
tests/                     # pytest smoke tests on a fixture dataset
notebooks/archive/         # the original exploratory notebooks
```

By convention `data/` is for inputs (the canonical `mario.scenes` dataset
points at its repo via `config.yaml > datasets.humans.path`, and the Zenodo
scene-pattern CSV is auto-downloaded to `data/external/`) and `output/` is
for everything the pipeline produces.

## Tests

```bash
pytest -q
ruff check src tests tasks.py
```
