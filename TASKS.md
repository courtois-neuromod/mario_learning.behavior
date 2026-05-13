# Tasks reference

Every task accepts the common flags:

| Flag | Description |
|---|---|
| `--dataset NAME` | Dataset name from `config.yaml > datasets`. **Required** for per-dataset tasks. |
| `--datasets A,B,...` | Comma-separated dataset names. Default: all configured. Used by `compare-*` tasks. |
| `--subject sub-XX[,sub-YY]` | Restrict to one or more BIDS subjects. Default: all. |
| `--session ses-XXX[,ses-YYY]` | Restrict to one or more BIDS sessions. Default: all. |
| `--run run-XX[,run-YY]` | Restrict to one or more BIDS runs (filtered post-DataFrame). Default: all. |
| `--force` | Re-run even if the output's JSON sidecar matches the request. |

All tasks emit JSON sidecars next to every output, capturing the git commit,
parameters, and inputs hash. Idempotency is sidecar-based.

---

## Per-dataset tasks

### `inv load --dataset NAME`

Walks `sub-*/ses-*/gamelogs/*_summary.json`, builds a per-clip DataFrame
(see schema in `src/mario_learning/loader.py`), and caches it as
`data/processed/load/{name}/clips.parquet`. Re-runs skip unchanged datasets.
Every other task implicitly calls this on entry.

### `inv descriptive --dataset NAME`

Emits `subjects.csv`, `scenes.csv`, `phases.csv`, `levels.csv` aggregations
and a `desc-qc.png` summary panel into `data/processed/descriptive/{name}/`.

### `inv learning-curves --dataset NAME`

Moving-average curves over ClipCode-ordered clips, per (Subject, Level,
Variable). Window and variables come from `config > analysis.learning_curves`.
Outputs `learning_curves.csv` + one `figures/sub-{s}_learning_curves.png`
per subject.

### `inv summary --dataset NAME`

Per-level facet figures of Cleared/Duration/Hits across the level's scenes,
with subject and phase overlays. One PNG per level under `figures/`.

### `inv scene-performance --dataset NAME`

`per_scene.csv` aggregates plus a `figures/sub-{s}_{level}_scene_perf.png`
trajectory grid per (Subject, Level).

### `inv pattern-difficulty --dataset NAME`

Joins clips to 27 binary scene-pattern features (auto-downloaded from
Zenodo) and computes clear-rate and a learning metric per pattern, per
subject. Outputs `pattern_metrics.csv` + per-subject bar plots.

### `inv clustering --dataset NAME`

UMAP of scene annotations (params from `config > clustering.umap`), Jaccard
and UMAP-Euclidean distance matrices, K-means at every configured `k`, plus
per-cluster performance aggregates. Outputs `umap_2d.csv`,
`distance_*.npy`, `scenes_clustered.csv`, `cluster_performance.csv`, two
figures.

### `inv traces --dataset NAME`

Reads `_variables.json` companions of each clip, extracts x/y player
positions, and overlays them on level and scene background PNGs from
`config.analysis.traces.background_source`. Outputs
`sub-{s}/levels/{level}.png` and `sub-{s}/scenes/{scene_id}.png`. Use
`--subject/--session/--run` to keep figure runtimes reasonable.

### `inv survival --dataset NAME`

Per-scene Kaplan-Meier curves. Two framings:

- `config.analysis.survival.event: death` (default) — within-clip
  frames-to-death, censored at clip end. Curves describe how mortality
  risk distributes over the clip.
- `event: first_clear` — attempts until first clear, per (Subject, SceneID).
  Curves describe time-to-mastery.

Outputs `per_scene_km.csv` (survival function points), `per_scene_summary.csv`
(median survival, n, n_events), and one `figures/{level}_km.png` per level.

### `inv distribution-distances --dataset NAME`

Wasserstein-1 distances between performance distributions across groups
within the dataset. Two modes:

- **scalar** — 1-D EMD on each configured variable
  (`scalar_variables`: default Duration, X_Traveled, Hits_taken).
- **scene-space** — EMD on per-scene failure-rate vectors with a scene-space
  ground metric (UMAP-Euclidean if `clustering` cache available;
  otherwise Jaccard on the 27-feature annotations).

Groups are derived from `grouping`: `phase` (discovery vs practice),
`subject` (all pairs), `early_late` (per subject), or `dataset` (used by
the compare task).

---

## Cross-dataset comparison tasks

Each `compare-*` task reads from the per-dataset cache directories under
`data/processed/{task}/{name}/`. **Run the per-dataset task first** for every
dataset you want to include — the compare task errors with a clear message
if any input cache is missing.

| Task | What it does |
|---|---|
| `compare-descriptive` | Concat per-subject tables and plot completion rate by dataset. |
| `compare-learning-curves` | Overlay smoothed curves per (Level, Variable). |
| `compare-scene-performance` | Per-scene clear rate scatter, colored by dataset. |
| `compare-pattern-difficulty` | Grouped horizontal bars of clear-rate by pattern. |
| `compare-clustering` | Overlay UMAP scatters (note: embeddings are computed per dataset, axes are not aligned). |
| `compare-summary` | Concat per-level/per-scene tables for downstream analysis. |
| `compare-traces` | Coverage table over the per-dataset trace PNGs. |
| `compare-survival` | Per-scene KM curve overlays + a scene-coverage table. |
| `compare-distribution-distances` | Pairwise scalar + scene-space EMD between datasets, with heatmap. |

---

## SLURM defaults

`--slurm` is not yet wired into the dispatch (single-machine execution only at
this point). Resource defaults live in `config > slurm.defaults`:

```yaml
slurm:
  username: <user>
  defaults:
    mem: 8G
    time: "01:00:00"
    cpus: 2
```

When SLURM dispatch is added, every task will use these as the per-job
defaults; tasks that need more (e.g. `clustering`, `traces`) will override.
