# Changelog

## 0.1.0 — 2026-05-13

Initial refactor from a flat notebook directory into a proper `invoke`-driven
analysis repo. Base commit: `b939a26`.

### Added

- `src/mario_learning/` package with one module per analysis: `loader`,
  `descriptive`, `learning_curves`, `summary`, `scene_performance`,
  `pattern_difficulty`, `clustering`, `traces`, `survival`,
  `distribution_distances`, `compare`. Shared helpers in `utils`, `provenance`,
  `plots`.
- `tasks.py` with 10 per-dataset and 9 cross-dataset `inv compare-*` tasks.
- `setup.sh` bootstrapper for `uv`-based venv + interactive `config.yaml` gen.
- `config.yaml.template` driving paths, dataset registry, and per-task knobs.
- JSON provenance sidecars next to every output (script, git commit, params,
  inputs digest, optional seed). Idempotency keyed on the sidecar.
- Repo-local clip loader: walks `gamelogs/*_summary.json`, builds a per-clip
  DataFrame, caches as parquet, supports `--subject/--session/--run` filters.
- New analytical primitives beyond the original notebooks:
  - **`survival`**: per-scene Kaplan-Meier curves, both within-clip (death) and
    attempts-to-mastery (first_clear) framings.
  - **`distribution_distances`**: 1-D Wasserstein on scalar metrics + EMD on
    scene-space failure vectors (UMAP-Euclidean or Jaccard ground metric).
- Pytest smoke tests on a synthetic fixture dataset.
- `README.md`, `TASKS.md`, `CHANGELOG.md`.

### Changed

- `.gitignore` extended to cover `.venv/`, `data/`, `logs/`, `config.yaml`,
  Python caches.

### Removed

- `requirements.txt` (replaced by `pyproject.toml`).

### Archived

- The original exploratory notebooks moved under `notebooks/archive/` (kept in
  git for reference until each task is validated against its source notebook).
