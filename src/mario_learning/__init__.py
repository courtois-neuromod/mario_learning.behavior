"""Behavioral analyses for mario.scenes-formatted datasets.

Public entry points (called from `tasks.py`):

- `loader.load_clips(dataset_path, cache_dir, ...)` — build/refresh clip dataframe
- `descriptive.run(clips, out_dir, params)` — per-dataset descriptive stats
- `learning_curves.run(...)`, `summary.run(...)`, `scene_performance.run(...)`,
  `pattern_difficulty.run(...)`, `clustering.run(...)`, `traces.run(...)`,
  `survival.run(...)`, `distribution_distances.run(...)`
- `compare.*` — cross-dataset companions

Every analysis writes its primary output(s) alongside a JSON provenance sidecar
via `provenance.write_output`.
"""

__version__ = "0.1.0"
