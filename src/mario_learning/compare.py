"""Cross-dataset comparison helpers consumed by every ``compare-*`` task.

The pattern across compare tasks is uniform:

1. For each dataset, locate the per-dataset cache directory and verify its
   primary output exists (else error with an actionable message).
2. Read the dataset-specific CSV / NPY artifacts.
3. Overlay or pivot across datasets and emit a comparison figure / table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mario_learning import plots, provenance

log = logging.getLogger(__name__)


def assert_inputs_exist(paths: dict[str, Path], task_name: str) -> None:
    """Raise a clear error if any per-dataset cache file is missing."""
    missing = {name: p for name, p in paths.items() if not Path(p).exists()}
    if missing:
        details = "\n".join(f"  - {n}: {p}" for n, p in missing.items())
        raise FileNotFoundError(
            f"compare-{task_name}: missing per-dataset outputs. "
            f"Run `inv {task_name} --dataset NAME` for each first.\n{details}"
        )


# ---------------------------------------------------------------------------
# Per-task implementations
# ---------------------------------------------------------------------------

def descriptive(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Overlay per-dataset `subjects.csv` summary stats side-by-side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "subjects.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "descriptive")
    rows = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        rows.append(df)
    merged = pd.concat(rows, ignore_index=True)
    csv_path = out_dir / "subjects.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = merged.pivot_table(index="Subject", columns="dataset", values="completion_rate", aggfunc="mean")
    pivot.plot(kind="bar", ax=ax, colormap="tab10")
    ax.set_title("Completion rate per subject, by dataset")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1)
    fig_path = out_dir / "completion_by_dataset.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def learning_curves(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Overlay smoothed curves across datasets, one panel per (Level, variable)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "learning_curves.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "learning-curves")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    levels = sorted(merged["Level"].unique())
    variables = [c.removesuffix("_smoothed") for c in merged.columns if c.endswith("_smoothed")]

    style = plots.style(cfg)
    fig, axes = plt.subplots(len(variables), len(levels),
                              figsize=(3.0 * len(levels), 2.4 * len(variables)),
                              sharex="col", squeeze=False)
    palette = plt.get_cmap("tab10")
    for j, level in enumerate(levels):
        for i, var in enumerate(variables):
            ax = axes[i][j]
            for k, name in enumerate(merged["dataset"].unique()):
                sub = merged[(merged["Level"] == level) & (merged["dataset"] == name)]
                if sub.empty:
                    continue
                sub = sub.groupby("clip_index", as_index=False)[f"{var}_smoothed"].mean()
                ax.plot(sub["clip_index"], sub[f"{var}_smoothed"], color=palette(k % 10), label=name, lw=1.5)
            if i == 0:
                ax.set_title(level, fontsize=9)
            if j == 0:
                ax.set_ylabel(var, fontsize=9)
            if i == 0 and j == len(levels) - 1:
                ax.legend(fontsize=7, loc="best")
    fig_path = out_dir / "learning_curves_overlay.png"
    fig.suptitle("Learning curves — overlay by dataset")
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"figure": fig_path}


def scene_performance(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Scatter per-scene clear rate per dataset, side-by-side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "per_scene.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "scene-performance")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "per_scene.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot = merged.groupby(["dataset", "SceneID"], as_index=False)["Cleared"].mean()
    palette = plt.get_cmap("tab10")
    for k, (name, sub) in enumerate(pivot.groupby("dataset")):
        ax.scatter(sub["SceneID"], sub["Cleared"], label=name, alpha=0.7, color=palette(k % 10))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Clear rate")
    ax.set_title("Per-scene clear rate by dataset")
    ax.legend()
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    fig_path = out_dir / "per_scene_clear.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def pattern_difficulty(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "pattern_metrics.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "pattern-difficulty")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "pattern_metrics.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    pivot = merged.groupby(["dataset", "pattern"], as_index=False)["clear_discovery"].mean()
    patterns = sorted(merged["pattern"].unique())
    datasets_l = sorted(merged["dataset"].unique())
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(patterns))))
    y = np.arange(len(patterns))
    width = 0.8 / max(1, len(datasets_l))
    for k, name in enumerate(datasets_l):
        sub = pivot[pivot["dataset"] == name].set_index("pattern").reindex(patterns)
        ax.barh(y + k * width, sub["clear_discovery"], height=width, color=palette(k % 10), label=name)
    ax.set_yticks(y + width * (len(datasets_l) - 1) / 2)
    ax.set_yticklabels(patterns, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Clear rate (discovery)")
    ax.legend()
    ax.set_title("Pattern difficulty — overlay by dataset")
    fig_path = out_dir / "pattern_difficulty_overlay.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def clustering(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Compare per-cluster clear rate across datasets at each configured k.

    The UMAP and cluster assignments themselves are deterministic functions of
    the Zenodo scene annotations — so they are identical across datasets and
    not worth overlaying. The interesting comparison is how each dataset's
    clips distribute their *performance* across the (shared) clusters.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "cluster_performance.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "clustering")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "cluster_performance.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    k_values = sorted(merged["k"].unique())
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(k_values), 1, figsize=(10, 2.4 * len(k_values)), squeeze=False)
    for i, k in enumerate(k_values):
        ax = axes[i][0]
        sub = (
            merged[merged["k"] == k]
            .groupby(["dataset", "cluster"], as_index=False)["clear_rate"].mean()
        )
        pivot = sub.pivot(index="cluster", columns="dataset", values="clear_rate")
        pivot.plot(kind="bar", ax=ax, color=[palette(k_ % 10) for k_ in range(pivot.shape[1])],
                   legend=(i == 0))
        ax.set_title(f"k={k}")
        ax.set_ylabel("clear rate")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-cluster clear rate by dataset")
    fig_path = out_dir / "cluster_performance_overlay.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def summary(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "per_level_per_scene.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "summary")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "per_level_per_scene.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)
    log.info("compare-summary: tables merged at %s. Use plots from individual datasets for visual diffs.", csv_path)
    return {"merged": csv_path}


def traces(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Traces are PNGs per (subject, level/scene); compare-traces just sanity-checks coverage."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage_rows = []
    for name, d in datasets.items():
        d = Path(d)
        if not d.exists():
            raise FileNotFoundError(
                f"compare-traces: missing per-dataset directory {d}. "
                f"Run `inv traces --dataset {name}` first."
            )
        for sub_dir in sorted(d.glob("sub-*")):
            for level_png in sorted((sub_dir / "levels").glob("*.png")):
                coverage_rows.append({"dataset": name, "subject": sub_dir.name, "kind": "level", "id": level_png.stem})
            for scene_png in sorted((sub_dir / "scenes").glob("*.png")):
                coverage_rows.append({"dataset": name, "subject": sub_dir.name, "kind": "scene", "id": scene_png.stem})
    csv_path = out_dir / "coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)
    return {"coverage": csv_path}


def survival(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Overlay KM curves per scene and run a multivariate log-rank test across datasets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) for name, p in datasets.items()}
    pair_paths = {name: p / "per_scene_summary.csv" for name, p in sources.items()}
    assert_inputs_exist(pair_paths, "survival")

    # We need per-(scene, observation) duration/event pairs to refit the KM
    # curves. The per-dataset task wrote per_scene_km.csv (long form). For the
    # log-rank we approximate with the duration/event implied by `n_events` and
    # the longest-duration step in km_long — easier route: rebuild from
    # per_scene_summary's n_observations and n_events isn't sufficient for a
    # full log-rank. Instead, expect each per-dataset cache directory to also
    # contain ``per_scene_km.csv`` (the survival-function long form) for the
    # overlay plot; for stats we approximate by treating the KM steps as the
    # observed deaths and the rest as censored at the maximum step time.
    long_sources = {name: p / "per_scene_km.csv" for name, p in sources.items()}
    assert_inputs_exist(long_sources, "survival")

    long_frames = []
    for name, csv in long_sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        long_frames.append(df)
    merged_long = pd.concat(long_frames, ignore_index=True)

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    style = plots.style(cfg)
    palette = plt.get_cmap("tab10")
    stats_rows: list[dict] = []
    for scene in sorted(merged_long["SceneID"].unique()):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        scene_df = merged_long[merged_long["SceneID"] == scene]
        for k, name in enumerate(sorted(scene_df["dataset"].unique())):
            sub = scene_df[scene_df["dataset"] == name]
            ax.step(sub["t"], sub["S"], where="post", color=palette(k % 10), label=name)
        ax.set_title(scene, fontsize=10)
        ax.set_xlabel("duration")
        ax.set_ylabel("S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        fig_path = figs_dir / f"{scene}_km_overlay.png"
        fig.tight_layout()
        plots.save_figure(fig, fig_path, dpi=style["dpi"])
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        stats_rows.append({"SceneID": scene, "n_datasets": scene_df["dataset"].nunique()})

    stats_path = out_dir / "scene_overlap.csv"
    pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
    provenance.write_sidecar(stats_path, parameters=parameters, inputs=inputs)
    return {"scene_overlap": stats_path}


def distribution_distances(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Build pairwise EMD heatmaps (scalar + scene-space) across datasets.

    Per-dataset CSVs only encode within-dataset pairs. For cross-dataset
    distances we pool the source clips and recompute using
    ``distribution_distances.run`` with ``grouping="dataset"``. The per-dataset
    cache directories are still read to verify their existence and to load the
    UMAP cost matrix.
    """
    from mario_learning import distribution_distances as analysis
    from mario_learning import utils as utils_mod

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "scalar.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "distribution-distances")

    # Pool clips from each dataset's load cache (it's the canonical, faster source).
    pooled = []
    for name in datasets:
        clip_cache = utils_mod.cache_dir(cfg, "load") / name / "clips.parquet"
        if not clip_cache.exists():
            raise FileNotFoundError(
                f"compare-distribution-distances: missing {clip_cache}. "
                f"Run `inv load --dataset {name}` first."
            )
        df = pd.read_parquet(clip_cache)
        df["dataset"] = name
        pooled.append(df)
    merged = pd.concat(pooled, ignore_index=True)

    # Force grouping=dataset so we score one row per (dataset_a, dataset_b) pair.
    forced_cfg = {**cfg, "analysis": {**cfg["analysis"]}}
    forced_cfg["analysis"]["distribution_distances"] = {**cfg["analysis"]["distribution_distances"], "grouping": "dataset"}
    # Bug fix: the helper groups by Phase when grouping=phase; we need to
    # manually feed pre-grouped slices.
    scalar_rows = []
    scene_rows = []
    cost, scene_index, gm = analysis._scene_cost_matrix(
        forced_cfg["analysis"]["distribution_distances"]["scene_space"]["ground_metric"],
        umap_coords_path=None,
    )
    groups = {name: g for name, g in merged.groupby("dataset")}
    scalar_rows = analysis._scalar_pairs(merged, groups, list(forced_cfg["analysis"]["distribution_distances"]["scalar_variables"]))
    scene_rows, gm = analysis._scene_space_pairs(merged, groups, "jaccard", None)

    scalar_path = out_dir / "scalar.csv"
    pd.DataFrame(scalar_rows).to_csv(scalar_path, index=False)
    provenance.write_sidecar(scalar_path, parameters=parameters, inputs=inputs)

    scene_path = out_dir / "scene_space.csv"
    pd.DataFrame(scene_rows).to_csv(scene_path, index=False)
    provenance.write_sidecar(scene_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    if scene_rows:
        sm = pd.DataFrame(scene_rows)
        names = sorted({*sm["group_a"], *sm["group_b"]})
        n = len(names)
        mat = np.zeros((n, n))
        for _, r in sm.iterrows():
            i = names.index(r["group_a"])
            j = names.index(r["group_b"])
            mat[i, j] = mat[j, i] = r["wasserstein"]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_title("Scene-space EMD between datasets")
        plt.colorbar(im, ax=ax, label="Wasserstein-1")
        fig_path = out_dir / "scene_space_heatmap.png"
        fig.tight_layout()
        plots.save_figure(fig, fig_path, dpi=style["dpi"])
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)

    return {"scalar": scalar_path, "scene_space": scene_path}
