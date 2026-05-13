"""UMAP scene embedding + per-cluster performance aggregates.

Computes (and caches) a 2-D UMAP of the 27 binary scene features, the Jaccard
distance matrix in feature space, and the Euclidean distance matrix in UMAP
space. Then runs K-means at each configured ``k`` and aggregates clip-level
performance by cluster — mean Cleared / Duration / Hits per (Subject, cluster, k).

Outputs:
- ``umap_2d.csv``                  — DR_1, DR_2 per scene_id
- ``distance_jaccard.npy``         — pairwise Jaccard (feature space)
- ``distance_umap.npy``            — pairwise Euclidean (UMAP space)
- ``scenes_clustered.csv``         — scene_id × k columns of cluster labels
- ``cluster_performance.csv``      — (Subject, k, cluster, metric) aggregates
- ``figures/umap_clusters.png``    — UMAP scatter colored by cluster
- ``figures/cluster_performance.png`` — Cleared by cluster, per subject
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from mario_learning import plots, provenance, utils

log = logging.getLogger(__name__)


def run(
    clips: pd.DataFrame,
    out_dir: Path,
    *,
    parameters: dict,
    inputs: list,
    cfg: dict,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    annotations = utils.scene_pattern_matrix()
    u_cfg = cfg["analysis"]["clustering"]["umap"]
    k_values = list(cfg["analysis"]["clustering"]["k_values"])

    coords = _compute_umap(annotations, u_cfg)
    coords_path = out_dir / "umap_2d.csv"
    coords.to_csv(coords_path)
    provenance.write_sidecar(coords_path, parameters=parameters, inputs=inputs, random_seed=int(u_cfg["random_state"]))

    jaccard = _jaccard(annotations)
    umap_d = squareform(pdist(coords.values, metric="euclidean"))
    np.save(out_dir / "distance_jaccard.npy", jaccard)
    np.save(out_dir / "distance_umap.npy", umap_d)
    pd.Series(annotations.index, name="scene_id").to_csv(out_dir / "scene_index.csv", index=False)
    for f in ("distance_jaccard.npy", "distance_umap.npy", "scene_index.csv"):
        provenance.write_sidecar(out_dir / f, parameters=parameters, inputs=inputs)

    clustered = _kmeans_grid(coords, k_values, seed=int(u_cfg["random_state"]))
    clustered_path = out_dir / "scenes_clustered.csv"
    clustered.to_csv(clustered_path)
    provenance.write_sidecar(clustered_path, parameters=parameters, inputs=inputs)

    perf = _per_cluster_performance(clips, clustered, k_values)
    perf_path = out_dir / "cluster_performance.csv"
    perf.to_csv(perf_path, index=False)
    provenance.write_sidecar(perf_path, parameters=parameters, inputs=inputs)

    # Figures: default to the median k for the scatter, and Cleared-by-cluster bars at all k.
    k_default = sorted(k_values)[len(k_values) // 2]
    scatter_path = figs_dir / f"umap_clusters_k{k_default}.png"
    _figure_umap(coords, clustered[f"k={k_default}"], scatter_path, k=k_default, cfg=cfg)
    provenance.write_sidecar(scatter_path, parameters=parameters, inputs=inputs)

    bar_path = figs_dir / "cluster_performance.png"
    _figure_perf(perf, bar_path, cfg=cfg)
    provenance.write_sidecar(bar_path, parameters=parameters, inputs=inputs)

    paths = {
        "umap": coords_path,
        "scenes_clustered": clustered_path,
        "cluster_performance": perf_path,
        "scatter": scatter_path,
        "bars": bar_path,
    }
    log.info("clustering: %d scenes, k=%s", len(annotations), k_values)
    return paths


def _compute_umap(annotations: pd.DataFrame, u_cfg: dict) -> pd.DataFrame:
    import umap.umap_ as umap_

    reducer = umap_.UMAP(
        n_components=2,
        n_neighbors=int(u_cfg["n_neighbors"]),
        min_dist=float(u_cfg["min_dist"]),
        metric=str(u_cfg["metric"]),
        random_state=int(u_cfg["random_state"]),
    )
    coords = reducer.fit_transform(annotations.values.astype(float))
    return pd.DataFrame(coords, columns=["DR_1", "DR_2"], index=annotations.index)


def _jaccard(annotations: pd.DataFrame) -> np.ndarray:
    """Pairwise Jaccard with safe handling of all-zero rows (scipy returns NaN)."""
    X = annotations.values.astype(bool)
    d = squareform(pdist(X, metric="jaccard"))
    is_empty = ~X.any(axis=1)
    if is_empty.any():
        d[is_empty, :] = 1.0
        d[:, is_empty] = 1.0
        ee = np.outer(is_empty, is_empty)
        d[ee] = 0.0
    np.fill_diagonal(d, 0.0)
    return d


def _kmeans_grid(coords: pd.DataFrame, k_values: list[int], *, seed: int) -> pd.DataFrame:
    out = pd.DataFrame(index=coords.index)
    for k in k_values:
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(coords.values)
        out[f"k={k}"] = labels
    return out


def _per_cluster_performance(
    clips: pd.DataFrame,
    clustered: pd.DataFrame,
    k_values: list[int],
) -> pd.DataFrame:
    long = clustered.reset_index().melt(id_vars="scene_id", var_name="k", value_name="cluster")
    long["k"] = long["k"].str.replace("k=", "").astype(int)
    merged = clips.merge(long, left_on="SceneID", right_on="scene_id", how="inner")
    g = merged.groupby(["Subject", "k", "cluster"])
    perf = pd.DataFrame({
        "n_clips": g.size(),
        "clear_rate": g["Cleared"].mean(),
        "median_duration_s": g["Duration"].median(),
        "median_hits": g["Hits_taken"].median(),
    }).reset_index()
    return perf


def _figure_umap(coords: pd.DataFrame, labels: pd.Series, out_path: Path, *, k: int, cfg: dict) -> None:
    style = plots.style(cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords["DR_1"], coords["DR_2"], c=labels, cmap="tab20", s=18, edgecolor="none")
    ax.set_title(f"UMAP scene-space (K-means, k={k})")
    ax.set_xlabel("DR_1")
    ax.set_ylabel("DR_2")
    plt.colorbar(sc, ax=ax, label="cluster")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_perf(perf: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    k_values = sorted(perf["k"].unique())
    fig, axes = plt.subplots(len(k_values), 1, figsize=(10, 2.4 * len(k_values)), sharex=False, squeeze=False)
    for i, k in enumerate(k_values):
        ax = axes[i][0]
        sub = perf[perf["k"] == k]
        pivot = sub.pivot(index="cluster", columns="Subject", values="clear_rate")
        pivot.plot(kind="bar", ax=ax, colormap="tab10", legend=(i == 0))
        ax.set_title(f"k={k}")
        ax.set_ylabel("clear rate")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cleared rate by cluster, per subject")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])
