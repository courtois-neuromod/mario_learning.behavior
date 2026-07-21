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
import seaborn as sns

from mario_learning import plots, provenance, utils

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


MARKERS = ["o", "^", "s", "D", "P", "*", "X"]
STAGE_ABBREV = ["E.Disc", "M.Disc", "L.Disc", "E.Prac", "M.Prac", "L.Prac"]


def _dataset_colors(datasets_sorted: list[str], n_per_dataset: int = 1) -> dict[str, list]:
    """Return colors: first dataset (humans) → viridis, second (model) → magma."""
    cmaps = [plt.get_cmap("viridis"), plt.get_cmap("magma")]
    result = {}
    for i, name in enumerate(datasets_sorted):
        cmap = cmaps[min(i, len(cmaps) - 1)]
        if n_per_dataset == 1:
            result[name] = [cmap(0.6)]
        else:
            result[name] = [cmap(j / max(n_per_dataset - 1, 1)) for j in range(n_per_dataset)]
    return result


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
    ds_colors = _dataset_colors(sorted(merged["dataset"].unique()))
    pivot.plot(kind="bar", ax=ax, color=[ds_colors[d][0] for d in pivot.columns])
    ax.set_title("Completion rate per subject, by dataset")
    ax.set_ylabel("Completion rate")
    ax.set_ylim(0, 1)
    fig_path = out_dir / "completion_by_dataset.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def learning_curves(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Overlay smoothed curves across datasets. Primary: per-pattern. Secondary: per-level."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for tag, fname, group_col, fig_name in [
        ("patterns", "learning_curves_by_pattern.csv", "pattern", "learning_curves_overlay_patterns.png"),
        ("levels", "learning_curves_by_level.csv", "Level", "learning_curves_overlay_levels.png"),
    ]:
        sources = {name: Path(p) / fname for name, p in datasets.items()}
        assert_inputs_exist(sources, "learning-curves")
        frames = []
        for name, csv in sources.items():
            df = pd.read_csv(csv)
            df["dataset"] = name
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
        groups = sorted(merged[group_col].unique())
        variables = [c.removesuffix("_smoothed") for c in merged.columns if c.endswith("_smoothed")]
        fig_path = out_dir / fig_name
        _plot_overlay_grid(merged, group_col, groups, variables, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_{tag}"] = fig_path
    return paths


def _plot_overlay_grid(merged, group_col, groups, variables, out_path, *, cfg) -> None:
    style = plots.style(cfg)
    fig, axes = plt.subplots(len(variables), len(groups),
                              figsize=(2.3 * len(groups), 2.2 * len(variables)),
                              sharex="col", squeeze=False)
    datasets = sorted(merged["dataset"].unique())
    ds_colors = _dataset_colors(datasets)
    for j, group in enumerate(groups):
        for i, var in enumerate(variables):
            ax = axes[i][j]
            for name in datasets:
                sub = merged[(merged[group_col] == group) & (merged["dataset"] == name)]
                if sub.empty:
                    continue
                sub = sub.groupby("clip_index", as_index=False)[f"{var}_smoothed"].mean()
                ax.plot(sub["clip_index"], sub[f"{var}_smoothed"], color=ds_colors[name][0], label=name, lw=1.3)
            if i == 0:
                ax.set_title(str(group), fontsize=7, rotation=0)
            if j == 0:
                ax.set_ylabel(var, fontsize=9)
            if i == 0 and j == len(groups) - 1:
                ax.legend(fontsize=7, loc="best")
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Learning curves — overlay by dataset (group: {group_col})")
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def pattern_performance(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Compare per-pattern clear rate across datasets — per-subject is primary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "per_subject_pattern.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "pattern-performance")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "per_subject_pattern.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    datasets_order = sorted(merged["dataset"].unique())
    ds_colors = _dataset_colors(datasets_order)
    subjects = sorted(merged["Subject"].unique())
    fig, axes = plt.subplots(len(subjects), 1, figsize=(12, 2.4 * len(subjects)), squeeze=False, sharex=True)
    for i, subject in enumerate(subjects):
        ax = axes[i][0]
        sub = merged[merged["Subject"] == subject]
        patterns = sorted(sub["pattern"].unique())
        for ds in datasets_order:
            row = sub[sub["dataset"] == ds].set_index("pattern").reindex(patterns)
            ax.plot(patterns, row["Cleared"], marker="o", color=ds_colors[ds][0], label=ds)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"sub-{subject}\nCleared")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1][0].tick_params(axis="x", labelrotation=60, labelsize=8)
    fig.suptitle("Per-pattern clear rate by dataset, per subject")
    fig_path = out_dir / "per_pattern_clear.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    return {"merged": csv_path, "figure": fig_path}


def pattern_difficulty(datasets: dict[str, Path], out_dir: Path, *, parameters, inputs, cfg) -> dict[str, Path]:
    """Compare pattern difficulty across datasets.

    Emits:

    - ``cleared_by_pattern_per_model.png`` — 3-panel figure: pooled stage bars,
      dataset comparison, and humans-vs-agent stage bars combined.
    - ``cleared_by_pattern_pooled_by_stage.png`` — pooled stage panel on its own.
    - ``improvement_by_pattern_per_subject.png`` — per-subject improvement lines.
    - ``per_subject/sub-XX_stage_by_dataset.png`` — one figure per subject showing
      humans (hatched) vs agent (solid) stage bars side by side.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset pattern_metrics carries the 4-stage breakdown.
    metric_sources = {name: Path(p) / "pattern_metrics.csv" for name, p in datasets.items()}
    assert_inputs_exist(metric_sources, "pattern-difficulty")
    metric_frames = []
    for name, csv in metric_sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        metric_frames.append(df)
    merged = pd.concat(metric_frames, ignore_index=True)
    csv_path = out_dir / "pattern_metrics.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)

    style = plots.style(cfg)
    paths: dict[str, Path] = {"merged": csv_path}

    # ---- killer 2-panel ----
    long_stage = _melt_stage_metrics(merged)
    long_model = _pool_clips_by_dataset(datasets, cfg)
    killer_path = out_dir / "cleared_by_pattern_per_model.png"
    _figure_killer(long_stage, long_model, killer_path, cfg=cfg)
    provenance.write_sidecar(killer_path, parameters=parameters, inputs=inputs)
    paths["killer"] = killer_path

    # ---- top panel only ----
    pooled_path = out_dir / "cleared_by_pattern_pooled_by_stage.png"
    _figure_stage_only(long_stage, pooled_path, cfg=cfg)
    provenance.write_sidecar(pooled_path, parameters=parameters, inputs=inputs)
    paths["figure_pooled"] = pooled_path

    # ---- per-subject improvement diagnostic ----
    subjects = sorted(merged["Subject"].unique())
    patterns = sorted(merged["pattern"].unique())
    datasets_l = sorted(merged["dataset"].unique())
    ds_colors = _dataset_colors(datasets_l)
    fig, axes = plt.subplots(len(subjects), 1, figsize=(12, max(3, 0.4 * len(patterns)) + 1.5 * len(subjects)),
                              squeeze=False, sharex=True)
    for i, subject in enumerate(subjects):
        ax = axes[i][0]
        for ds in datasets_l:
            sub = merged[(merged["Subject"] == subject) & (merged["dataset"] == ds)].set_index("pattern").reindex(patterns)
            ax.plot(patterns, sub["improvement"], marker="o", color=ds_colors[ds][0], label=ds)
        ax.set_ylabel(f"sub-{subject}\nimprovement\n(late_practice − early_discovery)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1][0].tick_params(axis="x", labelrotation=60, labelsize=8)
    fig.suptitle("Per-pattern improvement by dataset, per subject")
    fig_path = out_dir / "improvement_by_pattern_per_subject.png"
    fig.tight_layout()
    plots.save_figure(fig, fig_path, dpi=style["dpi"])
    provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
    paths["figure_improvement"] = fig_path

    # ---- per-subject stage comparison ----
    sub_dir = out_dir / "per_subject"
    sub_dir.mkdir(exist_ok=True)
    for subject in subjects:
        sub_long = long_stage[long_stage["Subject"] == subject]
        sub_id = str(subject).zfill(2)
        fig_path = sub_dir / f"sub-{sub_id}_stage_by_dataset.png"
        _figure_stage_per_subject(sub_long, sub_id, fig_path, cfg=cfg)
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        paths[f"figure_sub-{sub_id}_stage"] = fig_path

    # ---- all-patterns grid ----
    grid_path = out_dir / "all_patterns_subject_tracks.png"
    _figure_all_patterns_grid(long_stage, grid_path, cfg=cfg)
    provenance.write_sidecar(grid_path, parameters=parameters, inputs=inputs)
    paths["figure_all_patterns_grid"] = grid_path

    # ---- all-patterns aggregate (single panel, pooled across patterns) ----
    agg_path = out_dir / "all_patterns_aggregate.png"
    _figure_all_patterns_aggregate(long_stage, agg_path, cfg=cfg)
    provenance.write_sidecar(agg_path, parameters=parameters, inputs=inputs)
    paths["figure_all_patterns_aggregate"] = agg_path

    return paths


def _figure_stage_per_subject(long_stage: pd.DataFrame, subject: str, out_path: Path, *, cfg: dict) -> None:
    """Stage trajectory lines for one subject: solid = humans (viridis), dashed = agent (magma)."""
    style = plots.style(cfg)
    datasets_order = sorted(long_stage["dataset"].unique())
    patterns = sorted(long_stage["pattern"].unique())
    x_labels = utils.STAGES
    x_pos = np.arange(len(x_labels))

    viridis = plt.get_cmap("viridis")
    magma = plt.get_cmap("magma")
    cmaps = [viridis, magma]
    linestyles = ["-", "--"]

    n_pat = len(patterns)
    fig, ax = plt.subplots(figsize=(10, 5))

    for d_idx, ds in enumerate(datasets_order):
        cmap = cmaps[min(d_idx, 1)]
        ls = linestyles[min(d_idx, 1)]
        sub = long_stage[long_stage["dataset"] == ds]
        for p_idx, pattern in enumerate(patterns):
            vals = (sub[sub["pattern"] == pattern]
                    .groupby("Stage", observed=True)["Cleared"]
                    .mean().reindex(x_labels))
            color = cmap(p_idx / max(n_pat - 1, 1))
            label = f"{ds} — {pattern}" if d_idx == 0 else None
            ax.plot(x_pos, vals.values, marker="o", color=color, linestyle=ls, lw=1.2,
                    label=label, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Cleared")
    ax.set_title(f"sub-{subject} — stage trajectory (solid = humans, dashed = agent)", fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _melt_stage_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Wide-form pattern_metrics → long (dataset, Subject, pattern, Stage, Cleared)."""
    return merged.melt(
        id_vars=["dataset", "Subject", "pattern"],
        value_vars=utils.STAGES,
        var_name="Stage", value_name="Cleared",
    ).dropna(subset=["Cleared"])


def _pool_clips_by_dataset(datasets: dict[str, Path], cfg: dict) -> pd.DataFrame:
    """Read each dataset's `output/load/{name}/clips.parquet`, attach patterns, return long-form."""
    frames = []
    for name in datasets:
        cache = utils.output_dir(cfg, "load") / name / "clips.parquet"
        if not cache.exists():
            raise FileNotFoundError(
                f"compare-pattern-difficulty: missing {cache}. Run `inv load --dataset {name}` first."
            )
        df = pd.read_parquet(cache)
        df["dataset"] = name
        frames.append(utils.attach_patterns(df))
    return pd.concat(frames, ignore_index=True)


def _figure_killer(long_stage: pd.DataFrame, long_model: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    import matplotlib.gridspec as gridspec

    style = plots.style(cfg)
    patterns = sorted(set(long_stage["pattern"]).union(long_model["pattern"]))
    datasets_order = sorted(long_stage["dataset"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=False)
    ax0, ax1, ax2 = axes

    # Panel 1: stage breakdown pooled across datasets
    stage_grouped = long_stage.groupby(["pattern", "Stage"], observed=True)["Cleared"].mean().reset_index()
    sns.barplot(data=stage_grouped, x="pattern", y="Cleared", hue="Stage",
                order=patterns, hue_order=utils.STAGES, palette="viridis", ax=ax0)
    ax0.set_ylim(0, 1.05)
    ax0.set_xlabel("")
    ax0.set_ylabel("Cleared")
    ax0.set_title("Pooled subjects & datasets — by stage", fontsize=10)
    ax0.set_axisbelow(True)
    ax0.grid(axis="y", alpha=0.3)
    ax0.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax0.legend(title="Stage", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    # Panel 2: dataset comparison pooled across stages
    ds_colors = _dataset_colors(datasets_order)
    model_grouped = long_model.groupby(["pattern", "dataset"])["Cleared"].mean().reset_index()
    palette_list = [ds_colors[ds][0] for ds in datasets_order]
    sns.barplot(data=model_grouped, x="pattern", y="Cleared", hue="dataset",
                order=patterns, hue_order=datasets_order, palette=palette_list, ax=ax1)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("")
    ax1.set_ylabel("Cleared")
    ax1.set_title("Pooled subjects — by dataset", fontsize=10)
    ax1.set_axisbelow(True)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", labelrotation=60, labelsize=8)
    ax1.legend(title="Dataset", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    # Panel 3: humans vs agent — stage trajectory, pooled subjects (mean ± SEM)
    x_labels = utils.STAGES
    x_pos = np.arange(len(x_labels))
    for ds in datasets_order:
        sub = long_stage[long_stage["dataset"] == ds]
        grouped = sub.groupby("Stage", observed=True)["Cleared"].agg(["mean", "sem"]).reindex(x_labels)
        ax2.plot(x_pos, grouped["mean"], marker="o", color=ds_colors[ds][0], label=ds, lw=1.5)
        ax2.fill_between(x_pos,
                         grouped["mean"] - grouped["sem"],
                         grouped["mean"] + grouped["sem"],
                         color=ds_colors[ds][0], alpha=0.2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Stage")
    ax2.set_ylabel("Cleared")
    ax2.set_title("Humans vs agent — stage trajectory (pooled subjects, mean ± SEM)", fontsize=10)
    ax2.set_axisbelow(True)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=max(style["dpi"], 200))


def _figure_stage_only(long_stage: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    style = plots.style(cfg)
    grouped = long_stage.groupby(["pattern", "Stage"], observed=True)["Cleared"].mean().reset_index()
    patterns = sorted(grouped["pattern"].unique())
    fig, ax = plt.subplots(figsize=(16, 4))
    sns.barplot(
        data=grouped, x="pattern", y="Cleared", hue="Stage",
        order=patterns, hue_order=utils.STAGES,
        palette="viridis", ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Pattern")
    ax.set_ylabel("Cleared")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelrotation=60, labelsize=9)
    ax.legend(title="Stage", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _stage_track_style(datasets_order: list[str]) -> tuple[dict, dict, dict]:
    """Shared colour/linestyle scheme for subject-stage-track figures.

    alphabetical sort: agent_ppo_packnet < humans → index 0 = agent, index 1 = humans
    """
    n_stages = len(utils.STAGES)
    viridis = plt.get_cmap("viridis")
    magma = plt.get_cmap("magma")
    cmap_by_ds = {datasets_order[0]: magma, datasets_order[1]: viridis}
    ls_by_ds = {datasets_order[0]: "--", datasets_order[1]: "-"}
    stage_colors_by_ds = {
        ds: [cmap_by_ds[ds](i / max(n_stages - 1, 1)) for i in range(n_stages)]
        for ds in datasets_order
    }
    return cmap_by_ds, ls_by_ds, stage_colors_by_ds


def _draw_subject_stage_panel(ax, data: pd.DataFrame, *, datasets_order: list[str], subjects: list,
                               cmap_by_ds: dict, ls_by_ds: dict, stage_colors_by_ds: dict,
                               x_pos: np.ndarray, marker_size: float = 5) -> None:
    """Draw subject tracks (marker=subject, colour=stage) plus a bold per-dataset average line.

    ``data`` must carry columns dataset / Subject / Stage / Cleared, already
    restricted to whatever slice (one pattern, or pooled across patterns) the
    caller wants plotted in this single panel.
    """
    for ds in datasets_order:
        ds_data = data[data["dataset"] == ds]
        stage_colors = stage_colors_by_ds[ds]
        ls = ls_by_ds[ds]
        for s_idx, subject in enumerate(subjects):
            marker = MARKERS[s_idx % len(MARKERS)]
            vals = (ds_data[ds_data["Subject"] == subject]
                    .set_index("Stage")["Cleared"]
                    .reindex(utils.STAGES).values.astype(float))
            valid = ~np.isnan(vals)
            if not valid.any():
                continue
            ax.plot(x_pos[valid], vals[valid], color="gray", lw=0.6,
                    alpha=0.35, linestyle=ls, zorder=1)
            for xi, yi, color in zip(x_pos, vals, stage_colors):
                if not np.isnan(yi):
                    ax.plot(xi, yi, marker=marker, color=color, markersize=marker_size,
                            linestyle="none", markeredgecolor="white",
                            markeredgewidth=0.3, zorder=2)
        # average line across subjects
        avg = (ds_data.groupby("Stage", observed=True)["Cleared"]
               .mean().reindex(utils.STAGES).values.astype(float))
        ax.plot(x_pos, avg, color=cmap_by_ds[ds](0.6), lw=2.0,
                linestyle=ls, zorder=3)


def _subject_stage_legend_handles(datasets_order: list[str], subjects: list, cmap_by_ds: dict,
                                   ls_by_ds: dict, viridis, magma) -> list:
    from matplotlib.lines import Line2D

    n_stages = len(utils.STAGES)
    legend_handles = []
    for ds in datasets_order:
        legend_handles.append(
            Line2D([0], [0], color=cmap_by_ds[ds](0.6), linestyle=ls_by_ds[ds],
                   lw=2.0, label=f"avg {ds}")
        )
    legend_handles.append(Line2D([0], [0], color="none", label=""))
    for ds in datasets_order:
        legend_handles.append(
            Line2D([0], [0], color="gray", linestyle=ls_by_ds[ds], lw=1.0,
                   label=f"subjects ({ds})")
        )
    legend_handles.append(Line2D([0], [0], color="none", label=""))  # spacer
    for s_idx, subj in enumerate(subjects):
        legend_handles.append(
            Line2D([0], [0], marker=MARKERS[s_idx % len(MARKERS)], color="gray",
                   linestyle="none", markersize=5, label=f"sub-{subj}")
        )
    legend_handles.append(Line2D([0], [0], color="none", label=""))  # spacer
    for stage_idx, abbrev in enumerate(STAGE_ABBREV):
        legend_handles.append(
            Line2D([0], [0], marker="s", color=viridis(stage_idx / max(n_stages - 1, 1)),
                   linestyle="none", markersize=6, label=f"humans — {abbrev}")
        )
    for stage_idx, abbrev in enumerate(STAGE_ABBREV):
        legend_handles.append(
            Line2D([0], [0], marker="s", color=magma(stage_idx / max(n_stages - 1, 1)),
                   linestyle="none", markersize=6, label=f"agent — {abbrev}")
        )
    return legend_handles


def _figure_all_patterns_grid(long_stage: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    """7×4 grid with one subplot per pattern.

    Encoding:
    - Marker colour → training stage (viridis = humans, magma = agent)
    - Marker shape  → subject identity
    - Line style    → solid = humans, dashed = agent (thin grey connector)
    """
    style = plots.style(cfg)
    datasets_order = sorted(long_stage["dataset"].unique())
    patterns = sorted(long_stage["pattern"].unique())
    subjects = sorted(long_stage["Subject"].unique())
    n_stages = len(utils.STAGES)
    x_pos = np.arange(n_stages)

    viridis = plt.get_cmap("viridis")
    magma = plt.get_cmap("magma")
    cmap_by_ds, ls_by_ds, stage_colors_by_ds = _stage_track_style(datasets_order)

    nrows, ncols = 4, 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 3.4), squeeze=False)

    for idx, pattern in enumerate(patterns):
        row, col = idx // ncols, idx % ncols
        ax = axes[row][col]
        pat_data = long_stage[long_stage["pattern"] == pattern]

        _draw_subject_stage_panel(ax, pat_data, datasets_order=datasets_order, subjects=subjects,
                                   cmap_by_ds=cmap_by_ds, ls_by_ds=ls_by_ds,
                                   stage_colors_by_ds=stage_colors_by_ds, x_pos=x_pos)

        ax.set_title(pattern, fontsize=7, pad=2)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(STAGE_ABBREV, fontsize=4.5, rotation=45, ha="right")
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis="y", labelsize=5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("Cleared", fontsize=6)

    for idx in range(len(patterns), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # Legend: averages + individual line styles + marker shapes + stage color swatches
    legend_handles = _subject_stage_legend_handles(datasets_order, subjects, cmap_by_ds, ls_by_ds, viridis, magma)

    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=6, framealpha=0.9)
    fig.suptitle("Pattern difficulty — subject tracks  (shape = subject, colour = stage)",
                 fontsize=9, y=1.002)
    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_all_patterns_aggregate(long_stage: pd.DataFrame, out_path: Path, *, cfg: dict) -> None:
    """Single-panel version of the all-patterns grid, pooled across patterns.

    Same encoding as one cell of ``all_patterns_subject_tracks.png`` (marker
    shape = subject, marker colour = stage, bold line = per-dataset average),
    but each subject's per-stage value is averaged across *all* patterns first,
    so the figure shows one aggregate humans line and one aggregate agent line
    plus the individual subject tracks behind them.
    """
    style = plots.style(cfg)
    datasets_order = sorted(long_stage["dataset"].unique())
    subjects = sorted(long_stage["Subject"].unique())
    n_stages = len(utils.STAGES)
    x_pos = np.arange(n_stages)

    viridis = plt.get_cmap("viridis")
    magma = plt.get_cmap("magma")
    cmap_by_ds, ls_by_ds, stage_colors_by_ds = _stage_track_style(datasets_order)

    # Pool across patterns: mean Cleared per (dataset, Subject, Stage).
    agg = long_stage.groupby(["dataset", "Subject", "Stage"], observed=True)["Cleared"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_subject_stage_panel(ax, agg, datasets_order=datasets_order, subjects=subjects,
                               cmap_by_ds=cmap_by_ds, ls_by_ds=ls_by_ds,
                               stage_colors_by_ds=stage_colors_by_ds, x_pos=x_pos, marker_size=7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(STAGE_ABBREV, fontsize=9, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Cleared", fontsize=10)
    ax.set_xlabel("Stage")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("All patterns aggregated — subject tracks (shape = subject, colour = stage)", fontsize=11)

    legend_handles = _subject_stage_legend_handles(datasets_order, subjects, cmap_by_ds, ls_by_ds, viridis, magma)
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)

    fig.tight_layout()
    plots.save_figure(fig, out_path, dpi=style["dpi"])


def _figure_per_pattern_lines(long_stage: pd.DataFrame, out_dir: Path, *, cfg: dict) -> dict[str, Path]:
    """One figure per pattern using the same encoding as the grid (shape=subject, colour=stage)."""
    from matplotlib.lines import Line2D

    MARKERS = ["o", "^", "s", "D", "P", "*", "X"]
    STAGE_ABBREV = ["E.Disc", "M.Disc", "L.Disc", "E.Prac", "M.Prac", "L.Prac"]

    style = plots.style(cfg)
    datasets_order = sorted(long_stage["dataset"].unique())
    patterns = sorted(long_stage["pattern"].unique())
    subjects = sorted(long_stage["Subject"].unique())
    n_stages = len(utils.STAGES)
    x_pos = np.arange(n_stages)

    viridis = plt.get_cmap("viridis")
    magma = plt.get_cmap("magma")
    cmap_by_ds = {datasets_order[0]: magma, datasets_order[1]: viridis}
    ls_by_ds = {datasets_order[0]: "--", datasets_order[1]: "-"}
    stage_colors_by_ds = {
        ds: [cmap_by_ds[ds](i / max(n_stages - 1, 1)) for i in range(n_stages)]
        for ds in datasets_order
    }

    paths: dict[str, Path] = {}
    for pattern in patterns:
        fig, ax = plt.subplots(figsize=(5, 6))
        pat_data = long_stage[long_stage["pattern"] == pattern]

        for ds in datasets_order:
            ds_data = pat_data[pat_data["dataset"] == ds]
            stage_colors = stage_colors_by_ds[ds]
            ls = ls_by_ds[ds]
            for s_idx, subject in enumerate(subjects):
                marker = MARKERS[s_idx % len(MARKERS)]
                vals = (ds_data[ds_data["Subject"] == subject]
                        .set_index("Stage")["Cleared"]
                        .reindex(utils.STAGES).values.astype(float))
                valid = ~np.isnan(vals)
                if not valid.any():
                    continue
                ax.plot(x_pos[valid], vals[valid], color="gray", lw=0.6,
                        alpha=0.35, linestyle=ls, zorder=1)
                for xi, yi, color in zip(x_pos, vals, stage_colors):
                    if not np.isnan(yi):
                        ax.plot(xi, yi, marker=marker, color=color, markersize=6,
                                linestyle="none", markeredgecolor="white",
                                markeredgewidth=0.3, zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(STAGE_ABBREV, fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylabel("Cleared", fontsize=8)
        ax.set_title(f"{pattern}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        legend_handles = []
        for ds in datasets_order:
            legend_handles.append(
                Line2D([0], [0], color="gray", linestyle=ls_by_ds[ds], lw=1.2, label=ds)
            )
        legend_handles.append(Line2D([0], [0], color="none", label=""))
        for s_idx, subj in enumerate(subjects):
            legend_handles.append(
                Line2D([0], [0], marker=MARKERS[s_idx % len(MARKERS)], color="gray",
                       linestyle="none", markersize=5, label=f"sub-{subj}")
            )
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)

        fig.tight_layout()
        safe_name = pattern.replace("/", "_").replace(" ", "_")
        fig_path = out_dir / f"{safe_name}_subject_lines.png"
        plots.save_figure(fig, fig_path, dpi=style["dpi"])
        plt.close(fig)
        paths[f"figure_pat_{safe_name}"] = fig_path

    return paths


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
    datasets_order = sorted(merged["dataset"].unique())
    ds_colors = _dataset_colors(datasets_order)
    fig, axes = plt.subplots(len(k_values), 1, figsize=(10, 2.4 * len(k_values)), squeeze=False)
    for i, k in enumerate(k_values):
        ax = axes[i][0]
        sub = (
            merged[merged["k"] == k]
            .groupby(["dataset", "cluster"], as_index=False)["clear_rate"].mean()
        )
        pivot = sub.pivot(index="cluster", columns="dataset", values="clear_rate")
        pivot.plot(kind="bar", ax=ax, color=[ds_colors[d][0] for d in pivot.columns],
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
    """Concat per-(subject, pattern, stage) summary tables across datasets."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "per_subject_pattern_stage.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "summary")
    frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    csv_path = out_dir / "per_subject_pattern_stage.csv"
    merged.to_csv(csv_path, index=False)
    provenance.write_sidecar(csv_path, parameters=parameters, inputs=inputs)
    log.info("compare-summary: tables merged at %s. Use per-dataset figures for visual diffs.", csv_path)
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
    """Overlay per-(subject, pattern) KM curves across datasets.

    Reads each dataset's `per_subject_pattern_km.csv` (long-form survival
    function). One figure per pattern, panels per subject, lines per dataset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {name: Path(p) / "per_subject_pattern_km.csv" for name, p in datasets.items()}
    assert_inputs_exist(sources, "survival")

    long_frames = []
    for name, csv in sources.items():
        df = pd.read_csv(csv)
        df["dataset"] = name
        long_frames.append(df)
    merged_long = pd.concat(long_frames, ignore_index=True)

    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)
    style = plots.style(cfg)
    datasets_order = sorted(merged_long["dataset"].unique())
    ds_colors = _dataset_colors(datasets_order)
    coverage_rows: list[dict] = []
    patterns = sorted(merged_long["pattern"].unique())
    subjects = sorted(merged_long["Subject"].unique())
    for pattern in patterns:
        cols = min(3, len(subjects))
        rows = (len(subjects) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.4 * rows), squeeze=False, sharey=True)
        for i, subject in enumerate(subjects):
            ax = axes[i // cols][i % cols]
            sub = merged_long[(merged_long["pattern"] == pattern) & (merged_long["Subject"] == subject)]
            for ds in datasets_order:
                ds_df = sub[sub["dataset"] == ds]
                ax.step(ds_df["t"], ds_df["S"], where="post", color=ds_colors[ds][0], label=ds, lw=1.2)
            ax.set_title(f"sub-{subject}", fontsize=8)
            ax.set_ylim(0, 1.05)
            if i == 0:
                ax.legend(fontsize=7)
        for i in range(len(subjects), rows * cols):
            axes[i // cols][i % cols].axis("off")
        fig.suptitle(f"KM by pattern: {pattern}")
        fig_path = figs_dir / f"{pattern.replace('/', '_')}_km_overlay.png"
        fig.tight_layout()
        plots.save_figure(fig, fig_path, dpi=style["dpi"])
        provenance.write_sidecar(fig_path, parameters=parameters, inputs=inputs)
        coverage_rows.append({
            "pattern": pattern,
            "n_datasets": merged_long[merged_long["pattern"] == pattern]["dataset"].nunique(),
            "n_subjects": merged_long[merged_long["pattern"] == pattern]["Subject"].nunique(),
        })

    coverage_path = out_dir / "pattern_overlap.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)
    provenance.write_sidecar(coverage_path, parameters=parameters, inputs=inputs)
    return {"pattern_overlap": coverage_path}


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
