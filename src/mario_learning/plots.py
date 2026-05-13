"""Shared plotting helpers used across analyses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def style(cfg: dict) -> dict:
    """Pull figure-style knobs out of config with safe fallbacks."""
    fig = cfg.get("analysis", {}).get("figure", {})
    return {
        "dpi": fig.get("dpi", 150),
        "palette": fig.get("palette", "tab10"),
    }


def save_figure(fig: plt.Figure, path, *, dpi: int) -> None:
    """Write `fig` to `path` and close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def moving_average(values: np.ndarray | pd.Series, window: int) -> np.ndarray:
    """Centered rolling mean; trailing/leading edges use the available window."""
    s = pd.Series(values).rolling(window, min_periods=1, center=True).mean()
    return s.to_numpy()
