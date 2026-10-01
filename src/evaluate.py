"""
Shared scorecard so every model is judged the same way.

- accuracy:  share of games where the favourite (p >= 0.5 for home) won
- log loss:  punishes confident wrong predictions; the main metric (lower is better)
- Brier:     mean squared error of the probability (lower is better)
- bootstrap: is the gap between two models real, or could it be luck?
- calibration plot: do "70%" predictions actually win about 70% of the time?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-15


def per_game_log_loss(y, p) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n_games": int(len(y)),
        "accuracy": float(np.mean((p >= 0.5) == (y == 1))),
        "log_loss": float(per_game_log_loss(y, p).mean()),
        "brier": float(np.mean((p - y) ** 2)),
    }


def bootstrap_log_loss_diff(y, p_a, p_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap of mean log-loss difference (A minus B).

    Negative = model A is better. If the 95% interval excludes 0,
    the gap is unlikely to be luck. Note: resampling individual games
    treats them as independent, which is an approximation.
    """
    d = per_game_log_loss(y, p_a) - per_game_log_loss(y, p_b)
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "share_A_better": float(np.mean(means < 0)),
    }


def bootstrap_mean(x, n_boot: int = 2000, seed: int = 42) -> dict:
    """Mean of per-game values with a 95% bootstrap interval."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return {"mean": float(x.mean()), "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5))}


def bootstrap_group_gap(x_a, x_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Is the mean of group A different from group B? (resamples each group separately)"""
    x_a, x_b = np.asarray(x_a, dtype=float), np.asarray(x_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.array([x_a[rng.integers(0, len(x_a), len(x_a))].mean()
                      - x_b[rng.integers(0, len(x_b), len(x_b))].mean() for _ in range(n_boot)])
    return {"mean_diff": float(x_a.mean() - x_b.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}


def calibration_plot(y, preds: dict, path: Path, n_bins: int = 10, title: str = "") -> None:
    """Predicted probability (x) vs. how often the home team actually won (y)."""
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    markers = ["o", "s", "^", "D"]
    styles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1, label="Perfect calibration")
    for i, (name, p) in enumerate(preds.items()):
        p = np.asarray(p, dtype=float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() >= 20:  # skip near-empty bins: too noisy to read
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
        ax.plot(xs, ys, marker=markers[i % 4], linestyle=styles[i % 4], linewidth=2, label=name)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability home team wins")
    ax.set_ylabel("Observed home win rate")
    ax.set_title(title or "Calibration on test seasons", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
