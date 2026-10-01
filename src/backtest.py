"""
Step 1 backtest: Model 0 (home-win-rate baseline) vs. Model 1 (Elo).

Season roles (fixed BEFORE looking at test results):
    warm-up : 2014-15            Elo ratings settle; not scored
    tune    : 2015-16 .. 2018-19 Elo settings (K, home advantage) chosen here only
    test    : 2019-20 onwards    scored, never used for any choice

2019-20 (bubble) and 2020-21 (few or no fans) are "disrupted" seasons for
home-court advantage, so results are reported with and without them.

Usage (from repo root, after `python -m src.data`):
    python -m src.backtest
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from .data import RAW_PATH, ROOT
from .elo import EloConfig, run_elo
from .evaluate import bootstrap_log_loss_diff, calibration_plot, score

WARMUP = ["2014-15"]
TUNE = ["2015-16", "2016-17", "2017-18", "2018-19"]
DISRUPTED = {"2019-20", "2020-21"}
RESULTS = ROOT / "results"
PROCESSED = ROOT / "data" / "processed"

K_GRID = [10, 15, 20, 25, 30]
HOME_GRID = [50, 75, 100, 125]


def home_rate_baseline(games: pd.DataFrame) -> pd.Series:
    """Model 0: P(home win) = home win rate across all EARLIER seasons."""
    seasons = sorted(games["season"].unique())
    p = pd.Series(index=games.index, dtype=float)
    for i, s in enumerate(seasons):
        prior = games[games["season"].isin(seasons[:i])]
        p[games["season"] == s] = prior["home_win"].mean() if len(prior) else 0.5
    return p


def tune_elo(games: pd.DataFrame) -> tuple[EloConfig, pd.DataFrame]:
    """Grid-search K and home advantage, scored on tune seasons only."""
    rows = []
    for k, h in itertools.product(K_GRID, HOME_GRID):
        cfg = EloConfig(k=k, home_adv=h)
        preds, _ = run_elo(games, cfg)
        merged = games[["game_id", "season", "home_win"]].merge(preds, on="game_id")
        tune_rows = merged[merged["season"].isin(TUNE)]
        rows.append({"k": k, "home_adv": h, **score(tune_rows["home_win"], tune_rows["p_elo"])})
    table = pd.DataFrame(rows).sort_values("log_loss")
    best = table.iloc[0]
    return EloConfig(k=float(best["k"]), home_adv=float(best["home_adv"])), table


def fmt_table(df: pd.DataFrame) -> str:
    """Minimal markdown table (avoids an extra dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Model 0 and Model 1 (Elo).")
    parser.add_argument("--games", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    games = pd.read_csv(args.games, parse_dates=["game_date"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(games["season"].unique())
    test_seasons = [s for s in seasons if s not in WARMUP + TUNE]
    print(f"Loaded {len(games):,} games | test seasons: {test_seasons}")

    # Model 1: tune on tune seasons, then run once over everything
    best_cfg, tuning = tune_elo(games)
    print(f"Best Elo settings on tune seasons: K={best_cfg.k:g}, home_adv={best_cfg.home_adv:g}")
    preds, _ = run_elo(games, best_cfg)
    games = games.merge(preds, on="game_id", how="left")

    # Model 0
    games["p_home_rate"] = home_rate_baseline(games)

    test = games[games["season"].isin(test_seasons)]
    models = {"Model 0: home win rate": "p_home_rate", "Model 1: Elo": "p_elo"}

    # Per-season scorecard
    by_season = []
    for s, grp in test.groupby("season"):
        row = {"season": s, "n_games": len(grp), "actual_home_win_rate": grp["home_win"].mean()}
        for name, col in models.items():
            m = score(grp["home_win"], grp[col])
            short = "baseline" if "0" in name else "elo"
            row[f"{short}_acc"] = m["accuracy"]
            row[f"{short}_logloss"] = m["log_loss"]
        by_season.append(row)
    by_season = pd.DataFrame(by_season)

    # Overall scorecards + significance
    views = {"All test seasons": test,
             "Excluding 2019-20 & 2020-21": test[~test["season"].isin(DISRUPTED)]}
    overall, sig = [], []
    for view, df in views.items():
        for name, col in models.items():
            overall.append({"view": view, "model": name, **score(df["home_win"], df[col])})
        b = bootstrap_log_loss_diff(df["home_win"], df["p_elo"], df["p_home_rate"])
        sig.append({"view": view, "comparison": "Elo minus baseline", **b})
    overall, sig = pd.DataFrame(overall), pd.DataFrame(sig)

    # Save outputs
    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(RESULTS / "elo_tuning.csv", index=False)
    by_season.to_csv(RESULTS / "metrics_by_season.csv", index=False)
    overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
    sig.to_csv(RESULTS / "significance.csv", index=False)
    calibration_plot(test["home_win"], {n: test[c] for n, c in models.items()},
                     RESULTS / "calibration_step1.png",
                     title="Calibration on test seasons: baseline vs. Elo")
    # Pre-game Elo ratings: reused as a LightGBM feature in step 2 (kept out of git)
    games.to_csv(PROCESSED / "games_with_elo.csv", index=False)

    summary = [
        "# Step 1 results: Model 0 vs. Model 1 (Elo)", "",
        f"Elo settings chosen on tune seasons {TUNE[0]} to {TUNE[-1]}: "
        f"K = {best_cfg.k:g}, home advantage = {best_cfg.home_adv:g} Elo points.", "",
        "## Overall", fmt_table(overall), "",
        "## Is Elo's improvement real? (paired bootstrap, log loss)",
        "Negative mean_diff = Elo better. If the 95% interval excludes 0, the gap is unlikely to be luck.", "",
        fmt_table(sig), "",
        "## By season", fmt_table(by_season), "",
        "![Calibration](calibration_step1.png)",
    ]
    (RESULTS / "summary_step1.md").write_text("\n".join(summary))
    print("\n" + fmt_table(overall) + "\n\n" + fmt_table(sig))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
