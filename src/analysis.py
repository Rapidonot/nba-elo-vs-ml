"""
Step 3: where does LightGBM's gain come from, and when does Elo struggle?

    H4  ablation ladder: Elo only -> + team strength -> + four factors
        -> + situation -> + travel (= Model 3), plus leave-one-group-out
    H5  early-season games vs. later games
    H6  home-court advantage in 2020-21 and over-prediction by season

Measurement rules were fixed before this ran: see the dated note in
HYPOTHESES.md. Uses the same season roles, tuning and walk-forward scheme
as step 2.

Usage (from repo root, after `python -m src.data` and `python -m src.backtest_ml`):
    python -m src.analysis
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import PROCESSED, RESULTS, TUNE, WARMUP, fmt_table, tune_elo
from .backtest_ml import tune_lgb, walk_forward
from .data import RAW_PATH
from .elo import run_elo
from .evaluate import (bootstrap_group_gap, bootstrap_log_loss_diff, bootstrap_mean,
                       per_game_log_loss, score)
from .features import BUBBLE_START, ELO_FEATURES, FEATURE_GROUPS, build_features

LADDER_ORDER = ["team_strength", "four_factors", "situation", "travel"]
EARLY_GAMES = 20
GAME_BINS = [0, 10, 20, 40, 60, 83]


def _style(ax, title: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ablation(games: pd.DataFrame, test_seasons: list[str]):
    """H4: ladder and leave-one-group-out, each feature set tuned and tested like Model 3."""
    cache: dict = {}

    def predict(features: list[str]) -> pd.Series:
        key = tuple(features)
        if key not in cache:
            params, _ = tune_lgb(games, features)
            cache[key], _ = walk_forward(games, features, params, test_seasons)
            print(f"  {len(features):>2} features -> {params}")
        return cache[key]

    test = games["season"].isin(test_seasons)
    y = games.loc[test, "home_win"]

    rungs = [("L0: Elo only", list(ELO_FEATURES))]
    for i, g in enumerate(LADDER_ORDER):
        rungs.append((f"L{i + 1}: + {g.replace('_', ' ')}", rungs[-1][1] + FEATURE_GROUPS[g]))
    ladder, prev = [], None
    for name, feats in rungs:
        p = predict(feats)[test]
        row = {"rung": name, "n_features": len(feats), **score(y, p)}
        if prev is not None:  # negative = this rung improves on the one before
            b = bootstrap_log_loss_diff(y, p, prev)
            row.update({"step_diff": b["mean_diff"], "step_ci_low": b["ci_low"], "step_ci_high": b["ci_high"]})
        ladder.append(row)
        prev = p
    ladder = pd.DataFrame(ladder)

    full_feats = rungs[-1][1]
    full = predict(full_feats)[test]
    logo = []
    for g in LADDER_ORDER:
        feats = [f for f in full_feats if f not in FEATURE_GROUPS[g]]
        b = bootstrap_log_loss_diff(y, predict(feats)[test], full)  # positive = removing g hurts
        logo.append({"removed_group": g, **b})
    return ladder, pd.DataFrame(logo), full


def early_vs_late(pred: pd.DataFrame):
    """H5: is Model 3's edge over Elo bigger in each team's first 20 games?"""
    early = (pred["home_games_played"] < EARLY_GAMES) & (pred["away_games_played"] < EARLY_GAMES)
    rows, by_bin = [], []
    for label, elo_col in (("H5: Model 1 minus Model 3", "p_elo"),
                           ("Exploratory: Model 1c minus Model 3", "p_elo_1c")):
        gap = per_game_log_loss(pred["home_win"], pred[elo_col]) - per_game_log_loss(pred["home_win"], pred["p_lgb_elo"])
        rows.append({"comparison": label, "n_early": int(early.sum()), "n_later": int((~early).sum()),
                     "early_gap": gap[early].mean(), "later_gap": gap[~early].mean(),
                     **{f"diff_{k}": v for k, v in bootstrap_group_gap(gap[early], gap[~early]).items()}})
        game_no = (pred["home_games_played"] + pred["away_games_played"]) / 2
        bins = pd.cut(game_no, GAME_BINS, right=False)
        for b, idx in gap.groupby(bins, observed=True).groups.items():
            by_bin.append({"comparison": label, "games_played": f"{int(b.left)}-{int(b.right) - 1}",
                           "n_games": len(idx), **bootstrap_mean(gap[idx])})
    return pd.DataFrame(rows), pd.DataFrame(by_bin)


def home_court(games: pd.DataFrame, pred: pd.DataFrame, test_seasons: list[str]):
    """H6: home win rate by season, and how far each model over-predicts it."""
    rates = []
    for s, grp in games[~games["season"].isin(WARMUP)].groupby("season"):
        rates.append({"season": s, "n_games": len(grp), **bootstrap_mean(grp["home_win"])})
    bubble = games[(games["season"] == "2019-20") & (games["game_date"] >= BUBBLE_START)]
    rates.append({"season": "2019-20 bubble games only", "n_games": len(bubble), **bootstrap_mean(bubble["home_win"])})
    rates = pd.DataFrame(rates)

    tune_wins = games.loc[games["season"].isin(TUNE), "home_win"]
    drop = bootstrap_group_gap(games.loc[games["season"] == "2020-21", "home_win"], tune_wins)

    over = []
    for s, grp in pred.groupby("season"):
        for name, col in (("Model 1", "p_elo"), ("Model 1c", "p_elo_1c"), ("Model 3", "p_lgb_elo")):
            over.append({"season": s, "model": name, **bootstrap_mean(grp[col] - grp["home_win"])})
    return rates, drop, tune_wins.mean(), pd.DataFrame(over)


def plot_ladder(ladder: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(ladder["rung"], ladder["log_loss"], marker="o", linewidth=2)
    ax.set_ylabel("Log loss on test seasons (lower is better)")
    ax.tick_params(axis="x", labelrotation=15)
    _style(ax, "Ablation ladder: adding feature groups to Elo")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_gap_by_game(by_bin: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for (label, grp), marker in zip(by_bin.groupby("comparison", sort=False), ["o", "s"]):
        err = [grp["mean"] - grp["ci_low"], grp["ci_high"] - grp["mean"]]
        ax.errorbar(grp["games_played"], grp["mean"], yerr=err, marker=marker, capsize=4, linewidth=2,
                    label=label.split(": ")[1].replace(" minus ", " vs. "))
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xlabel("Games already played by the two teams (average)")
    ax.set_ylabel("Elo log loss minus LightGBM log loss\n(above 0 = LightGBM better)")
    ax.legend(loc="upper right")
    _style(ax, "When does LightGBM beat Elo?")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_home_rate(rates: pd.DataFrame, tune_mean: float, path: Path) -> None:
    r = rates[~rates["season"].str.contains("bubble")]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.errorbar(r["season"], r["mean"], yerr=[r["mean"] - r["ci_low"], r["ci_high"] - r["mean"]],
                marker="o", capsize=4, linewidth=2, label="Home win rate (95% interval)")
    ax.axhline(tune_mean, color="grey", linestyle="--", linewidth=1.5, label="2015-16 to 2018-19 average")
    ax.axhline(0.5, color="grey", linewidth=1)
    ax.set_ylabel("Share of games won by home team")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(loc="upper right")
    _style(ax, "Home-court advantage by season")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: ablation (H4), timing (H5), home court (H6).")
    parser.add_argument("--games", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    games = pd.read_csv(args.games, parse_dates=["game_date"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(games["season"].unique())
    test_seasons = [s for s in seasons if s not in WARMUP + TUNE]
    pred = pd.read_csv(PROCESSED / "predictions_step2.csv")

    cfg1, _ = tune_elo(games)
    games = games.merge(run_elo(games, cfg1)[0], on="game_id")
    games = games.merge(build_features(games), on="game_id")
    games["elo_diff"] = games["elo_home_pre"] - games["elo_away_pre"]

    print("H4: ablation ladder and leave-one-group-out")
    ladder, logo, full = ablation(games, test_seasons)
    # Sanity check: the full ladder rung must reproduce step 2's Model 3
    drift = np.abs(full.to_numpy() - pred["p_lgb_elo"].to_numpy()).max()
    assert drift < 1e-9, f"full rung differs from step 2 Model 3 by {drift}"

    h5, h5_bins = early_vs_late(pred)
    rates, drop, tune_mean, over = home_court(games, pred, test_seasons)

    total = ladder["log_loss"].iloc[0] - ladder["log_loss"].iloc[-1]
    ts_step = ladder["log_loss"].iloc[0] - ladder["log_loss"].iloc[1]
    ts_share = ts_step / total if total > 0 else float("nan")

    RESULTS.mkdir(exist_ok=True)
    ladder.to_csv(RESULTS / "step3_ladder.csv", index=False)
    logo.to_csv(RESULTS / "step3_leave_one_group_out.csv", index=False)
    h5.to_csv(RESULTS / "step3_h5_early_vs_later.csv", index=False)
    h5_bins.to_csv(RESULTS / "step3_h5_by_games_played.csv", index=False)
    rates.to_csv(RESULTS / "step3_h6_home_win_rate.csv", index=False)
    over.to_csv(RESULTS / "step3_h6_overprediction.csv", index=False)
    plot_ladder(ladder, RESULTS / "ablation_ladder.png")
    plot_gap_by_game(h5_bins, RESULTS / "gap_by_games_played.png")
    plot_home_rate(rates, tune_mean, RESULTS / "home_win_rate_by_season.png")

    summary = [
        "# Step 3 results: where the gains come from", "",
        "Measurement rules were fixed before this analysis ran (see HYPOTHESES.md).", "",
        "## H4: ablation ladder",
        "step_diff = this rung's log loss minus the previous rung's (negative = improvement), with 95% interval.", "",
        fmt_table(ladder), "",
        f"Total improvement L0 to L4: {total:.4f}. Share from team strength (L0 to L1): "
        f"{ts_share:.0%}." if total > 0 else
        f"Total improvement L0 to L4: {total:.4f} (no overall improvement, so 'share of gain' is undefined).", "",
        "### Leave one group out (from Model 3)",
        "Positive mean_diff = removing the group makes Model 3 worse.", "",
        fmt_table(logo), "",
        "![Ablation ladder](ablation_ladder.png)", "",
        f"## H5: early-season games (both teams fewer than {EARLY_GAMES} games played)",
        "gap = Elo log loss minus Model 3 log loss (positive = LightGBM better). "
        "diff = early gap minus later gap.", "",
        fmt_table(h5), "",
        "### Gap by games already played", fmt_table(h5_bins), "",
        "![Gap by games played](gap_by_games_played.png)", "",
        "## H6: home-court advantage",
        f"Home win rate 2020-21 minus 2015-16 to 2018-19 average ({tune_mean:.3f}): "
        f"{drop['mean_diff']:+.4f} (95% interval {drop['ci_low']:+.4f} to {drop['ci_high']:+.4f}).", "",
        fmt_table(rates), "",
        "### Over-prediction: mean predicted home-win probability minus actual home-win rate",
        "Positive = the model expects the home team to win more often than it did.", "",
        fmt_table(over), "",
        "![Home win rate by season](home_win_rate_by_season.png)",
    ]
    (RESULTS / "summary_step3.md").write_text("\n".join(summary))
    print("\n" + fmt_table(ladder) + "\n\n" + fmt_table(logo) + "\n\n" + fmt_table(h5))
    print(f"\nH6 drop: {drop}")
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
