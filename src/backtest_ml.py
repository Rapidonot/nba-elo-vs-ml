"""
Step 2 backtest: Elo vs. LightGBM.

    Model 0  : home win rate from earlier seasons
    Model 1  : Elo, fixed home advantage (as registered)
    Model 1b : Elo whose home advantage is learned game by game (fairness check,
               see the dated note in HYPOTHESES.md)
    Model 2  : LightGBM on all feature groups, WITHOUT Elo
    Model 3  : LightGBM on all feature groups plus Elo ratings

Same season roles as step 1. LightGBM settings are chosen by walk-forward
validation inside the tune seasons only (train on earlier tune seasons,
validate on 2017-18 and 2018-19). Each test season is then predicted by a
model trained on every non-warm-up season before it, so like Elo it can
adapt to recent seasons but never sees the season it predicts.

Usage (from repo root, after `python -m src.data`):
    python -m src.backtest_ml
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import DISRUPTED, PROCESSED, RESULTS, TUNE, WARMUP, fmt_table, home_rate_baseline, tune_elo
from .data import RAW_PATH
from .elo import run_elo
from .evaluate import bootstrap_log_loss_diff, calibration_plot, score
from .features import ELO_FEATURES, FEATURE_GROUPS, build_features

HOME_K_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
LGB_FIXED = dict(objective="binary", learning_rate=0.02, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, random_state=42, deterministic=True,
                 force_row_wise=True, n_jobs=1, verbose=-1)
LGB_GRID = dict(num_leaves=[4, 8, 16], min_child_samples=[50, 200], n_estimators=[200, 500, 1000])
VALIDATION = TUNE[2:]  # 2017-18, 2018-19: each validated by a model trained on the tune seasons before it

MODELS = {
    "Model 0: home win rate": "p_home_rate",
    "Model 1: Elo": "p_elo",
    "Model 1b: Elo, learned home adv.": "p_elo_1b",
    "Model 2: LightGBM, no Elo": "p_lgb_no_elo",
    "Model 3: LightGBM + Elo": "p_lgb_elo",
}
# (label, model A, model B): negative mean_diff = A has lower log loss
COMPARISONS = [
    ("H2 (as registered): Model 3 minus Model 1", "p_lgb_elo", "p_elo"),
    ("Fair check: Model 3 minus Model 1b", "p_lgb_elo", "p_elo_1b"),
    ("H3: Model 2 minus Model 3", "p_lgb_no_elo", "p_lgb_elo"),
    ("Model 1b minus Model 1", "p_elo_1b", "p_elo"),
]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], params: dict):
    model = lgb.LGBMClassifier(**LGB_FIXED, **params)
    model.fit(train[features], train["home_win"])
    return model.predict_proba(test[features])[:, 1], model


def tune_lgb(games: pd.DataFrame, features: list[str]) -> tuple[dict, pd.DataFrame]:
    rows = []
    for values in itertools.product(*LGB_GRID.values()):
        params = dict(zip(LGB_GRID, values))
        losses = []
        for s in VALIDATION:
            train = games[games["season"].isin(TUNE[:TUNE.index(s)])]
            val = games[games["season"] == s]
            p, _ = fit_predict(train, val, features, params)
            losses.append(score(val["home_win"], p)["log_loss"])
        rows.append({**params, "val_log_loss": float(np.mean(losses))})
    table = pd.DataFrame(rows).sort_values("val_log_loss")
    best = {k: int(table.iloc[0][k]) for k in LGB_GRID}
    return best, table


def walk_forward(games: pd.DataFrame, features: list[str], params: dict, test_seasons: list[str]):
    """Predict each test season with a model trained on all earlier non-warm-up seasons."""
    seasons = sorted(games["season"].unique())
    preds = pd.Series(np.nan, index=games.index)
    importance = []
    for s in test_seasons:
        earlier = [x for x in seasons[:seasons.index(s)] if x not in WARMUP]
        train, test = games[games["season"].isin(earlier)], games[games["season"] == s]
        p, model = fit_predict(train, test, features, params)
        preds[test.index] = p
        importance.append(pd.Series(model.booster_.feature_importance("gain"), index=features))
    gain = pd.concat(importance, axis=1).mean(axis=1)
    return preds, (gain / gain.sum()).sort_values(ascending=False)


def home_adv_plot(games: pd.DataFrame, fixed: float, path: Path) -> None:
    """Model 1b's learned home advantage over time, against Model 1's fixed value."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    ax.plot(games["game_date"], games["elo_home_adv_pre_1b"], linewidth=1.5, label="Model 1b: learned")
    ax.axhline(fixed, color="grey", linestyle="--", linewidth=1.5, label="Model 1: fixed")
    ax.set_ylabel("Home advantage (Elo points)")
    ax.set_title("How much is playing at home worth?", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: Elo vs. LightGBM backtest.")
    parser.add_argument("--games", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    games = pd.read_csv(args.games, parse_dates=["game_date"])
    games = games.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    seasons = sorted(games["season"].unique())
    test_seasons = [s for s in seasons if s not in WARMUP + TUNE]
    print(f"Loaded {len(games):,} games | test seasons: {test_seasons}")

    # Models 1 and 1b
    cfg1, _ = tune_elo(games)
    cfg1b, tuning_1b = tune_elo(games, HOME_K_GRID)
    p1, _ = run_elo(games, cfg1)
    p1b, _ = run_elo(games, cfg1b)
    games = games.merge(p1, on="game_id").merge(
        p1b[["game_id", "elo_home_adv_pre", "p_elo"]].rename(
            columns={"elo_home_adv_pre": "elo_home_adv_pre_1b", "p_elo": "p_elo_1b"}), on="game_id")
    print(f"Model 1 : K={cfg1.k:g}, home_adv={cfg1.home_adv:g}")
    print(f"Model 1b: K={cfg1b.k:g}, start home_adv={cfg1b.home_adv:g}, home_k={cfg1b.home_k:g}")

    # Model 0
    games["p_home_rate"] = home_rate_baseline(games)

    # Models 2 and 3
    games = games.merge(build_features(games), on="game_id")
    games["elo_diff"] = games["elo_home_pre"] - games["elo_away_pre"]
    all_features = [c for group in FEATURE_GROUPS.values() for c in group]
    feature_sets = {"p_lgb_no_elo": all_features, "p_lgb_elo": all_features + ELO_FEATURES}
    chosen, importances = {}, {}
    for col, feats in feature_sets.items():
        params, _ = tune_lgb(games, feats)
        chosen[col] = params
        games[col], importances[col] = walk_forward(games, feats, params, test_seasons)
        print(f"{col}: {params}")

    test = games[games["season"].isin(test_seasons)]
    views = {"All test seasons": test,
             "Excluding 2019-20 & 2020-21": test[~test["season"].isin(DISRUPTED)]}

    overall = pd.DataFrame([{"view": v, "model": name, **score(df["home_win"], df[col])}
                            for v, df in views.items() for name, col in MODELS.items()])
    sig = pd.DataFrame([{"view": v, "comparison": label, **bootstrap_log_loss_diff(df["home_win"], df[a], df[b])}
                        for v, df in views.items() for label, a, b in COMPARISONS])
    by_season = pd.DataFrame([
        {"season": s, "n_games": len(grp), "home_win_rate": grp["home_win"].mean(),
         **{col: score(grp["home_win"], grp[col])["log_loss"] for col in MODELS.values()}}
        for s, grp in test.groupby("season")])
    importance = pd.DataFrame(importances).fillna(0).sort_values("p_lgb_elo", ascending=False)
    group_of = {f: g for g, feats in FEATURE_GROUPS.items() for f in feats} | {f: "elo" for f in ELO_FEATURES}
    group_share = importance.groupby(importance.index.map(group_of)).sum().sort_values("p_lgb_elo", ascending=False)

    # Save outputs
    RESULTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    overall.to_csv(RESULTS / "step2_metrics_overall.csv", index=False)
    sig.to_csv(RESULTS / "step2_significance.csv", index=False)
    by_season.to_csv(RESULTS / "step2_logloss_by_season.csv", index=False)
    importance.to_csv(RESULTS / "step2_feature_importance.csv")
    tuning_1b.to_csv(RESULTS / "elo_1b_tuning.csv", index=False)
    calibration_plot(test["home_win"], {n: test[c] for n, c in MODELS.items() if c != "p_home_rate"},
                     RESULTS / "calibration_step2.png", title="Calibration on test seasons: Elo vs. LightGBM")
    home_adv_plot(games, cfg1.home_adv, RESULTS / "home_advantage_learned.png")
    keep = ["game_id", "season", "game_date", "home_team", "away_team", "home_win",
            "home_games_played", "away_games_played", *MODELS.values()]
    test[keep].to_csv(PROCESSED / "predictions_step2.csv", index=False)  # used by step 3 analyses

    summary = [
        "# Step 2 results: Elo vs. LightGBM", "",
        "## Settings (chosen on tune seasons only)",
        f"- Model 1: K = {cfg1.k:g}, home advantage = {cfg1.home_adv:g} Elo points (fixed)",
        f"- Model 1b: K = {cfg1b.k:g}, starting home advantage = {cfg1b.home_adv:g}, "
        f"home-advantage learning rate = {cfg1b.home_k:g}",
        f"- Model 2 (LightGBM, no Elo): {chosen['p_lgb_no_elo']}",
        f"- Model 3 (LightGBM + Elo): {chosen['p_lgb_elo']}", "",
        "## Overall", fmt_table(overall), "",
        "## Are the gaps real? (paired bootstrap, log loss)",
        "Negative mean_diff = the first model is better. A gap counts only if the 95% interval excludes 0.", "",
        fmt_table(sig), "",
        "## Log loss by season", fmt_table(by_season.rename(columns={v: k.split(":")[0] for k, v in MODELS.items()})), "",
        "## Share of LightGBM gain by feature group", fmt_table(group_share.reset_index(names="group")), "",
        "![Calibration](calibration_step2.png)", "",
        "![Learned home advantage](home_advantage_learned.png)",
    ]
    (RESULTS / "summary_step2.md").write_text("\n".join(summary))
    print("\n" + fmt_table(overall) + "\n\n" + fmt_table(sig))
    print(f"\nSaved results to {RESULTS}")


if __name__ == "__main__":
    main()
