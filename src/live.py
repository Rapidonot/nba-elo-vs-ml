"""
Step 4: live, timestamped predictions for the 2026-27 season.

Two models go live, the two that tied in the backtest:
    Model 1c : Elo, tuned on every season before 2026-27
    Model 3  : LightGBM + Elo, trained on every season before 2026-27

Commands (from repo root):
    python -m src.live freeze               # once, before opening night: lock both models
    python -m src.live predict [--commit]   # each day: predict today's games (US Eastern date)
    python -m src.live score [--commit]     # grade every prediction whose game has finished

Rules that keep the record honest:
    - Models are frozen before the season and committed to git; they are not
      retrained during the season (ratings and rolling stats still update).
    - A day's prediction file is never overwritten.
    - Each prediction stores when it was made; only predictions made before
      the scheduled tip-off are scored.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import lightgbm as lgb
import pandas as pd

from .backtest import ROOT, WARMUP, fmt_table, tune_elo
from .backtest_ml import HOME_GRID_1C, fit_predict, tune_lgb
from .data import RAW_PATH, fetch_schedule, fetch_season, team_rows_to_games
from .elo import EloConfig, run_elo
from .evaluate import bootstrap_log_loss_diff, score
from .features import ELO_FEATURES, FEATURE_GROUPS, build_features

SEASON = "2026-27"
MODEL_DIR = ROOT / "models" / SEASON
PRED_DIR = ROOT / "predictions" / SEASON
FEATURES = [c for group in FEATURE_GROUPS.values() for c in group] + ELO_FEATURES


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def load_history() -> pd.DataFrame:
    """All finished regular-season games: past seasons from disk, this season live."""
    past = pd.read_csv(RAW_PATH, parse_dates=["game_date"])
    past = past[past["season"] < SEASON]
    rows = fetch_season(SEASON)
    rows = rows[rows["WL"].notna()] if len(rows) else rows  # skip games still in progress
    current = team_rows_to_games(rows, SEASON) if len(rows) else past.iloc[:0]
    return pd.concat([past, current], ignore_index=True).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def predict_upcoming(history: pd.DataFrame, upcoming: pd.DataFrame, cfg_1c: EloConfig,
                     cfg_feat: EloConfig, booster: lgb.Booster) -> pd.DataFrame:
    """Pre-game probabilities for `upcoming` (one US date, so each team appears at most once).

    The upcoming games are appended with placeholder results. Elo and the
    features record everything BEFORE a game's result is used, so the
    placeholders never affect these predictions.
    """
    placeholder = upcoming.assign(home_pts=0, away_pts=0, home_win=0)
    frame = pd.concat([history, placeholder], ignore_index=True)
    p1c, _ = run_elo(frame, cfg_1c)
    pf, _ = run_elo(frame, cfg_feat)
    feats = build_features(frame)  # placeholder rows have no box score (NaN); features never use their own
    out = upcoming[["game_id"]].merge(p1c[["game_id", "p_elo"]], on="game_id")
    x = (upcoming[["game_id"]].merge(feats, on="game_id")
         .merge(pf[["game_id", "elo_home_pre", "elo_away_pre"]], on="game_id"))
    x["elo_diff"] = x["elo_home_pre"] - x["elo_away_pre"]
    out["p_model3"] = booster.predict(x[FEATURES])
    return out.rename(columns={"p_elo": "p_model1c"})


def freeze() -> None:
    """Tune and train both live models on every season before 2026-27, then save them."""
    games = pd.read_csv(RAW_PATH, parse_dates=["game_date"])
    games = games[games["season"] < SEASON].sort_values(["game_date", "game_id"]).reset_index(drop=True)
    train_seasons = [s for s in sorted(games["season"].unique()) if s not in WARMUP]

    cfg_1c, _ = tune_elo(games, seasons=train_seasons, home_grid=HOME_GRID_1C)  # Model 1c's recipe
    cfg_feat, _ = tune_elo(games)  # Elo feature exactly as Model 3 was built in step 2
    games = games.merge(run_elo(games, cfg_feat)[0], on="game_id").merge(build_features(games), on="game_id")
    games["elo_diff"] = games["elo_home_pre"] - games["elo_away_pre"]
    params, _ = tune_lgb(games, FEATURES)  # same tuning as step 2 (tune seasons only)
    _, model = fit_predict(games[games["season"].isin(train_seasons)], games.iloc[:1], FEATURES, params)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / "model3.txt"))
    config = {
        "season": SEASON, "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on_seasons": train_seasons, "n_training_games": int(games["season"].isin(train_seasons).sum()),
        "model1c_elo": {"k": cfg_1c.k, "home_adv": cfg_1c.home_adv},
        "model3_elo_feature": {"k": cfg_feat.k, "home_adv": cfg_feat.home_adv},
        "model3_lightgbm_params": params, "features": FEATURES,
    }
    (MODEL_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({k: v for k, v in config.items() if k != "features"}, indent=2))


def predict(day: str | None, commit: bool) -> None:
    config = json.loads((MODEL_DIR / "config.json").read_text())
    cfg_1c = EloConfig(**config["model1c_elo"])
    cfg_feat = EloConfig(**config["model3_elo_feature"])
    booster = lgb.Booster(model_file=str(MODEL_DIR / "model3.txt"))

    day = day or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    out_path = PRED_DIR / f"{day}.csv"
    if out_path.exists():
        print(f"{out_path.relative_to(ROOT)} already exists; predictions are never overwritten.")
        return

    schedule = fetch_schedule(SEASON)
    now = pd.Timestamp.now(tz="UTC")
    upcoming = schedule[(schedule["game_date"] == pd.Timestamp(day)) & (schedule["status"] == 1)
                        & (schedule["tipoff_utc"] > now)]
    if upcoming.empty:
        print(f"No upcoming regular-season games on {day}.")
        return

    history = load_history()
    if history["game_date"].max() >= pd.Timestamp(day):
        raise RuntimeError("history already contains games on or after the prediction date")
    preds = predict_upcoming(history, upcoming.drop(columns=["status", "tipoff_utc"]), cfg_1c, cfg_feat, booster)

    out = upcoming[["game_id", "game_date", "tipoff_utc", "home_team", "away_team"]].merge(preds, on="game_id")
    out["game_date"] = out["game_date"].dt.strftime("%Y-%m-%d")
    out["predicted_at_utc"] = now.isoformat(timespec="seconds")
    out["code_version"] = git("rev-parse", "--short", "HEAD")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out.round({"p_model1c": 4, "p_model3": 4}).to_csv(out_path, index=False)
    print(out[["home_team", "away_team", "p_model1c", "p_model3"]].round(3).to_string(index=False))
    print(f"Saved {len(out)} predictions to {out_path.relative_to(ROOT)}")
    if commit:
        git("add", str(out_path))
        git("commit", "-m", f"Predictions for {day} ({len(out)} games)")
        git("push")
        print("Committed and pushed.")


def score_live(commit: bool) -> None:
    files = sorted(PRED_DIR.glob("20*.csv"))
    if not files:
        print("No predictions yet.")
        return
    preds = pd.concat([pd.read_csv(f, dtype={"game_id": str}) for f in files], ignore_index=True)
    results = load_history()
    results = results.loc[results["season"] == SEASON, ["game_id", "home_pts", "away_pts", "home_win"]]
    graded = preds.merge(results, on="game_id")
    on_time = pd.to_datetime(graded["predicted_at_utc"]) < pd.to_datetime(graded["tipoff_utc"])
    late = int((~on_time).sum())
    graded = graded[on_time]
    if graded.empty:
        print("No finished games with on-time predictions yet.")
        return

    overall = pd.DataFrame([{"model": name, **score(graded["home_win"], graded[col])}
                            for name, col in (("Model 1c: Elo", "p_model1c"), ("Model 3: LightGBM + Elo", "p_model3"))])
    b = bootstrap_log_loss_diff(graded["home_win"], graded["p_model3"], graded["p_model1c"])
    lines = [
        f"# Live scoreboard: {SEASON}", "",
        f"Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. {len(graded)} finished games with predictions "
        f"made before tip-off ({len(preds)} predicted in total"
        + (f"; {late} excluded for being made after tip-off" if late else "") + ").", "",
        fmt_table(overall), "",
        "## Model 3 minus Model 1c (log loss, paired bootstrap)",
        "Negative = Model 3 better. A gap counts only if the 95% interval excludes 0.", "",
        fmt_table(pd.DataFrame([b])), "",
        f"Home teams have won {graded['home_win'].mean():.1%} of these games.",
    ]
    out_path = PRED_DIR / "scoreboard.md"
    out_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if commit:
        git("add", str(out_path))
        if git("status", "--porcelain", str(out_path)):
            git("commit", "-m", f"Live scoreboard: {len(graded)} games graded")
            git("push")
            print("Committed and pushed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live 2026-27 predictions.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze", help="tune, train and save both live models (run once, before the season)")
    p = sub.add_parser("predict", help="predict one day's games")
    p.add_argument("--date", help="US Eastern date, YYYY-MM-DD (default: today)")
    p.add_argument("--commit", action="store_true", help="git commit and push the prediction file")
    s = sub.add_parser("score", help="grade predictions for finished games")
    s.add_argument("--commit", action="store_true", help="git commit and push the scoreboard")
    args = parser.parse_args()

    if args.cmd == "freeze":
        freeze()
    elif args.cmd == "predict":
        predict(args.date, args.commit)
    else:
        score_live(args.commit)


if __name__ == "__main__":
    main()
