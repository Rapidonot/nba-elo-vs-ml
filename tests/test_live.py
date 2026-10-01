"""Live predictions must match what the backtest code gives for the same game."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.elo import EloConfig, run_elo
from src.features import build_features
from src.live import FEATURES, predict_upcoming
from tests.test_features import make_games


@pytest.fixture
def booster():
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(300, len(FEATURES))), columns=FEATURES)
    y = (x["elo_diff"] + rng.normal(size=300) > 0).astype(int)
    return lgb.LGBMClassifier(n_estimators=10, num_leaves=4, verbose=-1).fit(x, y).booster_


def test_live_prediction_matches_prediction_made_with_result_known(booster):
    games = make_games(24)
    cfg_1c, cfg_feat = EloConfig(k=15, home_adv=50), EloConfig(k=15, home_adv=75)
    history, target = games.iloc[:20], games.iloc[20]

    # Live: the target game is appended WITHOUT its result or box score
    upcoming = target[["game_id", "season", "game_date", "home_team", "away_team",
                       "home_team_id", "away_team_id"]].to_frame().T.infer_objects()
    live = predict_upcoming(history, upcoming, cfg_1c, cfg_feat, booster).iloc[0]

    # Backtest-style: the target game is in the data WITH its real result
    known = games.iloc[:21]
    p1c = run_elo(known, cfg_1c)[0].set_index("game_id").loc[target["game_id"], "p_elo"]
    pf = run_elo(known, cfg_feat)[0].set_index("game_id").loc[target["game_id"]]
    x = build_features(known).set_index("game_id").loc[[target["game_id"]]]
    x["elo_home_pre"], x["elo_away_pre"] = pf["elo_home_pre"], pf["elo_away_pre"]
    x["elo_diff"] = x["elo_home_pre"] - x["elo_away_pre"]

    assert live["p_model1c"] == pytest.approx(p1c)
    assert live["p_model3"] == pytest.approx(booster.predict(x[FEATURES])[0])
