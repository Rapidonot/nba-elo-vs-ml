"""Checks that step 2 features only use information from before tip-off."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_GROUPS, build_features
from src.elo import EloConfig, run_elo

TEAMS = ["BOS", "NYK", "LAL", "DEN"]


def make_games(n=24, seed=0):
    """Round-robin fake season with random but realistic box scores."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        h, a = TEAMS[i % 4], TEAMS[(i + 1 + i // 4) % 4]
        if h == a:
            a = TEAMS[(i + 2) % 4]
        row = {"game_id": f"G{i:03d}", "season": "2021-22", "game_date": pd.Timestamp("2021-10-20") + pd.Timedelta(days=i),
               "home_team": h, "away_team": a, "home_team_id": TEAMS.index(h), "away_team_id": TEAMS.index(a)}
        for side in ("home", "away"):
            row.update({f"{side}_min": 240, f"{side}_fga": rng.integers(80, 95), f"{side}_fgm": rng.integers(35, 45),
                        f"{side}_fg3m": rng.integers(8, 16), f"{side}_fta": rng.integers(15, 30), f"{side}_ftm": rng.integers(10, 15),
                        f"{side}_oreb": rng.integers(6, 14), f"{side}_dreb": rng.integers(30, 40), f"{side}_tov": rng.integers(10, 18)})
            row[f"{side}_pts"] = 2 * row[f"{side}_fgm"] + row[f"{side}_fg3m"] + row[f"{side}_ftm"]
        row["home_win"] = int(row["home_pts"] > row["away_pts"])
        rows.append(row)
    return pd.DataFrame(rows)


def test_every_listed_feature_is_built():
    f = build_features(make_games())
    for group in FEATURE_GROUPS.values():
        assert set(group) <= set(f.columns)


def test_a_games_own_box_score_never_changes_its_features():
    games = make_games()
    changed = games.copy()
    k = 12
    changed.loc[k, ["home_pts", "home_fgm", "away_tov"]] += [30, 15, 10]  # rewrite game k's result
    a = build_features(games).set_index("game_id")
    b = build_features(changed).set_index("game_id")
    gid = games.loc[k, "game_id"]
    pd.testing.assert_series_equal(a.loc[gid], b.loc[gid])  # game k itself: unchanged
    later = games.index > k
    assert not a.loc[games.loc[later, "game_id"]].equals(b.loc[games.loc[later, "game_id"]])  # later games: changed


def test_first_game_of_season_has_no_rolling_stats():
    f = build_features(make_games()).set_index("game_id")
    assert np.isnan(f.loc["G000", "home_net_std"])
    assert f.loc["G000", "home_rest"] == 7


@pytest.mark.parametrize("home_k", [0.0, 2.0])
def test_learned_home_advantage_is_recorded_before_the_update(home_k):
    games = make_games()
    preds, _ = run_elo(games, EloConfig(home_k=home_k))
    assert preds.loc[0, "elo_home_adv_pre"] == EloConfig().home_adv
    if home_k == 0:
        assert preds["elo_home_adv_pre"].nunique() == 1
    else:
        assert preds["elo_home_adv_pre"].nunique() > 1
