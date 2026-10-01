"""Basic checks that the Elo model behaves correctly and never peeks at results."""
import pandas as pd
import pytest

from src.elo import EloConfig, expected_score, run_elo


def make_games(results):
    """Tiny fake schedule: team 1 hosts team 2 repeatedly. results = list of (home_pts, away_pts)."""
    return pd.DataFrame([
        {"game_id": f"G{i}", "season": "2020-21", "game_date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=i),
         "home_team_id": 1, "away_team_id": 2, "home_pts": h, "away_pts": a, "home_win": int(h > a)}
        for i, (h, a) in enumerate(results)
    ])


def test_probabilities_are_symmetric():
    assert expected_score(0) == pytest.approx(0.5)
    assert expected_score(120) + expected_score(-120) == pytest.approx(1.0)


def test_rating_points_are_conserved():
    _, ratings = run_elo(make_games([(110, 100), (95, 105), (120, 90)]))
    assert sum(ratings.values()) == pytest.approx(2 * EloConfig().init)


def test_prediction_does_not_use_its_own_result():
    """Changing game 2's result must not change game 2's prediction (only later ones)."""
    a, _ = run_elo(make_games([(110, 100), (120, 90), (100, 100 + 1)]))
    b, _ = run_elo(make_games([(110, 100), (90, 120), (100, 100 + 1)]))
    assert a.loc[1, "p_elo"] == pytest.approx(b.loc[1, "p_elo"])
    assert a.loc[2, "p_elo"] != pytest.approx(b.loc[2, "p_elo"])


def test_first_game_uses_initial_ratings_plus_home_advantage():
    preds, _ = run_elo(make_games([(110, 100)]))
    assert preds.loc[0, "p_elo"] == pytest.approx(expected_score(EloConfig().home_adv))
