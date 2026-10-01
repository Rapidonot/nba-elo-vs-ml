"""
Model 1: Elo ratings for NBA teams.

Each team has one number. Before each game we convert the rating gap
(plus a home-court bonus) into a win probability, then update both teams
based on the result and margin of victory.

Leakage rule: every prediction is recorded BEFORE that game's result is
used to update ratings, so each prediction only uses past information.
The pre-game ratings saved here are later reused as a feature for LightGBM.

Model 1b is the same model with `home_k > 0`: the home-court bonus is also
learned from results, so it can drift if home advantage changes over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0                # how fast ratings react to results
    home_adv: float = 100.0        # home-court bonus, in Elo points
    init: float = 1500.0           # rating for a team seen for the first time
    carryover: float = 0.75        # share of last season's rating kept in the new season
    revert_to: float = 1505.0      # league mean that ratings regress toward between seasons
    use_mov: bool = True           # scale updates by margin of victory
    home_k: float = 0.0            # Model 1b: how fast home_adv learns from results (0 = fixed)


def expected_score(rating_diff: float) -> float:
    """Win probability for the side that is `rating_diff` Elo points stronger."""
    return 1.0 / (1.0 + 10.0 ** (-rating_diff / 400.0))


def mov_multiplier(margin: float, winner_diff: float) -> float:
    """Bigger wins move ratings more, but less so when a strong favourite wins big
    (this stops Elo from inflating teams that beat weak opponents).
    `winner_diff` = winner's pre-game rating (incl. home bonus) minus loser's.
    """
    return ((abs(margin) + 3.0) ** 0.8) / (7.5 + 0.006 * winner_diff)


def run_elo(games: pd.DataFrame, cfg: EloConfig = EloConfig()) -> tuple[pd.DataFrame, dict]:
    """Walk through games in date order, predicting then updating.

    `games` needs: game_id, season, game_date, home_team_id, away_team_id,
    home_pts, away_pts, home_win.
    Returns (per-game pre-game ratings + probability, final ratings dict).
    """
    games = games.sort_values(["game_date", "game_id"])
    ratings: dict = {}
    season = None
    home_adv = cfg.home_adv  # only changes when cfg.home_k > 0 (Model 1b)
    rows = []

    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:  # new season: pull everyone partway back to the mean
                ratings = {t: cfg.carryover * r + (1 - cfg.carryover) * cfg.revert_to
                           for t, r in ratings.items()}
            season = g.season

        r_home = ratings.get(g.home_team_id, cfg.init)
        r_away = ratings.get(g.away_team_id, cfg.init)
        diff = r_home + home_adv - r_away
        p_home = expected_score(diff)
        rows.append((g.game_id, r_home, r_away, home_adv, p_home))  # recorded BEFORE the update

        mult = 1.0
        if cfg.use_mov:
            winner_diff = diff if g.home_win == 1 else -diff
            mult = mov_multiplier(g.home_pts - g.away_pts, winner_diff)
        delta = cfg.k * mult * (g.home_win - p_home)
        ratings[g.home_team_id] = r_home + delta
        ratings[g.away_team_id] = r_away - delta
        # Home teams winning more (less) than expected nudges home_adv up (down)
        home_adv += cfg.home_k * (g.home_win - p_home)

    preds = pd.DataFrame(rows, columns=["game_id", "elo_home_pre", "elo_away_pre", "elo_home_adv_pre", "p_elo"])
    return preds, ratings
