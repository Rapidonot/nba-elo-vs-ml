"""
Step 2 features: everything LightGBM knows about a game before tip-off.

Feature groups follow the ablation ladder in HYPOTHESES.md (H4):
    team strength -> four factors -> situation -> travel
Elo ratings are kept separate, because Model 2 (LightGBM without Elo) must
not see them and Model 3 (LightGBM with Elo) adds them on top.

Leakage rule: every rolling statistic is shifted by one game, so a game's own
box score never feeds its own features. Rolling windows reset each season;
the previous season's net rating is carried over for the cold start.

Possessions and the "four factors" use the standard Basketball-Reference
formulas, computed from team box-score totals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Arena location (lat, lon) and standard UTC offset (no daylight saving).
# Current arenas are used for the whole period; the few moves since 2014
# (DET, GSW, SAC, LAC) are within ~40 km of the listed site.
ARENAS = {
    "ATL": (33.757, -84.396, -5), "BKN": (40.683, -73.975, -5), "BOS": (42.366, -71.062, -5),
    "CHA": (35.225, -80.839, -5), "CHI": (41.881, -87.674, -6), "CLE": (41.496, -81.688, -5),
    "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7), "DET": (42.341, -83.055, -5),
    "GSW": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.156, -5),
    "LAC": (34.043, -118.267, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6),
    "MIA": (25.781, -80.188, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.980, -93.276, -6),
    "NOP": (29.949, -90.082, -6), "NYK": (40.751, -73.993, -5), "OKC": (35.463, -97.515, -6),
    "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SAS": (29.427, -98.438, -6),
    "TOR": (43.643, -79.379, -5), "UTA": (40.768, -111.901, -7), "WAS": (38.898, -77.021, -5),
}
# 2019-20 restart: every game from July 2020 was played in the Orlando bubble
BUBBLE = (28.337, -81.556, -5)
BUBBLE_START = pd.Timestamp("2020-07-01")

BOX = ["min", "fgm", "fga", "fg3m", "ftm", "fta", "oreb", "dreb", "tov"]
RATING_STATS = ["ortg", "drtg", "net"]
STYLE_STATS = ["pace", "efg", "tov_pct", "oreb_pct", "ftr", "opp_efg", "opp_tov_pct", "dreb_pct", "opp_ftr"]


def _both(names):
    return [f"{side}_{n}" for side in ("home", "away") for n in names]


FEATURE_GROUPS = {
    "team_strength": _both([f"{s}_{w}" for s in RATING_STATS for w in ("std", "l10")] + ["prev_net", "games_played"])
                     + ["net_std_diff", "net_l10_diff", "prev_net_diff"],
    "four_factors": _both([f"{s}_std" for s in STYLE_STATS]),
    "situation": _both(["rest", "b2b", "games_last7"]) + ["rest_diff"],
    "travel": _both(["travel_km", "tz_change"]) + ["away_road_streak"],
}
ELO_FEATURES = ["elo_home_pre", "elo_away_pre", "elo_diff"]


def team_games(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game (two rows per game), with that game's own stats."""
    sides = []
    for side, opp in (("home", "away"), ("away", "home")):
        t = pd.DataFrame({
            "game_id": games["game_id"], "season": games["season"], "game_date": games["game_date"],
            "team_id": games[f"{side}_team_id"], "venue": games["home_team"], "is_home": side == "home",
            "pts": games[f"{side}_pts"], "opp_pts": games[f"{opp}_pts"],
        })
        for c in BOX:
            t[c] = games[f"{side}_{c}"]
            t[f"opp_{c}"] = games[f"{opp}_{c}"]
        sides.append(t)
    return pd.concat(sides, ignore_index=True).sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)


def per_game_stats(t: pd.DataFrame) -> pd.DataFrame:
    """Ratings and four factors for each single game (NOT yet safe to use as features)."""
    poss_own = t["fga"] - t["oreb"] + t["tov"] + 0.44 * t["fta"]
    poss_opp = t["opp_fga"] - t["opp_oreb"] + t["opp_tov"] + 0.44 * t["opp_fta"]
    poss = (poss_own + poss_opp) / 2
    t["ortg"] = 100 * t["pts"] / poss
    t["drtg"] = 100 * t["opp_pts"] / poss
    t["net"] = t["ortg"] - t["drtg"]
    t["pace"] = poss * 240 / t["min"]  # possessions per 48 minutes (team minutes = 5 x 48)
    t["efg"] = (t["fgm"] + 0.5 * t["fg3m"]) / t["fga"]
    t["tov_pct"] = t["tov"] / (t["fga"] + 0.44 * t["fta"] + t["tov"])
    t["oreb_pct"] = t["oreb"] / (t["oreb"] + t["opp_dreb"])
    t["ftr"] = t["ftm"] / t["fga"]
    t["opp_efg"] = (t["opp_fgm"] + 0.5 * t["opp_fg3m"]) / t["opp_fga"]
    t["opp_tov_pct"] = t["opp_tov"] / (t["opp_fga"] + 0.44 * t["opp_fta"] + t["opp_tov"])
    t["dreb_pct"] = t["dreb"] / (t["dreb"] + t["opp_oreb"])
    t["opp_ftr"] = t["opp_ftm"] / t["opp_fga"]
    return t


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _games_in_prior_7_days(dates: pd.Series) -> np.ndarray:
    s = pd.Series(1.0, index=pd.DatetimeIndex(dates.values))
    return s.rolling("7D", closed="left").sum().fillna(0).to_numpy()


def pregame_team_features(t: pd.DataFrame) -> pd.DataFrame:
    """Turn per-game stats into features that only use earlier games."""
    by_season = t.groupby(["team_id", "season"], sort=False)

    # Team strength and style: season-to-date and last-10 averages of PREVIOUS games
    for s in RATING_STATS + STYLE_STATS:
        prior = by_season[s].shift()
        t[f"{s}_std"] = prior.groupby([t["team_id"], t["season"]]).transform(lambda x: x.expanding().mean())
        if s in RATING_STATS:
            t[f"{s}_l10"] = prior.groupby([t["team_id"], t["season"]]).transform(
                lambda x: x.rolling(10, min_periods=3).mean())
    t["games_played"] = by_season.cumcount()

    # Cold start: last season's average net rating
    seasons = sorted(t["season"].unique())
    prev_of = dict(zip(seasons[1:], seasons[:-1]))
    season_net = t.groupby(["team_id", "season"])["net"].mean().to_dict()
    t["prev_net"] = [season_net.get((team, prev_of.get(s)), np.nan) for team, s in zip(t["team_id"], t["season"])]

    # Situation: rest days (season openers count as fully rested), back-to-backs, recent load
    t["rest"] = by_season["game_date"].diff().dt.days.fillna(7).clip(upper=7)
    t["b2b"] = (t["rest"] == 1).astype(int)
    t["games_last7"] = t.groupby("team_id", sort=False)["game_date"].transform(_games_in_prior_7_days)

    # Travel: where this game is played vs. where the team played last
    loc = t["venue"].map(ARENAS)
    if loc.isna().any():
        raise ValueError(f"no arena for: {sorted(t.loc[loc.isna(), 'venue'].unique())}")
    loc = loc.where(~((t["season"] == "2019-20") & (t["game_date"] >= BUBBLE_START)), pd.Series([BUBBLE] * len(t)))
    t["lat"], t["lon"], t["tz"] = zip(*loc)
    prev = by_season[["lat", "lon", "tz"]].shift()
    t["travel_km"] = haversine_km(prev["lat"], prev["lon"], t["lat"], t["lon"]).fillna(0)
    t["tz_change"] = (t["tz"] - prev["tz"]).fillna(0)
    new_block = t["is_home"] != by_season["is_home"].shift()
    streak = t.groupby(new_block.cumsum()).cumcount() + 1
    t["road_streak"] = np.where(t["is_home"], 0, streak)
    return t


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """One row per game: game_id plus every feature in FEATURE_GROUPS."""
    t = pregame_team_features(per_game_stats(team_games(games)))
    keep = ([f"{s}_{w}" for s in RATING_STATS for w in ("std", "l10")] + [f"{s}_std" for s in STYLE_STATS]
            + ["prev_net", "games_played", "rest", "b2b", "games_last7", "travel_km", "tz_change", "road_streak"])
    home = t[t["is_home"]].set_index("game_id")[keep].add_prefix("home_")
    away = t[~t["is_home"]].set_index("game_id")[keep].add_prefix("away_")
    f = home.join(away, how="inner")
    f["net_std_diff"] = f["home_net_std"] - f["away_net_std"]
    f["net_l10_diff"] = f["home_net_l10"] - f["away_net_l10"]
    f["prev_net_diff"] = f["home_prev_net"] - f["away_prev_net"]
    f["rest_diff"] = f["home_rest"] - f["away_rest"]
    cols = [c for group in FEATURE_GROUPS.values() for c in group]
    return f[cols].reset_index()
