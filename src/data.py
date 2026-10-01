"""
Download NBA regular-season results with nba_api and save one row per game.

Usage (from the repo root):
    python -m src.data                    # seasons 2014-15 .. 2025-26
    python -m src.data --start 2014 --end 2025

Source: stats.nba.com via the open-source `nba_api` package.
The raw data is NOT committed to this repo (see .gitignore). Anyone can
regenerate it with this script. Used for non-commercial research only.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "games.csv"

# Team box-score totals kept for each side (used for ratings and four factors in step 2)
BOX_COLS = ["MIN", "FGM", "FGA", "FG3M", "FTM", "FTA", "OREB", "DREB", "TOV"]


def season_label(start_year: int) -> str:
    """2014 -> '2014-15'."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def fetch_season(season: str, retries: int = 4, pause: float = 2.0) -> pd.DataFrame:
    """Fetch every team-game row for one regular season (two rows per game)."""
    from nba_api.stats.endpoints import leaguegamefinder  # lazy import: tests don't need it

    for attempt in range(1, retries + 1):
        try:
            frame = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable="Regular Season",
                timeout=60,
            ).get_data_frames()[0]
            time.sleep(pause)  # be polite to the API
            return frame
        except Exception as err:  # network hiccups / rate limiting
            if attempt == retries:
                raise
            wait = pause * attempt * 2
            print(f"  {season}: attempt {attempt} failed ({err}); retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def team_rows_to_games(rows: pd.DataFrame, season: str) -> pd.DataFrame:
    """Pair the home row and away row of each game into a single game row.

    In the MATCHUP column, 'BOS vs. NYK' and 'NYK @ BOS' both mean BOS is at home.
    The home team is read from the matchup text rather than from which separator a
    row uses, because some games (e.g. the 2025 Paris games) list the same
    'IND @ SAS' matchup on both teams' rows.
    """
    rows = rows.copy()
    parts = rows["MATCHUP"].str.extract(r"^(\w+) (vs\.|@) (\w+)$")
    if parts.isna().any().any():
        raise ValueError(f"{season}: unexpected MATCHUP format: {rows.loc[parts[0].isna(), 'MATCHUP'].unique()}")
    home_abbr = parts[0].where(parts[1] == "vs.", parts[2])
    rows["is_home"] = rows["TEAM_ABBREVIATION"] == home_abbr
    per_game = rows.groupby("GAME_ID")["is_home"].agg(["sum", "size"])
    bad = per_game[(per_game["sum"] != 1) | (per_game["size"] != 2)]
    if not bad.empty:
        raise ValueError(f"{season}: games without exactly one home and one away row: {list(bad.index[:5])}")
    box = [c for c in BOX_COLS if c in rows.columns]  # tests may pass minimal rows
    home = rows.loc[rows["is_home"], ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", *box]]
    away = rows.loc[~rows["is_home"], ["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", *box]]

    games = home.merge(away, on="GAME_ID", suffixes=("_home", "_away"), validate="one_to_one")
    games = games.rename(columns={f"{c}_{side}": f"{side}_{c.lower()}" for c in box for side in ("home", "away")})
    games = games.rename(columns={
        "GAME_ID": "game_id",
        "GAME_DATE": "game_date",
        "TEAM_ID_home": "home_team_id",
        "TEAM_ABBREVIATION_home": "home_team",
        "PTS_home": "home_pts",
        "TEAM_ID_away": "away_team_id",
        "TEAM_ABBREVIATION_away": "away_team",
        "PTS_away": "away_pts",
    })
    games["season"] = season
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["home_win"] = (games["home_pts"] > games["away_pts"]).astype(int)
    return games.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fetch_schedule(season: str) -> pd.DataFrame:
    """Regular-season schedule (played and upcoming), one row per game.

    Games whose teams aren't known yet (e.g. NBA Cup knockouts) are dropped.
    `game_date` is the US Eastern date; `tipoff_utc` is the scheduled start.
    """
    from nba_api.stats.endpoints import scheduleleaguev2

    s = scheduleleaguev2.ScheduleLeagueV2(season=season, league_id="00", timeout=60).get_data_frames()[0]
    s = s[s["gameId"].str.startswith("002")].dropna(subset=["homeTeam_teamTricode", "awayTeam_teamTricode"])
    return pd.DataFrame({
        "game_id": s["gameId"],
        "game_date": pd.to_datetime(s["gameDateEst"].str[:10]),
        "tipoff_utc": pd.to_datetime(s["gameDateTimeUTC"], utc=True),
        "home_team_id": s["homeTeam_teamId"].astype(int), "home_team": s["homeTeam_teamTricode"],
        "away_team_id": s["awayTeam_teamId"].astype(int), "away_team": s["awayTeam_teamTricode"],
        "status": s["gameStatus"].astype(int),  # 1 = not started, 2 = in progress, 3 = final
        "season": season,
    }).reset_index(drop=True)


def sanity_check(games: pd.DataFrame, season: str) -> None:
    """Warn (don't fail) if a team's game count looks unusual for the season."""
    counts = pd.concat([games["home_team"], games["away_team"]]).value_counts()
    median = counts.median()
    odd = counts[(counts - median).abs() > 2]
    print(f"  {season}: {len(games)} games, {counts.size} teams, median {median:.0f} games/team")
    if not odd.empty:
        # Expected for 2019-20 (season cut short) and possibly NBA Cup knockout games.
        print(f"    note: unusual game counts -> {odd.to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NBA regular-season games.")
    parser.add_argument("--start", type=int, default=2014, help="first season start year")
    parser.add_argument("--end", type=int, default=2025, help="last season start year")
    args = parser.parse_args()

    all_games = []
    for year in range(args.start, args.end + 1):
        season = season_label(year)
        print(f"Fetching {season} ...")
        games = team_rows_to_games(fetch_season(season), season)
        sanity_check(games, season)
        all_games.append(games)

    out = pd.concat(all_games, ignore_index=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW_PATH, index=False)
    print(f"Saved {len(out):,} games to {RAW_PATH}")


if __name__ == "__main__":
    main()
