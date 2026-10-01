# Elo vs. Machine Learning: Predicting NBA Games

Can modern machine learning beat a decades-old chess rating system at
predicting NBA games, and if it does, where does the improvement come from?

This project compares an Elo rating model against LightGBM gradient boosting
on more than ten seasons of NBA regular-season games, using the same data
splits and the same scorecard for every model. It then runs live, timestamped
predictions through the 2026-27 season.

## Study design

| Part | What it is | Why it's there |
|---|---|---|
| Model 0 | Home win rate from earlier seasons | The floor any model must beat |
| Model 1 | Elo ratings | Simple, transparent benchmark |
| Model 2 | LightGBM without Elo | Isolates what the algorithm adds |
| Model 3 | LightGBM with Elo as a feature | Headline challenger |
| Ablation ladder | Model 3 built up feature group by group | Shows where the gains come from |
| Leave-one-group-out | Model 3 with each group removed in turn | Checks the ladder isn't an artefact of order |

**Feature groups (step 2):**
team strength (Elo, rolling offensive/defensive/net rating) ·
style (pace, eFG%, turnover %, offensive rebound %, free-throw rate) ·
situation (home/away, rest days, back-to-back, games in last 7 days) ·
opponent-relative (rating differences between the two teams) ·
travel (distance, time-zone change, consecutive road games).

Hypotheses were written down before any experiments: see
[HYPOTHESES.md](HYPOTHESES.md).

## Evaluation

- **Seasons:** 2014-15 warm-up · 2015-16 to 2018-19 tuning ·
  2019-20 onwards testing. Test seasons are never used to choose settings.
- **Metrics:** log loss (primary), Brier score, accuracy, calibration plots.
- **Is a gap real?** Paired bootstrap on per-game log-loss differences;
  a difference counts only if the 95% interval excludes zero.
- **No leakage:** every prediction uses only information available before tip-off.
- **Disrupted seasons:** 2019-20 (bubble) and 2020-21 (few or no fans) are
  reported both included and excluded, because home advantage changed.

## Progress

- [x] Step 1: data pipeline, Model 0, Model 1 (Elo), backtest
- [ ] Step 2: feature engineering, Models 2 and 3
- [ ] Step 3: ablation ladder, leave-one-group-out, significance tests
- [ ] Step 4: live 2026-27 prediction log
- [ ] Step 5: write-up, demo app, video

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                 # check the Elo model and leakage rules
python -m src.data               # download games (a few minutes)
python -m src.backtest           # Model 0 vs. Model 1 -> results/
```

Results are written to `results/` (summary, per-season metrics, significance
tests, calibration plot).

## Repository layout

```
src/data.py       download and clean game results
src/elo.py        Model 1: Elo ratings
src/evaluate.py   shared metrics, bootstrap test, calibration plot
src/backtest.py   step 1 experiment
tests/            unit tests (Elo correctness, no leakage)
results/          generated metrics and figures
```

## Data and limitations

- Game data comes from stats.nba.com via the open-source
  [`nba_api`](https://github.com/swar/nba_api) package, used for
  non-commercial research. Raw data is not stored in this repository;
  `src/data.py` regenerates it.
- Neutral-site games (international games, NBA Cup knockouts) are treated
  as home games for the listed home team.
- The bootstrap treats games as independent, which is an approximation.
- Injuries and confirmed lineups are not modelled in the current version.
