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
| Model 1 | Elo ratings, home advantage fixed | Simple, transparent benchmark (as pre-registered) |
| Model 1b | Elo, home advantage learned game by game | Fairness check |
| Model 1c | Elo re-tuned before each season | Exploratory: gives Elo the same "retrain every season" treatment as LightGBM |
| Model 2 | LightGBM without Elo | Isolates what the algorithm adds |
| Model 3 | LightGBM with Elo as a feature | Headline challenger |
| Ablation ladder | Model 3 built up feature group by group | Shows where the gains come from |
| Leave-one-group-out | Model 3 with each group removed in turn | Checks the ladder isn't an artefact of order |

**Feature groups** (all computed from games before tip-off; see `src/features.py`):
team strength (rolling offensive/defensive/net rating, last season's net rating, rating differences) ·
four factors (pace, eFG%, turnover %, rebound %, free-throw rate, for and against) ·
situation (rest days, back-to-backs, games in last 7 days) ·
travel (distance, time-zone change, consecutive road games).

Hypotheses were written down before any experiments: see
[HYPOTHESES.md](HYPOTHESES.md). Every later change to the design is recorded
there as a dated note, committed before the analysis it affects.

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
- [x] Step 2: feature engineering, Models 1b, 1c, 2 and 3
- [x] Step 3: ablation ladder, leave-one-group-out, early-season and home-court analyses
- [ ] Step 4: live 2026-27 prediction log
- [ ] Step 5: write-up, demo app, video

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest                 # check the Elo model and leakage rules
python -m src.data               # download games (a few minutes)
python -m src.backtest           # step 1: Model 0 vs. Model 1 -> results/
python -m src.backtest_ml        # step 2: Elo vs. LightGBM (about 1 minute)
python -m src.analysis           # step 3: ablation, early season, home court (about 2 minutes)
```

On macOS, LightGBM also needs OpenMP: `brew install libomp`.

Results are written to `results/` (summary, per-season metrics, significance
tests, calibration plot).

## Repository layout

```
src/data.py       download and clean game results
src/elo.py        Models 1 and 1b: Elo ratings
src/features.py   pre-game features for LightGBM
src/evaluate.py   shared metrics, bootstrap tests, calibration plot
src/backtest.py   step 1 experiment
src/backtest_ml.py  step 2 experiment (Models 1b, 1c, 2, 3)
src/analysis.py   step 3 analyses (H4-H6)
tests/            unit tests (Elo correctness, no leakage in Elo or features)
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
