# Live scoreboard: 2026-27

Updated 2026-10-24 04:00 UTC. 23 finished games with predictions made before tip-off (28 predicted in total).

| model | n_games | accuracy | log_loss | brier |
|---|---|---|---|---|
| Model 1c: Elo | 23 | 0.5217 | 0.6247 | 0.2201 |
| Model 3: LightGBM + Elo | 23 | 0.5652 | 0.6304 | 0.2207 |

## Model 3 minus Model 1c (log loss, paired bootstrap)
Negative = Model 3 better. A gap counts only if the 95% interval excludes 0.

| mean_diff | ci_low | ci_high | share_A_better |
|---|---|---|---|
| 0.0056 | -0.0295 | 0.0401 | 0.3895 |

Home teams have won 47.8% of these games.
