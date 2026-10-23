# Live scoreboard: 2026-27

Updated 2026-10-23 04:00 UTC. 15 finished games with predictions made before tip-off (16 predicted in total).

| model | n_games | accuracy | log_loss | brier |
|---|---|---|---|---|
| Model 1c: Elo | 15 | 0.5333 | 0.6210 | 0.2192 |
| Model 3: LightGBM + Elo | 15 | 0.5333 | 0.6304 | 0.2209 |

## Model 3 minus Model 1c (log loss, paired bootstrap)
Negative = Model 3 better. A gap counts only if the 95% interval excludes 0.

| mean_diff | ci_low | ci_high | share_A_better |
|---|---|---|---|
| 0.0094 | -0.0380 | 0.0555 | 0.3395 |

Home teams have won 60.0% of these games.
