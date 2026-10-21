# Live scoreboard: 2026-27

Updated 2026-10-21 04:08 UTC. 2 finished games with predictions made before tip-off (3 predicted in total).

| model | n_games | accuracy | log_loss | brier |
|---|---|---|---|---|
| Model 1c: Elo | 2 | 0.5000 | 0.8829 | 0.3376 |
| Model 3: LightGBM + Elo | 2 | 0.5000 | 0.7953 | 0.2985 |

## Model 3 minus Model 1c (log loss, paired bootstrap)
Negative = Model 3 better. A gap counts only if the 95% interval excludes 0.

| mean_diff | ci_low | ci_high | share_A_better |
|---|---|---|---|
| -0.0877 | -0.1528 | -0.0225 | 1.0000 |

Home teams have won 50.0% of these games.
