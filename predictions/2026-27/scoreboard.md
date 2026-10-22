# Live scoreboard: 2026-27

Updated 2026-10-22 04:00 UTC. 11 finished games with predictions made before tip-off (14 predicted in total).

| model | n_games | accuracy | log_loss | brier |
|---|---|---|---|---|
| Model 1c: Elo | 11 | 0.4545 | 0.6743 | 0.2448 |
| Model 3: LightGBM + Elo | 11 | 0.4545 | 0.6618 | 0.2365 |

## Model 3 minus Model 1c (log loss, paired bootstrap)
Negative = Model 3 better. A gap counts only if the 95% interval excludes 0.

| mean_diff | ci_low | ci_high | share_A_better |
|---|---|---|---|
| -0.0125 | -0.0638 | 0.0391 | 0.6845 |

Home teams have won 45.5% of these games.
