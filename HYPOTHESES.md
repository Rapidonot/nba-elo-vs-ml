# Pre-registered hypotheses

## Main question

Does a machine-learning model (LightGBM) predict NBA game outcomes better
than Elo ratings, and if so, where does the improvement come from?

## Hypotheses

**H1. Elo vs. baseline.** Elo will clearly beat the home win-rate baseline on
log loss across the test seasons.

**H2. ML vs. Elo.** LightGBM with Elo as a feature (Model 3) will beat Elo
(Model 1), but by a small margin: a log-loss improvement of less than 0.005.

**H3. Algorithm vs. information.** LightGBM *without* Elo (Model 2) will perform
close to Model 3, because team ratings carry most of the same
information as Elo.

**H4. Feature groups.** In the ablation ladder
(team strength → four factors → situation → travel):
- team strength will provide most of the gain;
- situation features (rest, back-to-backs) will add a small but detectable gain;
- travel features will give no detectable improvement once rest and team
  strength are already included.

**H5. Timing.** Elo will be weakest early in each season (each team's first
20 games), when last season's ratings are out of date; the gap to LightGBM
will shrink as the season goes on.

**H6. Disrupted seasons.** Home-court advantage will be noticeably lower in
2020-21, so models tuned on earlier seasons will over-predict home wins there.

## What would count as "detectable"

A difference counts only if the 95% bootstrap interval for the log-loss
difference excludes zero.

## Notes added after registration

_(dated entries only)_

**2026-10-01, written before any LightGBM model was trained.** Step 1 showed
Elo over-predicting home wins in every test season: its home advantage was
tuned on 2015-16 to 2018-19, when home teams won more often than they do now.
LightGBM is retrained before each test season, so it can learn the lower home
advantage, which would make the Elo comparison unfair. I am adding
**Model 1b**: Elo whose home advantage updates a little after every game,
tuned on the same seasons as Model 1. H2 is still tested against Model 1, as
registered; Model 3 vs. Model 1b is reported alongside it as the fairer
comparison. H1 to H6 are unchanged.

**2026-10-01, written AFTER seeing step 2 results (exploratory, not
pre-registered).** Tuning on 2015-16 to 2018-19 chose a home-advantage
learning rate of 0 for Model 1b, so it ended up identical to Model 1 and did
not remove LightGBM's advantage of being retrained each season. I am adding
**Model 1c** as a robustness check: before each test season, Elo's K and home
advantage are re-tuned on all earlier non-warm-up seasons, exactly as
LightGBM is retrained. The home-advantage grid is widened to 0-125 Elo points
(steps of 25) so the grid cannot cap the answer. Model 1c is reported as
exploratory; the registered tests (H2 against Model 1) stand as they are.

**2026-10-01, written before any step 3 analysis was run.** How H4 to H6 will
be measured (hypotheses unchanged; this only fixes the definitions):
- *H4 ladder:* LightGBM rungs L0 = Elo features only, L1 = + team strength,
  L2 = + four factors, L3 = + situation, L4 = + travel (= Model 3). Every rung
  is tuned and walk-forward tested exactly like Model 3. "Most of the gain" =
  the L0 to L1 step gives more than half of the total L0 to L4 log-loss
  improvement. Situation and travel gains are judged by the bootstrap rule on
  the L2 to L3 and L3 to L4 steps. Leave-one-group-out (Model 3 minus each
  group) is reported as a check on the ladder order.
- *H5:* an "early" game is one where both teams have played fewer than 20
  games that season before tip-off; all other games are "later". The gap is
  Model 1's log loss minus Model 3's. H5 holds if the early gap is larger than
  the later gap and the bootstrap interval for (early gap minus later gap)
  excludes zero.
- *H6:* home-court advantage = share of games won by the home team. Compared
  for 2020-21 against the 2015-16 to 2018-19 average. "Over-predict" = Model 1's
  mean predicted home-win probability minus the actual home-win rate, with a
  bootstrap interval; reported for every test season so 2020-21 can be
  compared with the others.
