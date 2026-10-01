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
