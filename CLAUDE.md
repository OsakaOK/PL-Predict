# CLAUDE.md

Guidance for working in this repository.

## What this project is

**PL-Predict** predicts the English Premier League **2026-2027** table — the
Champion, Top 4, and Relegation spots — from **2025-2026** season data using a
machine-learning model.

## Layout

[main.py](main.py) is a thin entry point that orchestrates the `pl_predict`
package, one module per pipeline stage:

- [pl_predict/config.py](pl_predict/config.py) — constants: API token/URL, cache
  dir, seasons, features.
- [pl_predict/fetch.py](pl_predict/fetch.py) — `fetch_season` (org API call +
  cache; provides the prediction base).
- [pl_predict/fetch_couk.py](pl_predict/fetch_couk.py) — `fetch_couk_matches`
  (football-data.co.uk historical CSVs, 1995-96 onward; provides the training
  data).
- [pl_predict/features.py](pl_predict/features.py) — `clean_matches`,
  `compute_team_stats` (raw JSON → per-team season stats).
- [pl_predict/model.py](pl_predict/model.py) — `build_training_data`,
  `train_model`, `evaluate_predictors` (baseline scoreboard), `predict_table`.
- [pl_predict/simulate.py](pl_predict/simulate.py) — Monte Carlo probabilities
  from empirical residuals.
- [pl_predict/report.py](pl_predict/report.py) — standalone HTML page.
- [experiments/feature_sweep.py](experiments/feature_sweep.py) — Phase 3
  harness: re-judges candidate features/alphas on the gate (not part of the
  pipeline; run it when considering a model change).

Imports are absolute (`from pl_predict.fetch import ...`), so run from the project
root with `python main.py`.

## How it works (pipeline)

Two data sources, never joined against each other:

- **Training** — [football-data.co.uk](https://www.football-data.co.uk/) free
  CSVs, every season since **1995-96** (the first 20-team/38-game season),
  cached to `data/couk_E0_<season>.csv`. Team names are the join keys; they are
  internally consistent across all seasons (audited: 49 clubs, no drift).
- **Prediction base** — the [football-data.org](https://www.football-data.org/)
  v4 API's 2025-26 season (stable integer ids, pretty display names), cached to
  `data/matches_<season>.json`.

1. **`fetch_couk_matches(season)` / `fetch_season(season)`** — Cached fetches;
   both yield the same tidy one-row-per-finished-match shape (`fetch_season`
   via `clean_matches`). An integration test asserts both sources produce
   identical per-team stats for the shared 2025-26 season.
2. **`compute_team_stats(matches)`** — Builds the full-season table per team:
   wins/draws/losses, goals, `points`, `goal_difference`, `win_percentage`,
   `recent_form` (points from the last 5 matches), and final `rank`.
3. **`build_training_data(...)`** — Pairs each team's **season-N features** with its
   **season-N+1 points** (`next_points`). Only teams present in *both* seasons form
   a training row (an inner join on `team_id`), since relegated/promoted teams have
   no continuation. 30 transitions × 17 continuing teams = **510 rows**.
4. **`train_model` + `evaluate_predictors`** — Fits `StandardScaler` + `Ridge`,
   reports leave-one-out CV metrics, and scores the model against naive
   baselines (see Roadmap) on the same split; the scoreboard prints and saves
   to `data/validation.csv`.
5. **`predict_table` + `simulate_probabilities`** — Applies the model to 2025-26
   stats to predict 2026-27 points, then Monte-Carlo-simulates 10,000 seasons
   from the empirical residuals for champion/top-4/relegation probabilities.
   `main` prints the table and saves `data/prediction_2026_2027.csv` and
   `index.html`.

## Design decisions worth knowing

- **Why Ridge on a tiny feature set.** With only ~34 training rows, plain
  `LinearRegression` on the full feature set (points, goal_difference,
  goals_scored, win_percentage, recent_form) is unstable: those features are
  collinear, producing wild coefficients and a nonsense result (it ranked
  Bournemouth above Man City). The current model uses a small, non-redundant set
  (`points`, `goal_difference`, `recent_form`) with standardisation + L2
  regularisation, which cross-validates better and gives a sensible ordering.
  **If you add features, re-check LOO-CV and the predicted ordering — don't
  reintroduce collinear columns.**
- **Honest metrics.** Quality is reported via leave-one-out CV
  (`cross_val_predict`), not in-sample fit, and always against the baseline
  scoreboard. On 510 rows expect Spearman ~0.69, MAE ~8.4, R² ~0.58 — and note
  persistence alone scores Spearman ~0.68, so the model's edge is real but
  slim. Don't "improve" this by reporting in-sample numbers or dropping the
  baselines.
- **Promoted-team limitation.** Neither free source has usable lower-division
  data, so the 3 promoted clubs can't be predicted. Output covers continuing
  teams only — stated in the README and the printed header. The relegation label
  here is therefore the 3 weakest *continuing* teams, not a true bottom-3.
- **Joins use `team_id`, not names — within one source.** org ids are stable
  integers; co.uk "ids" are its team names (audited consistent 1995-2025).
  Never join the two sources against each other: training and prediction only
  share feature *columns*, not keys.

## Running it

```bash
pip install -r requirements.txt
export FOOTBALL_DATA_TOKEN=your_token   # optional; falls back to committed key
python main.py
```

First run fetches and caches; later runs reuse `data/`. To force a refetch, delete
the relevant `data/matches_*.json`.

## Testing

```bash
python3 -m pytest tests/ -q   # run from the project root (absolute imports)
```

Unit tests in [tests/](tests/) run on hand-built fixtures, fully offline:
per-team stats math ([tests/test_features.py](tests/test_features.py)),
training-data joins / scoreboard invariants
([tests/test_model.py](tests/test_model.py)), and Monte Carlo probability
coherence — P(champion) sums to 1, P(top-4) to 4, P(relegation) to 3, seeded
determinism ([tests/test_simulate.py](tests/test_simulate.py)).
[tests/test_pipeline_integration.py](tests/test_pipeline_integration.py) runs
the whole pipeline against the cached real seasons and is auto-skipped when
`data/` is empty (fresh clone). Keep new behaviour covered by these invariants
— especially the scoreboard shape, which Phase 2 experiments depend on.

## Gotchas

- **API token.** A key is committed inline at [main.py](main.py) as a fallback. It
  is public in git history and should be rotated; prefer the `FOOTBALL_DATA_TOKEN`
  env var.
- **Rate limit.** Free tier is ~10 requests/min; `fetch_season` sleeps 6s after a
  live fetch. The cache means this rarely bites.
- **Adding seasons.** Each July, bump `LAST_COMPLETED_SEASON` (which also extends
  `COUK_SEASONS`) and update `PREDICT_SEASON_LABEL`. Don't extend `COUK_SEASONS`
  before 1995 — earlier seasons had 22 teams / 42 games, a different points scale.
- **co.uk CSV quirks.** Some seasons are latin-1, some rows have extra trailing
  betting columns, and the column layout shifts across eras — the parser in
  [pl_predict/fetch_couk.py](pl_predict/fetch_couk.py) locates fields by header
  name and must stay tolerant of ragged rows.

## Roadmap: evaluation gate + measurable accuracy

The project's priority order is **trustworthiness first, accuracy second**: prove
the model earns its keep before trying to make it better. Accuracy changes only
count if they move the evaluation scoreboard.

### Phase 1 — the gate + uncertainty output (current data)

1. **Baselines** ([pl_predict/model.py](pl_predict/model.py)) — three reference
   predictors evaluated under the *same* LOO-CV split as Ridge:
   - **mean** — predict training-mean points for everyone (the R²=0 floor).
   - **persistence** — next points = this season's points (the real bar to beat).
   - **regress-to-mean** — linear fit on `points` alone; diagnoses whether Ridge
     does anything beyond statistical shrinkage.
2. **Scoreboard metrics** — per predictor: **Spearman ρ** (per `from_season`,
   averaged) and **top-4 hit rate** as primary criteria; MAE and R² as secondary.
   The model must win Spearman + top-4 to justify itself. Champion accuracy and
   probabilistic scores (Brier etc.) are *excluded* as criteria — with only 2
   season transitions they are anecdotes, not metrics.
3. **Monte Carlo probabilities** ([pl_predict/simulate.py](pl_predict/simulate.py)) —
   resample the model's **empirical LOO-CV residuals** (no Gaussian assumption;
   real season-over-season swings supply the upset asymmetry) onto predicted
   points, ~10,000 simulated tables → P(champion), P(top-4), P(relegation).
4. **Outputs — augment, don't replace.** The point-estimate ranking stays the
   spine. Console/CSV/HTML gain P(top-4) and P(relegation) columns; the HTML
   champion card shows P(champion); the scoreboard prints in full on the console
   and persists to `data/validation.csv`; the HTML footer carries one honest
   line comparing model vs. persistence on rank — **stated even when the model
   loses**. The gate tells the truth or it isn't a gate.

### Phase 2 — expand training data (DONE, 2026-07)

- **Result: the gate flipped.** On the original 34 rows the model *lost* to
  persistence on rank (Spearman 0.524 vs 0.536). On 510 rows (30 transitions,
  1995-96 onward, via football-data.co.uk) it wins on every metric:
  Spearman **0.688 vs 0.677**, top-4 hit rate **0.750 vs 0.742**,
  MAE **8.38 vs 8.87**, R² **0.58 vs 0.51**. The edge over persistence is real
  but slim — that's the honest state of season-ahead prediction.
- The feared name→id reconciliation turned out unnecessary: co.uk names are
  internally consistent across all 31 seasons (audited), and training never
  joins against the org source — only the feature columns are shared. A test
  asserts both sources yield identical stats for the shared season.

### Phase 3 — feature experiments (RUN 2026-07; verdict: keep current model)

Eleven candidate feature sets and an alpha sweep were judged on the gate
(same LOO-CV split, primary = Spearman + top-4 vs the *current* model):

- **No candidate adopted.** The best, `base + momentum` (2nd-half minus
  1st-half points), edged Spearman 0.694 vs 0.688 but won only 18 of 28
  seasons in a paired per-season test (sign-test p=0.185 — not
  distinguishable from luck) and worsened champion hit. Adopting on that
  evidence would be the overclaiming the gate exists to prevent.
- `gd only` and `points+gd` (dropping recent_form) lose more seasons than
  they win; the kitchen sink underperforms base — the collinearity warning
  above still holds even at 510 rows. Alpha is flat around 5.0.
- **Adopted from this phase:** `champion_hit_rate` as a scoreboard
  *diagnostic* (currently 0.43 for the model vs 0.37 persistence) — never a
  decision criterion; top-1 over 30 transitions is a coin-flip-per-season.
- **Momentum is the candidate to retest** when more transitions accumulate
  (one per July) or with a stronger significance protocol.
- A positive validation fell out too: `points only` is *significantly* worse
  (7-23 seasons, p=0.005), so `goal_difference` and `recent_form` do earn
  their places.

### Judging future changes

- Every change is judged on the scoreboard; keep only what beats the current
  model on Spearman + top-4, and check the win is consistent across seasons
  (paired per-season comparison), not just on the pooled average.
- The whole protocol is automated in
  [experiments/feature_sweep.py](experiments/feature_sweep.py) — add the
  candidate there and run `python3 experiments/feature_sweep.py`.

## Conventions

- Single script, organised into small functions; standard data-science stack
  (`requests`, `pandas`, `scikit-learn`) pinned loosely in `requirements.txt`.
- `data/` (cache + generated CSV) and `__pycache__/` are git-ignored.
