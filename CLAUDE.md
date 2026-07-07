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
- [pl_predict/fetch.py](pl_predict/fetch.py) — `fetch_season` (API call + cache).
- [pl_predict/features.py](pl_predict/features.py) — `clean_matches`,
  `compute_team_stats` (raw JSON → per-team season stats).
- [pl_predict/model.py](pl_predict/model.py) — `build_training_data`,
  `train_model`, `evaluate_predictors` (baseline scoreboard), `predict_table`.
- [pl_predict/simulate.py](pl_predict/simulate.py) — Monte Carlo probabilities
  from empirical residuals.
- [pl_predict/report.py](pl_predict/report.py) — standalone HTML page.

Imports are absolute (`from pl_predict.fetch import ...`), so run from the project
root with `python main.py`.

## How it works (pipeline)

1. **`fetch_season(season)`** — Pulls finished matches from the
   [football-data.org](https://www.football-data.org/) v4 API, cached to
   `data/matches_<season>.json` so reruns need no network. The free tier exposes
   seasons **2023, 2024, 2025** (the 2023-24, 2024-25, 2025-26 campaigns).
2. **`clean_matches(matches)`** — Flattens the nested match JSON into one tidy row
   per finished match, keyed by stable team **ids** (used for joins) plus display
   names.
3. **`compute_team_stats(matches)`** — Builds the full-season table per team:
   wins/draws/losses, goals, `points`, `goal_difference`, `win_percentage`,
   `recent_form` (points from the last 5 matches), and final `rank`.
4. **`build_training_data(...)`** — Pairs each team's **season-N features** with its
   **season-N+1 points** (`next_points`). Only teams present in *both* seasons form
   a training row (an inner join on `team_id`), since relegated/promoted teams have
   no continuation. This yields ~34 rows from the two available transitions.
5. **`train_model` + `evaluate_predictors`** — Fits `StandardScaler` + `Ridge`,
   reports leave-one-out CV metrics, and scores the model against naive
   baselines (see Roadmap) on the same split; the scoreboard prints and saves
   to `data/validation.csv`.
6. **`predict_table` + `simulate_probabilities`** — Applies the model to 2025-26
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
  scoreboard. Expect a modest R² (~0.25, MAE ~10 points) — predicting football
  a season ahead from one season of stats is inherently noisy. Don't "improve"
  this by reporting in-sample numbers or dropping the baselines.
- **Promoted-team limitation.** The free tier has no lower-division data, so the 3
  promoted clubs can't be predicted. Output covers continuing teams only — stated
  in the README and the printed header. The relegation label here is therefore the
  3 weakest *continuing* teams, not a true bottom-3.
- **Joins use `team_id`, not names.** Team display names can vary; ids are stable
  across seasons. Keep joins on `team_id`.

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
- **Adding seasons.** When football-data grants more historical seasons, add them to
  `SEASONS` — more transitions means more training data and a better model.

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

### Phase 2 — real accuracy (pending; gated by Phase 1 results)

- The binding constraint is ~34 training rows; feature/model tweaks are
  unmeasurable at that size. First experiment: **expand training data** with
  football-data.co.uk free historical CSVs (decades of PL matches → hundreds of
  transitions).
- Hard part: that source keys teams by name strings, this pipeline joins on
  stable `team_id` — a name→id reconciliation table is required.
- Every change is judged on the Phase 1 scoreboard; keep only what beats the
  baselines.

## Conventions

- Single script, organised into small functions; standard data-science stack
  (`requests`, `pandas`, `scikit-learn`) pinned loosely in `requirements.txt`.
- `data/` (cache + generated CSV) and `__pycache__/` are git-ignored.
