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
  `train_model`, `predict_table`.

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
5. **`train_model` + `predict_table`** — Fits `StandardScaler` + `Ridge` on the
   transitions, reports leave-one-out CV metrics, then applies the model to 2025-26
   stats to predict 2026-27 points. Teams are ranked and labelled champion / top_4 /
   relegation; `main` prints the table and saves `data/prediction_2026_2027.csv`.

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
  (`cross_val_predict`), not in-sample fit. Expect a modest R² (~0.25, MAE ~10
  points) — predicting football a season ahead from one season of stats is
  inherently noisy. Don't "improve" this by reporting in-sample numbers.
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

## Gotchas

- **API token.** A key is committed inline at [main.py](main.py) as a fallback. It
  is public in git history and should be rotated; prefer the `FOOTBALL_DATA_TOKEN`
  env var.
- **Rate limit.** Free tier is ~10 requests/min; `fetch_season` sleeps 6s after a
  live fetch. The cache means this rarely bites.
- **Adding seasons.** When football-data grants more historical seasons, add them to
  `SEASONS` — more transitions means more training data and a better model.

## Conventions

- Single script, organised into small functions; standard data-science stack
  (`requests`, `pandas`, `scikit-learn`) pinned loosely in `requirements.txt`.
- `data/` (cache + generated CSV) and `__pycache__/` are git-ignored.
