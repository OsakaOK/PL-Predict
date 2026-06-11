# PL-Predict

Predict the **English Premier League 2026-2027** table — Champion, Top 4, and
Relegation spots — from **2025-2026** season data, using machine learning.

## How it works

`main.py` runs a small end-to-end pipeline:

1. **Fetch** finished matches for the available seasons (2023-24, 2024-25,
   2025-26) from the [football-data.org](https://www.football-data.org/) API,
   cached locally in `data/`.
2. **Aggregate** each season into a per-team table (points, goals, goal
   difference, recent form, rank).
3. **Train** a regularised linear model (`StandardScaler` + `Ridge`) on
   season-to-season transitions: a team's stats in season *N* → its points in
   season *N+1*.
4. **Predict** 2026-2027 points by applying the model to 2025-2026 stats, then
   rank teams and label Champion / Top 4 / Relegation.

Model quality is reported with leave-one-out cross-validation (out-of-sample),
not the optimistic in-sample fit.

### Limitation

The free API tier has no lower-division data, so promoted clubs are unknown.
Predictions therefore cover the teams continuing from 2025-26 into 2026-27. This
is fine for predicting the champion (always a continuing top side).

## Usage

```bash
pip install -r requirements.txt

# Optional: use your own API key (the committed one is public and rate-limited)
export FOOTBALL_DATA_TOKEN=your_token_here

python main.py
```

The first run fetches and caches data in `data/`; later runs reuse the cache.
Predictions are printed and written to `data/prediction_2026_2027.csv`.
