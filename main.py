"""PL-Predict — predict the Premier League 2026-2027 table from 2025-2026 data.

Pipeline (see the `pl_predict` package for each stage):
  1. fetch    — finished matches per season, cached locally.
  2. features — per-team season stats (points, goals, form, rank).
  3. model    — train on season-N -> season-N+1 transitions, then predict.

Note: the free API tier exposes no lower-division data, so promoted teams are
unknown. Predictions cover the teams continuing from 2025-2026 into 2026-2027.
"""

import os

from pl_predict.config import (
    CACHE_DIR,
    LAST_COMPLETED_SEASON,
    PREDICT_SEASON_LABEL,
    SEASONS,
)
from pl_predict.features import clean_matches, compute_team_stats
from pl_predict.fetch import fetch_season
from pl_predict.model import (
    build_training_data,
    evaluate_predictors,
    predict_table,
    train_model,
)
from pl_predict.report import write_html
from pl_predict.simulate import simulate_probabilities

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    stats_by_season = {
        s: compute_team_stats(clean_matches(fetch_season(s))) for s in SEASONS
    }

    train = build_training_data(stats_by_season)
    model, metrics = train_model(train)

    print("Model trained on season-to-season transitions")
    print(f"  training rows : {metrics['rows']} (teams that stayed up across seasons)")
    print(f"  LOO-CV MAE    : {metrics['mae']:.2f} points")
    print(f"  LOO-CV R^2    : {metrics['r2']:.3f}")

    # The gate: the model must beat the naive baselines on the rank metrics
    # (Spearman, top-4 hit rate) to justify itself. Reported win or lose.
    scoreboard = evaluate_predictors(train)
    print("\nModel vs. baselines (same LOO-CV split; rank metrics are primary):")
    print(scoreboard.round(3).to_string(index=False))

    validation_path = os.path.join(CACHE_DIR, "validation.csv")
    scoreboard.to_csv(validation_path, index=False)
    print(f"Saved scoreboard to {validation_path}")

    table = predict_table(model, stats_by_season[LAST_COMPLETED_SEASON])
    table = simulate_probabilities(table, metrics["cv_residuals"])

    print(f"\nPredicted {PREDICT_SEASON_LABEL} table "
          f"(continuing teams only — promoted clubs unknown):\n")
    out = table[["predicted_rank", "team", "predicted_points",
                 "p_top4", "p_relegation", "label"]]
    out = out.assign(
        predicted_points=out["predicted_points"].round(1),
        p_top4=out["p_top4"].round(2),
        p_relegation=out["p_relegation"].round(2),
    )
    print(out.to_string(index=False))

    print(f"\nPredicted {PREDICT_SEASON_LABEL} champion: {table.loc[0, 'team']} "
          f"(P = {table.loc[0, 'p_champion']:.0%} over 10,000 simulated seasons)")

    out_path = os.path.join(CACHE_DIR, "prediction_2026_2027.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

    html_path = os.path.join(_PROJECT_ROOT, "index.html")
    write_html(table, metrics, scoreboard, html_path)
    print(f"Saved webpage to {html_path}")


if __name__ == "__main__":
    main()
