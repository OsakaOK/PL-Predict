"""Unit tests for training data, the model, and the baseline scoreboard."""

import numpy as np
import pandas as pd

from pl_predict.model import (
    build_training_data,
    evaluate_predictors,
    predict_table,
    train_model,
)


def _season(points_by_id):
    """A per-team season-stats frame with the model's feature columns filled."""
    ids = list(points_by_id)
    pts = np.array([points_by_id[t] for t in ids], dtype=float)
    return pd.DataFrame(
        {
            "team_id": ids,
            "team": [f"Team {t}" for t in ids],
            "points": pts,
            # Deterministic but not perfectly collinear with points.
            "goal_difference": pts - 52 + np.arange(len(ids)),
            "recent_form": (pts % 16).astype(float),
        }
    )


# Season 2023 -> 2024: team 8 relegated, team 9 promoted; 1..7 continue.
STATS_BY_SEASON = {
    2023: _season({1: 88, 2: 75, 3: 69, 4: 62, 5: 55, 6: 48, 7: 40, 8: 30}),
    2024: _season({1: 84, 2: 60, 3: 71, 4: 66, 5: 42, 6: 52, 7: 38, 9: 45}),
}


def test_build_training_data_inner_joins_continuing_teams():
    train = build_training_data(STATS_BY_SEASON)
    # Teams 8 (relegated) and 9 (promoted) have no transition row.
    assert sorted(train["team_id"]) == [1, 2, 3, 4, 5, 6, 7]
    assert (train["from_season"] == 2023).all()


def test_build_training_data_pairs_next_season_points():
    train = build_training_data(STATS_BY_SEASON).set_index("team_id")
    assert train.loc[1, "points"] == 88
    assert train.loc[1, "next_points"] == 84
    assert train.loc[2, "next_points"] == 60


def test_train_model_returns_loo_metrics_and_residuals():
    train = build_training_data(STATS_BY_SEASON)
    model, metrics = train_model(train)
    assert metrics["rows"] == len(train)
    assert len(metrics["cv_residuals"]) == len(train)
    assert metrics["mae"] > 0
    # The fitted pipeline predicts one value per team.
    assert len(model.predict(train[["points", "goal_difference", "recent_form"]])) \
        == len(train)


def test_evaluate_predictors_scoreboard_shape():
    scoreboard = evaluate_predictors(build_training_data(STATS_BY_SEASON))
    assert list(scoreboard["predictor"]) == [
        "mean", "persistence", "regress_to_mean", "ridge_model"
    ]
    assert list(scoreboard.columns) == [
        "predictor", "spearman", "top4_hit_rate", "champion_hit_rate", "mae", "r2"
    ]


def test_evaluate_predictors_mean_baseline_has_undefined_rank_metrics():
    scoreboard = evaluate_predictors(build_training_data(STATS_BY_SEASON))
    mean_row = scoreboard.set_index("predictor").loc["mean"]
    # A constant predictor has no ranking; naive LOO would report a bogus -1.
    assert np.isnan(mean_row["spearman"])
    assert np.isnan(mean_row["top4_hit_rate"])
    assert np.isnan(mean_row["champion_hit_rate"])


def test_evaluate_predictors_persistence_mae_matches_hand_computation():
    train = build_training_data(STATS_BY_SEASON)
    scoreboard = evaluate_predictors(train).set_index("predictor")
    expected = (train["next_points"] - train["points"]).abs().mean()
    assert scoreboard.loc["persistence", "mae"] == expected


def test_evaluate_predictors_rank_metrics_are_bounded():
    scoreboard = evaluate_predictors(build_training_data(STATS_BY_SEASON))
    ranked = scoreboard[scoreboard["predictor"] != "mean"]
    assert ranked["spearman"].between(-1, 1).all()
    assert ranked["top4_hit_rate"].between(0, 1).all()
    assert ranked["champion_hit_rate"].between(0, 1).all()


def test_predict_table_ranks_and_labels():
    train = build_training_data(STATS_BY_SEASON)
    model, _ = train_model(train)
    table = predict_table(model, STATS_BY_SEASON[2024].drop(columns=["team"])
                          .assign(team=STATS_BY_SEASON[2024]["team"]))
    assert list(table["predicted_rank"]) == list(range(1, len(table) + 1))
    assert table["predicted_points"].is_monotonic_decreasing
    assert table.loc[0, "label"] == "champion"
    assert list(table.loc[1:3, "label"]) == ["top_4"] * 3
    assert list(table["label"].tail(3)) == ["relegation"] * 3
    assert (table["label"].iloc[4:-3] == "mid_table").all()
