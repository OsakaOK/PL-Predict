"""End-to-end pipeline test against the cached real data (no network).

Skipped when data/matches_*.json is absent, so the unit suite stays offline-safe
on a fresh clone.
"""

import os

import pytest

from pl_predict.config import CACHE_DIR, LAST_COMPLETED_SEASON, SEASONS
from pl_predict.features import clean_matches, compute_team_stats
from pl_predict.fetch import fetch_season
from pl_predict.model import (
    build_training_data,
    evaluate_predictors,
    predict_table,
    train_model,
)
from pl_predict.simulate import simulate_probabilities

_CACHED = all(
    os.path.exists(os.path.join(CACHE_DIR, f"matches_{s}.json")) for s in SEASONS
)

pytestmark = pytest.mark.skipif(
    not _CACHED, reason="cached season data not present; run `python3 main.py` once"
)


def test_full_pipeline_on_cached_seasons():
    stats_by_season = {
        s: compute_team_stats(clean_matches(fetch_season(s))) for s in SEASONS
    }
    # A full PL season has 20 teams; max points is 38 wins * 3.
    for stats in stats_by_season.values():
        assert len(stats) == 20
        assert stats["points"].between(0, 114).all()

    train = build_training_data(stats_by_season)
    assert len(train) == 17 * (len(SEASONS) - 1)  # 17 continuing teams/transition

    model, metrics = train_model(train)
    scoreboard = evaluate_predictors(train)
    assert len(scoreboard) == 4

    table = predict_table(model, stats_by_season[LAST_COMPLETED_SEASON])
    table = simulate_probabilities(table, metrics["cv_residuals"], n_sims=2000)
    assert table["p_champion"].sum() == pytest.approx(1.0)
    assert table["p_top4"].sum() == pytest.approx(4.0)
    assert table["p_relegation"].sum() == pytest.approx(3.0)
    assert (table["label"] == "champion").sum() == 1
