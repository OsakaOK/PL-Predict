"""End-to-end pipeline test against the cached real data (no network).

Skipped when the cached files are absent, so the unit suite stays offline-safe
on a fresh clone.
"""

import os

import pytest

from pl_predict.config import CACHE_DIR, COUK_SEASONS, LAST_COMPLETED_SEASON
from pl_predict.features import clean_matches, compute_team_stats
from pl_predict.fetch import fetch_season
from pl_predict.fetch_couk import fetch_couk_matches
from pl_predict.model import (
    build_training_data,
    evaluate_predictors,
    predict_table,
    train_model,
)
from pl_predict.simulate import simulate_probabilities

_CACHED = all(
    os.path.exists(os.path.join(CACHE_DIR, f"couk_E0_{s}.csv")) for s in COUK_SEASONS
) and os.path.exists(
    os.path.join(CACHE_DIR, f"matches_{LAST_COMPLETED_SEASON}.json")
)

pytestmark = pytest.mark.skipif(
    not _CACHED, reason="cached season data not present; run `python3 main.py` once"
)


def test_full_pipeline_on_cached_seasons():
    train_stats = {
        s: compute_team_stats(fetch_couk_matches(s)) for s in COUK_SEASONS
    }
    # Every PL season since 1995-96: 20 teams, 38 games, max 114 points.
    for stats in train_stats.values():
        assert len(stats) == 20
        assert stats["points"].between(0, 114).all()

    train = build_training_data(train_stats)
    assert len(train) == 17 * (len(COUK_SEASONS) - 1)  # 17 continuing/transition

    model, metrics = train_model(train)
    scoreboard = evaluate_predictors(train)
    assert len(scoreboard) == 4

    base_stats = compute_team_stats(clean_matches(fetch_season(LAST_COMPLETED_SEASON)))
    table = predict_table(model, base_stats)
    table = simulate_probabilities(table, metrics["cv_residuals"], n_sims=2000)
    assert table["p_champion"].sum() == pytest.approx(1.0)
    assert table["p_top4"].sum() == pytest.approx(4.0)
    assert table["p_relegation"].sum() == pytest.approx(3.0)
    assert (table["label"] == "champion").sum() == 1


def test_sources_agree_on_the_shared_season():
    """The org API and co.uk describe the same 2025-26 matches.

    Feature values must be identical across sources (names differ, so compare
    the sorted per-team stat tuples) — otherwise training and prediction
    features live on different scales and the model is silently miscalibrated.
    """
    org = compute_team_stats(clean_matches(fetch_season(LAST_COMPLETED_SEASON)))
    couk = compute_team_stats(fetch_couk_matches(LAST_COMPLETED_SEASON))

    cols = ["points", "goal_difference", "recent_form"]
    org_stats = sorted(map(tuple, org[cols].to_numpy()))
    couk_stats = sorted(map(tuple, couk[cols].to_numpy()))
    assert org_stats == couk_stats
