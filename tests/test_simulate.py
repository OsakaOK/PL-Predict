"""Unit tests for the Monte Carlo probability layer."""

import numpy as np
import pandas as pd
import pytest

from pl_predict.simulate import simulate_probabilities


@pytest.fixture
def table():
    """A 10-team predicted table, sorted by points as predict_table returns it."""
    points = np.array([82, 74, 68, 63, 58, 54, 50, 45, 40, 33], dtype=float)
    return pd.DataFrame(
        {
            "team": [f"Team {i}" for i in range(10)],
            "predicted_points": points,
            "predicted_rank": np.arange(1, 11),
        }
    )


RESIDUALS = np.array([-9.0, -5.0, -2.0, 0.0, 1.0, 3.0, 6.0, 10.0])


def test_probabilities_sum_to_slot_counts(table):
    result = simulate_probabilities(table, RESIDUALS, n_sims=4000)
    # Every simulated season has exactly 1 champion, 4 top-4, 3 relegated.
    assert result["p_champion"].sum() == pytest.approx(1.0)
    assert result["p_top4"].sum() == pytest.approx(4.0)
    assert result["p_relegation"].sum() == pytest.approx(3.0)


def test_probabilities_are_valid_and_ordered(table):
    result = simulate_probabilities(table, RESIDUALS, n_sims=4000)
    for col in ("p_champion", "p_top4", "p_relegation"):
        assert result[col].between(0, 1).all()
    # The strongest team should be likelier champion than the weakest, and
    # vice versa for relegation — the point spread dwarfs the residual spread.
    assert result.loc[0, "p_champion"] > result.loc[9, "p_champion"]
    assert result.loc[9, "p_relegation"] > result.loc[0, "p_relegation"]


def test_simulation_is_deterministic_for_a_seed(table):
    a = simulate_probabilities(table, RESIDUALS, n_sims=1000, seed=7)
    b = simulate_probabilities(table, RESIDUALS, n_sims=1000, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_input_table_is_not_mutated(table):
    before = table.copy()
    simulate_probabilities(table, RESIDUALS, n_sims=100)
    pd.testing.assert_frame_equal(table, before)


def test_zero_residuals_reproduce_the_point_estimate_labels(table):
    # With no uncertainty, the simulation must collapse to the hard table.
    result = simulate_probabilities(table, np.zeros(5), n_sims=50)
    assert list(result["p_champion"]) == [1.0] + [0.0] * 9
    assert list(result["p_top4"]) == [1.0] * 4 + [0.0] * 6
    assert list(result["p_relegation"]) == [0.0] * 7 + [1.0] * 3
