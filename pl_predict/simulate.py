"""Monte Carlo season simulation from the model's empirical residuals.

A single predicted-points number overstates precision: the model's own
out-of-sample error is ~10 points, and weaker teams sometimes overperform.
Instead of assuming Gaussian errors, each simulated season resamples the
model's actual LOO-CV residuals — real season-over-season surprises, upset
asymmetry included — onto the predicted points, then ranks the table.
Repeating this many times turns hard labels into probabilities.

With ~34 residuals the empirical distribution is coarse; the probabilities
are rough guides, not precise odds.
"""

import numpy as np

N_SIMS = 10_000


def simulate_probabilities(table, residuals, n_sims=N_SIMS, seed=42):
    """Add P(champion) / P(top-4) / P(relegation) columns to the predicted table.

    `table` must be sorted by predicted_points (as `predict_table` returns it);
    `residuals` are the model's LOO-CV residuals (actual - predicted).
    Relegation means the bottom 3 of the continuing teams, matching the labels.
    """
    rng = np.random.default_rng(seed)
    base = table["predicted_points"].to_numpy()
    n_teams = len(base)

    # (n_sims, n_teams): every team draws its own surprise in every season.
    sims = base + rng.choice(residuals, size=(n_sims, n_teams), replace=True)
    order = np.argsort(-sims, axis=1)  # column j holds the team at rank j+1

    champion = np.bincount(order[:, 0], minlength=n_teams)
    top4 = np.bincount(order[:, :4].ravel(), minlength=n_teams)
    relegation = np.bincount(order[:, -3:].ravel(), minlength=n_teams)

    table = table.copy()
    table["p_champion"] = champion / n_sims
    table["p_top4"] = top4 / n_sims
    table["p_relegation"] = relegation / n_sims
    return table
