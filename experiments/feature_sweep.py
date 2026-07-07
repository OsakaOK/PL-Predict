"""Phase 3 experiment harness: candidate features / alphas, judged on the gate.

Run from anywhere: `python3 experiments/feature_sweep.py` (uses the cached
season data; run `python3 main.py` once first on a fresh clone).

Everything is evaluated exactly like `pl_predict.model.evaluate_predictors`:
LOO-CV predictions, Spearman per from_season (averaged), top-4 hit rate,
champion hit rate (diagnostic), MAE, R2. A candidate earns adoption only if it
beats the CURRENT model (not just persistence) on the primary rank metrics
AND wins consistently in the paired per-season comparison — see the
"Judging future changes" rule in CLAUDE.md.

Verdict as of 2026-07 (30 transitions): no candidate adopted. `base+momentum`
was closest (18-10 seasons, sign-test p=0.185 — not distinguishable from
luck). Retest when new transitions accumulate (one per July).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pl_predict.config import COUK_SEASONS, FEATURES
from pl_predict.features import compute_team_stats
from pl_predict.fetch_couk import fetch_couk_matches
from pl_predict.model import (
    _champion_hit_rate,
    _mean_spearman,
    _mean_top4_hit_rate,
    build_training_data,
)


def _team_points_series(matches, team_id):
    """Chronological per-match points for one team."""
    tm = matches[(matches["home_id"] == team_id) | (matches["away_id"] == team_id)]
    pts = []
    for _, m in tm.iterrows():
        is_home = m["home_id"] == team_id
        gf = m["home_score"] if is_home else m["away_score"]
        ga = m["away_score"] if is_home else m["home_score"]
        pts.append(3 if gf > ga else 1 if gf == ga else 0)
    return np.array(pts)


def extended_stats(matches):
    """compute_team_stats plus the candidate features under test."""
    base = compute_team_stats(matches)
    extras = []
    for team_id in base["team_id"]:
        pts = _team_points_series(matches, team_id)
        half = len(pts) // 2
        home = matches[matches["home_id"] == team_id]
        away = matches[matches["away_id"] == team_id]
        home_pts = sum(3 if h > a else 1 if h == a else 0
                       for h, a in zip(home["home_score"], home["away_score"]))
        away_pts = sum(3 if a > h else 1 if a == h else 0
                       for h, a in zip(away["home_score"], away["away_score"]))
        extras.append({
            "team_id": team_id,
            "momentum": pts[half:].sum() - pts[:half].sum(),
            "recent_form10": pts[-10:].sum(),
            "home_points": home_pts,
            "away_points": away_pts,
        })
    return base.merge(pd.DataFrame(extras), on="team_id")


CANDIDATE_SETS = [
    (FEATURES, "CURRENT (base)"),
    (["points"], "points only"),
    (["goal_difference"], "gd only"),
    (["points", "goal_difference"], "base - recent_form"),
    (FEATURES + ["momentum"], "base + momentum"),
    (FEATURES + ["recent_form10"], "base + form10"),
    (["points", "goal_difference", "recent_form10"], "form10 instead of form5"),
    (FEATURES + ["goals_scored", "goals_conceded"], "base + goals split"),
    (FEATURES + ["home_points", "away_points"], "base + home/away"),
    (FEATURES + ["win_percentage"], "base + win% (collinear)"),
    (FEATURES + ["momentum", "recent_form10", "goals_scored", "goals_conceded",
                 "home_points", "away_points", "win_percentage"], "kitchen sink"),
]

ALPHAS = (0.1, 1, 5, 10, 25, 50, 100)


def main():
    print(f"building extended stats for {len(COUK_SEASONS)} seasons...")
    stats = {s: extended_stats(fetch_couk_matches(s)) for s in COUK_SEASONS}
    train = build_training_data(stats)
    y = train["next_points"]
    print(f"training rows: {len(train)}")

    def loo_pred(features, alpha=5.0):
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        return cross_val_predict(model, train[features], y, cv=LeaveOneOut())

    def score(pred, name, alpha=5.0):
        return {
            "candidate": name,
            "alpha": alpha,
            "spearman": _mean_spearman(train, pred),
            "top4": _mean_top4_hit_rate(train, pred),
            "champ": _champion_hit_rate(train, pred),
            "mae": mean_absolute_error(y, pred),
            "r2": r2_score(y, pred),
        }

    preds = {name: loo_pred(feats) for feats, name in CANDIDATE_SETS}

    print("\n=== Feature sets (Ridge alpha=5, LOO-CV) ===")
    sweep = pd.DataFrame([score(p, n) for n, p in preds.items()])
    print(sweep.drop(columns="alpha").round(4).to_string(index=False))

    print("\n=== Paired per-season Spearman vs CURRENT (sign test) ===")
    base_pred = preds["CURRENT (base)"]
    for name, pred in preds.items():
        if name == "CURRENT (base)":
            continue
        diffs = []
        for _, grp in train.groupby("from_season"):
            rho_c = spearmanr(pred[grp.index], grp["next_points"])[0]
            rho_b = spearmanr(base_pred[grp.index], grp["next_points"])[0]
            diffs.append(rho_c - rho_b)
        diffs = np.array(diffs)
        wins, losses = int((diffs > 0).sum()), int((diffs < 0).sum())
        p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
        print(f"  {name:24s} won {wins:2d} lost {losses:2d} "
              f"mean d-rho {diffs.mean():+.4f}  p={p:.3f}")

    print("\n=== Alpha sweep on the current feature set ===")
    adf = pd.DataFrame(
        [score(loo_pred(FEATURES, alpha=a), "base", alpha=a) for a in ALPHAS]
    )
    print(adf.drop(columns="candidate").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
