"""Train a season-to-season model and produce next-season predictions."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pl_predict.config import FEATURES, TARGET


def _make_model():
    """The production model: standardised features + L2-regularised regression."""
    return make_pipeline(StandardScaler(), Ridge(alpha=5.0))


def build_training_data(stats_by_season):
    """Pair each team's season-N features with its season-N+1 points."""
    seasons = sorted(stats_by_season)
    frames = []
    for n in seasons:
        if (n + 1) not in stats_by_season:
            continue
        cur = stats_by_season[n]
        nxt = stats_by_season[n + 1][["team_id", TARGET]].rename(
            columns={TARGET: "next_points"}
        )
        merged = cur.merge(nxt, on="team_id", how="inner")  # only teams that stayed up
        merged["from_season"] = n
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def train_model(train):
    """Fit StandardScaler + Ridge and return (model, leave-one-out CV metrics).

    Regularisation keeps the fit stable on this small, correlated dataset.
    Metrics use leave-one-out CV (honest out-of-sample error), not in-sample fit.
    `metrics["cv_residuals"]` (actual - predicted under LOO-CV) feeds the Monte
    Carlo simulation: real season-over-season surprises, no Gaussian assumption.
    """
    X, y = train[FEATURES], train["next_points"]
    model = _make_model()
    cv_pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    model.fit(X, y)
    metrics = {
        "rows": len(train),
        "mae": mean_absolute_error(y, cv_pred),
        "r2": r2_score(y, cv_pred),
        "cv_residuals": (y - cv_pred).to_numpy(),
    }
    return model, metrics


def _mean_spearman(train, pred):
    """Spearman rank correlation per from_season, averaged across seasons."""
    rhos = []
    for _, grp in train.groupby("from_season"):
        rho, _ = spearmanr(pred[grp.index], grp["next_points"])
        rhos.append(rho)
    return float(np.mean(rhos))


def _mean_top4_hit_rate(train, pred):
    """Of the 4 teams predicted highest, how many actually finished top 4?

    Computed within each from_season's continuing teams, averaged across seasons.
    """
    hits = []
    for _, grp in train.groupby("from_season"):
        predicted_top4 = set(grp.loc[pd.Series(pred[grp.index], index=grp.index)
                                     .nlargest(4).index, "team_id"])
        actual_top4 = set(grp.nlargest(4, "next_points")["team_id"])
        hits.append(len(predicted_top4 & actual_top4) / 4)
    return float(np.mean(hits))


def evaluate_predictors(train):
    """Score the model against naive baselines under the same LOO-CV split.

    Baselines:
      mean            — predict the training-mean points for everyone (R²=0 floor).
      persistence     — next points = this season's points (the bar to beat).
      regress_to_mean — linear fit on `points` alone; if Ridge only ties this,
                        the extra features add nothing beyond shrinkage.

    Primary criteria are rank-based (Spearman, top-4 hit rate) because the
    product is a table, not a points estimate. The model must win those to
    justify itself over persistence.
    """
    X, y = train[FEATURES], train["next_points"]
    loo = LeaveOneOut()
    predictions = {
        "mean": cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=loo),
        "persistence": train["points"].to_numpy(dtype=float),
        "regress_to_mean": cross_val_predict(
            LinearRegression(), train[["points"]], y, cv=loo
        ),
        "ridge_model": cross_val_predict(_make_model(), X, y, cv=loo),
    }
    rows = []
    for name, pred in predictions.items():
        # The mean baseline predicts a constant per season, so rank metrics are
        # undefined (under LOO the tiny leave-out shifts even anti-rank it).
        constant = name == "mean"
        rows.append(
            {
                "predictor": name,
                "spearman": np.nan if constant else _mean_spearman(train, pred),
                "top4_hit_rate": np.nan if constant else _mean_top4_hit_rate(train, pred),
                "mae": mean_absolute_error(y, pred),
                "r2": r2_score(y, pred),
            }
        )
    return pd.DataFrame(rows)


def predict_table(model, base_stats):
    """Predict next-season points, then rank and label the continuing teams."""
    table = base_stats.copy()
    table["predicted_points"] = model.predict(table[FEATURES])
    table = table.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    table["predicted_rank"] = table.index + 1

    table["label"] = "mid_table"
    table.loc[0, "label"] = "champion"
    table.loc[1:3, "label"] = "top_4"
    table.loc[table.index[-3:], "label"] = "relegation"
    return table
