"""Train a season-to-season model and produce next-season predictions."""

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pl_predict.config import FEATURES, TARGET


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
    """
    X, y = train[FEATURES], train["next_points"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    cv_pred = cross_val_predict(model, X, y, cv=LeaveOneOut())
    model.fit(X, y)
    metrics = {
        "rows": len(train),
        "mae": mean_absolute_error(y, cv_pred),
        "r2": r2_score(y, cv_pred),
    }
    return model, metrics


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
