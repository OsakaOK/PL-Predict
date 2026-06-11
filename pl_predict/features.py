"""Turn raw match JSON into per-team season stats used as model features."""

import pandas as pd


def clean_matches(matches):
    """Flatten the nested match JSON into one tidy row per finished match."""
    rows = []
    for m in matches:
        if m.get("status") != "FINISHED":
            continue
        ft = m["score"]["fullTime"]
        if ft["home"] is None or ft["away"] is None:
            continue
        rows.append(
            {
                "date": m["utcDate"],
                "home_id": m["homeTeam"]["id"],
                "home_team": m["homeTeam"]["name"],
                "away_id": m["awayTeam"]["id"],
                "away_team": m["awayTeam"]["name"],
                "home_score": ft["home"],
                "away_score": ft["away"],
                "winner": m["score"]["winner"],
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _recent_form(matches, team_id, num_matches=5):
    """Points won in a team's last `num_matches` matches of the season."""
    team_matches = matches[
        (matches["home_id"] == team_id) | (matches["away_id"] == team_id)
    ].tail(num_matches)

    points = 0
    for _, match in team_matches.iterrows():
        is_home = match["home_id"] == team_id
        scored = match["home_score"] if is_home else match["away_score"]
        conceded = match["away_score"] if is_home else match["home_score"]
        if scored > conceded:
            points += 3
        elif scored == conceded:
            points += 1
    return points


def compute_team_stats(matches):
    """Build the full-season table (one row per team) with model features."""
    stats = {}
    for _, m in matches.iterrows():
        for team_id, team_name, gf, ga in (
            (m["home_id"], m["home_team"], m["home_score"], m["away_score"]),
            (m["away_id"], m["away_team"], m["away_score"], m["home_score"]),
        ):
            s = stats.setdefault(
                team_id,
                {"team": team_name, "wins": 0, "draws": 0, "losses": 0,
                 "goals_scored": 0, "goals_conceded": 0},
            )
            s["goals_scored"] += gf
            s["goals_conceded"] += ga
            if gf > ga:
                s["wins"] += 1
            elif gf == ga:
                s["draws"] += 1
            else:
                s["losses"] += 1

    df = pd.DataFrame.from_dict(stats, orient="index").rename_axis("team_id").reset_index()
    df["total_games"] = df["wins"] + df["draws"] + df["losses"]
    df["points"] = df["wins"] * 3 + df["draws"]
    df["goal_difference"] = df["goals_scored"] - df["goals_conceded"]
    df["win_percentage"] = df["wins"] / df["total_games"] * 100
    df["recent_form"] = df["team_id"].apply(lambda t: _recent_form(matches, t))

    df = df.sort_values(["points", "goal_difference"], ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
