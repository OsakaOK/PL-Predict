"""Unit tests for clean_matches / compute_team_stats on hand-built fixtures."""

from pl_predict.features import clean_matches, compute_team_stats


def _match(date, home_id, home, away_id, away, home_score, away_score,
           status="FINISHED"):
    """Build one raw match dict in the football-data.org v4 shape."""
    if home_score is None or away_score is None or home_score == away_score:
        winner = "DRAW"
    elif home_score > away_score:
        winner = "HOME_TEAM"
    else:
        winner = "AWAY_TEAM"
    return {
        "status": status,
        "utcDate": date,
        "homeTeam": {"id": home_id, "name": home},
        "awayTeam": {"id": away_id, "name": away},
        "score": {"fullTime": {"home": home_score, "away": away_score},
                  "winner": winner},
    }


# A tiny 3-team season: A beats B, B draws C, A beats C.
THREE_TEAM_SEASON = [
    _match("2025-08-01T14:00:00Z", 1, "Team A", 2, "Team B", 2, 0),
    _match("2025-08-08T14:00:00Z", 2, "Team B", 3, "Team C", 1, 1),
    _match("2025-08-15T14:00:00Z", 3, "Team C", 1, "Team A", 0, 3),
]


def test_clean_matches_drops_unfinished_and_null_scores():
    raw = THREE_TEAM_SEASON + [
        _match("2025-08-22T14:00:00Z", 1, "Team A", 2, "Team B", None, None),
        _match("2025-08-29T14:00:00Z", 1, "Team A", 3, "Team C", 1, 0,
               status="SCHEDULED"),
    ]
    df = clean_matches(raw)
    assert len(df) == 3


def test_clean_matches_sorts_by_date():
    df = clean_matches(list(reversed(THREE_TEAM_SEASON)))
    assert df["date"].is_monotonic_increasing


def test_compute_team_stats_points_and_goal_difference():
    stats = compute_team_stats(clean_matches(THREE_TEAM_SEASON))
    by_id = stats.set_index("team_id")

    # A: two wins. B: one draw, one loss. C: one draw, one loss.
    assert by_id.loc[1, "points"] == 6
    assert by_id.loc[2, "points"] == 1
    assert by_id.loc[3, "points"] == 1
    assert by_id.loc[1, "goal_difference"] == 5   # 5 scored, 0 conceded
    assert by_id.loc[2, "goal_difference"] == -2  # 1 scored, 3 conceded
    assert by_id.loc[3, "goal_difference"] == -3  # 1 scored, 4 conceded


def test_compute_team_stats_rank_breaks_points_tie_on_goal_difference():
    stats = compute_team_stats(clean_matches(THREE_TEAM_SEASON))
    by_id = stats.set_index("team_id")
    # B and C both have 1 point; B's better goal difference ranks it above C.
    assert by_id.loc[1, "rank"] == 1
    assert by_id.loc[2, "rank"] == 2
    assert by_id.loc[3, "rank"] == 3


def test_recent_form_counts_only_last_five_matches():
    # A vs B six times: A wins the first, then five 0-0 draws.
    raw = [_match("2025-08-01T14:00:00Z", 1, "Team A", 2, "Team B", 1, 0)] + [
        _match(f"2025-09-{d:02d}T14:00:00Z", 1, "Team A", 2, "Team B", 0, 0)
        for d in range(1, 6)
    ]
    stats = compute_team_stats(clean_matches(raw))
    by_id = stats.set_index("team_id")
    # Season points include the win; recent form (last 5) is draws only.
    assert by_id.loc[1, "points"] == 8
    assert by_id.loc[1, "recent_form"] == 5
    assert by_id.loc[2, "recent_form"] == 5
