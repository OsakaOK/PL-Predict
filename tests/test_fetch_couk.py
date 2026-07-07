"""Unit tests for the football-data.co.uk parser (offline, fixture text)."""

import pandas as pd

from pl_predict.fetch_couk import _season_code, parse_couk_csv

# A fixture reproducing every real-world quirk the parser must survive:
# a Time column (2019+ layout), a ragged row with extra trailing fields,
# a blank line, an unplayed row with empty scores, and dd/mm/yy dates.
FIXTURE = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee
E0,16/08/25,12:30,Arsenal,Everton,2,1,H,M Oliver
E0,09/08/25,15:00,Leeds,Man City,0,3,A,A Taylor,extra,trailing,odds

E0,23/08/25,15:00,Everton,Leeds,1,1,D,P Tierney
E0,30/08/25,15:00,Arsenal,Man City,,,,
"""


def test_parse_skips_blank_and_unplayed_rows():
    df = parse_couk_csv(FIXTURE)
    assert len(df) == 3


def test_parse_survives_ragged_rows_and_sorts_by_date():
    df = parse_couk_csv(FIXTURE)
    assert df["date"].is_monotonic_increasing
    # The ragged Leeds-City row parsed correctly and sorted first.
    assert df.loc[0, "home_team"] == "Leeds"
    assert df.loc[0, "away_score"] == 3
    assert df.loc[0, "winner"] == "AWAY_TEAM"


def test_parse_produces_the_clean_matches_shape():
    df = parse_couk_csv(FIXTURE)
    assert list(df.columns) == [
        "date", "home_id", "home_team", "away_id", "away_team",
        "home_score", "away_score", "winner",
    ]
    # Team ids in this source are the (consistent) team names.
    assert df.loc[0, "home_id"] == "Leeds"
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_parse_locates_columns_by_header_name_not_position():
    # Pre-2019 layout: no Time column — positions shift by one.
    old = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,16/08/97,Arsenal,Everton,2,0,H\n"
    )
    df = parse_couk_csv(old)
    assert df.loc[0, "away_team"] == "Everton"
    assert df.loc[0, "date"] == pd.Timestamp("1997-08-16")


def test_season_code():
    assert _season_code(1995) == "9596"
    assert _season_code(1999) == "9900"
    assert _season_code(2025) == "2526"
