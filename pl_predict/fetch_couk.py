"""Fetch historical PL seasons from football-data.co.uk (free CSVs to 1995-96).

This is the training-data source: ~30 season transitions instead of the 2 the
football-data.org free tier allows. The org API remains the prediction base
(stable integer ids, pretty display names); this source is never joined
against it — training rows only need internally consistent keys.

Format quirks handled here (audited against the real 1995-2025 files):
  - some seasons are latin-1, not UTF-8;
  - some rows carry extra trailing betting columns (ragged CSV);
  - the column layout shifts across eras (a Time column appears in 2019-20),
    so fields are located by header name, never by position;
  - occasional blank lines.
"""

import csv
import io
import os

import pandas as pd

from pl_predict.config import CACHE_DIR, COUK_BASE_URL

_NEEDED = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")


def _season_code(season):
    """1995 -> '9596', 2025 -> '2526' (the URL path segment)."""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def _decode(raw):
    """The files are usually UTF-8 (with BOM) but a few seasons are latin-1."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def parse_couk_csv(text):
    """Parse one season's CSV text into the tidy shape `clean_matches` produces.

    Team ids are the source's team names: audited as internally consistent
    across all seasons 1995-2025 (49 clubs, no spelling drift), so they are
    stable join keys *within* this source. Never join them against
    football-data.org integer ids.
    """
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    idx = {col: header.index(col) for col in _NEEDED}
    max_idx = max(idx.values())

    rows = []
    for r in reader:
        if len(r) <= max_idx:
            continue  # blank or truncated line
        vals = {col: r[i].strip() for col, i in idx.items()}
        if not vals["HomeTeam"] or not vals["FTHG"] or not vals["FTAG"]:
            continue  # unplayed or malformed row
        home_score, away_score = int(float(vals["FTHG"])), int(float(vals["FTAG"]))
        if home_score > away_score:
            winner = "HOME_TEAM"
        elif home_score < away_score:
            winner = "AWAY_TEAM"
        else:
            winner = "DRAW"
        rows.append(
            {
                "date": vals["Date"],
                "home_id": vals["HomeTeam"],
                "home_team": vals["HomeTeam"],
                "away_id": vals["AwayTeam"],
                "away_team": vals["AwayTeam"],
                "home_score": home_score,
                "away_score": away_score,
                "winner": winner,
            }
        )
    df = pd.DataFrame(rows)
    # Dates are dd/mm/yy in older seasons, dd/mm/yyyy in newer ones.
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df.sort_values("date").reset_index(drop=True)


def fetch_couk_matches(season):
    """Return the tidy finished-match frame for a season, using a local cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"couk_E0_{season}.csv")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return parse_couk_csv(_decode(f.read()))

    import requests  # imported lazily so cached runs need no network

    response = requests.get(COUK_BASE_URL.format(code=_season_code(season)))
    response.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(response.content)
    return parse_couk_csv(_decode(response.content))
