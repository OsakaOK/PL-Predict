"""Fetch Premier League matches from football-data.org, with a local cache."""

import json
import os
import time

from pl_predict.config import API_TOKEN, BASE_URL, CACHE_DIR


def fetch_season(season):
    """Return the raw match list for a season, using a local cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"matches_{season}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import requests  # imported lazily so cached runs need no network

    response = requests.get(
        BASE_URL, headers={"X-Auth-Token": API_TOKEN}, params={"season": season}
    )
    response.raise_for_status()
    matches = response.json()["matches"]
    with open(cache_path, "w") as f:
        json.dump(matches, f)
    time.sleep(6)  # free tier allows ~10 requests/minute
    return matches
