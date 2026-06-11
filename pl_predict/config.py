"""Shared configuration for the prediction pipeline."""

import os

# Prefer an environment variable; fall back to the original committed key.
# (That key is public in git history and should be rotated.)
API_TOKEN = os.environ.get("FOOTBALL_DATA_TOKEN", "ae9f6015497f4eb79283f9b20d9cf471")
BASE_URL = "https://api.football-data.org/v4/competitions/PL/matches"

# Cache + output directory at the project root (one level above this package).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_PROJECT_ROOT, "data")

# Season start years available on the free tier. The PL "2025" season is the
# 2025-2026 campaign; we predict the following ("2026") season.
SEASONS = [2023, 2024, 2025]
LAST_COMPLETED_SEASON = 2025          # the 2025-2026 season (our prediction base)
PREDICT_SEASON_LABEL = "2026-2027"

# Features fed to the model (a season's stats) and the target (next season points).
# Kept deliberately small and non-redundant: with only a few dozen training rows,
# adding collinear features (win_percentage, goals_scored) destabilises the fit.
FEATURES = ["points", "goal_difference", "recent_form"]
TARGET = "points"
