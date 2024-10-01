# fetching data
import requests

# Set up your API URL and headers
url = "https://api.football-data.org/v4/competitions/PL/matches?season=2023"
headers = {"X-Auth-Token": "ae9f6015497f4eb79283f9b20d9cf471"}

# Make the request
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()  # The data will be in JSON format
    # print(data)  # Print it to inspect the structure
else:
    print(f"Error fetching data: {response.status_code}")

# debugging
# # Inspect the structure of the raw data
# import json

# # print the raw data to better understand the structure
# print(json.dumps(data, indent=4))

# cleaning data
import pandas as pd

# Assuming 'data' contains the raw API data
matches = pd.DataFrame(data["matches"])

# Select relevant columns
matches_cleaned = matches[
    ["utcDate", "homeTeam", "awayTeam", "score", "status", "matchday"]
].copy()

# Rename columns for readability
matches_cleaned.rename(
    columns={"utcDate": "date", "matchday": "matchday"}, inplace=True
)

# Extract team names from 'homeTeam' and 'awayTeam'
matches_cleaned["home_team"] = matches_cleaned["homeTeam"].apply(lambda x: x["name"])
matches_cleaned["away_team"] = matches_cleaned["awayTeam"].apply(lambda x: x["name"])

# Extract full-time scores from the 'score' column (nested)
matches_cleaned["home_score"] = matches_cleaned["score"].apply(
    lambda x: x["fullTime"]["home"]
)
matches_cleaned["away_score"] = matches_cleaned["score"].apply(
    lambda x: x["fullTime"]["away"]
)

# Extract the winner (home, away, or draw)
matches_cleaned["winner"] = matches_cleaned["score"].apply(lambda x: x["winner"])

# Drop the original 'score', 'homeTeam', and 'awayTeam' columns
matches_cleaned.drop(columns=["score", "homeTeam", "awayTeam"], inplace=True)

# Convert 'date' column to datetime format
matches_cleaned["date"] = pd.to_datetime(matches_cleaned["date"])

# Handle missing values (if any)
matches_cleaned.fillna(0, inplace=True)

# Preview the cleaned data
print(matches_cleaned.head())
