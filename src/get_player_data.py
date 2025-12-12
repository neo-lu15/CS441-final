# import pandas as pd
# import time
# from nba_api.stats.endpoints import leaguedashplayerstats

# # 1. Define the seasons you want (2016-17 to 2024-25)
# # NBA API formats seasons as "YYYY-YY", e.g., "2016-17"
# seasons = []
# for year in range(2016, 2024):
#     next_year = str(year + 1)[-2:] # Get last two digits of next year
#     seasons.append(f"{year}-{next_year}")

# print(f"Fetching data for seasons: {seasons}")

# all_season_data = []

# # 2. Loop through each season and fetch data
# for season in seasons:
#     print(f"Fetching stats for {season}...")
    
#     # leaguedashplayerstats gets rows for EVERY player in that season
#     # MeasureType='Base' gives you PTS, REB, AST, etc.
#     # You can change MeasureType to 'Advanced' for PER, USG%, etc.
#     api_call = leaguedashplayerstats.LeagueDashPlayerStats(
#         season=season,
#         measure_type_detailed_defense='Base' 
#     )
    
#     # Get the dataframe
#     df = api_call.get_data_frames()[0]
    
#     # Add a column to identify which season this data belongs to
#     df['SEASON'] = season
    
#     # Append to our list
#     all_season_data.append(df)
    
#     # Sleep to avoid getting blocked by the API (Rate Limiting)
#     time.sleep(2)

# # 3. Combine all seasons into one DataFrame
# final_df = pd.concat(all_season_data, ignore_index=True)

# # 4. Save to CSV for your ML project
# final_df.to_csv('nba_player_stats_2016_2024.csv', index=False)
# print("Success! Data saved to 'nba_player_stats_2016_2024.csv'")
# import pandas as pd
# import time
# from nba_api.stats.endpoints import leaguegamelog

# # 1. Define the seasons (2016-17 to 2024-25)
# seasons = []
# for year in range(2016, 2024):
#     next_year = str(year + 1)[-2:]
#     seasons.append(f"{year}-{next_year}")

# print(f"Targeting Seasons: {seasons}")

# all_games = []

# # 2. Loop through seasons only (Efficient: ~1 call per season)
# for season in seasons:
#     print(f"Fetching full game logs for {season}...")
    
#     try:
#         # PlayerOrTeam='P' gets Player stats instead of Team stats
#         # direction='DESC' sorts by latest games first
#         log = leaguegamelog.LeagueGameLog(
#             season=season,
#             player_or_team_abbreviation='P', 
#             season_type_all_star='Regular Season' 
#         )
        
#         df = log.get_data_frames()[0]
        
#         # Add a clear season column (API format can be weird)
#         df['SEASON_LABEL'] = season
        
#         all_games.append(df)
#         print(f"  - Found {len(df)} game rows.")
        
#         # Short sleep to be polite to the server
#         time.sleep(2)
        
#     except Exception as e:
#         print(f"Error fetching {season}: {e}")

# # 3. Combine and Save
# if all_games:
#     final_df = pd.concat(all_games, ignore_index=True)
    
#     # Save to a large CSV
#     final_df.to_csv('nba_all_games_2016_2024.csv', index=False)
#     print(f"\nDONE! Saved {len(final_df)} rows to 'nba_all_games_2016_2024.csv'")
# else:
#     print("No data found.")
import pandas as pd
import numpy as np

# --- 1. SETUP COORDINATES & LOGIC ---

# Full dictionary of NBA Team Locations (Lat, Lon)
team_coords = {
    'ATL': (33.7573, -84.3963), 'BOS': (42.3662, -71.0621), 'BKN': (40.6826, -73.9754),
    'CHA': (35.2251, -80.8392), 'CHI': (41.8807, -87.6742), 'CLE': (41.4965, -81.6881),
    'DAL': (32.7905, -96.8103), 'DEN': (39.7487, -105.0076), 'DET': (42.3411, -83.0553),
    'GSW': (37.7680, -122.3877), 'HOU': (29.7508, -95.3621), 'IND': (39.7640, -86.1555),
    'LAC': (34.0430, -118.2673), 'LAL': (34.0430, -118.2673), 'MEM': (35.1381, -90.0506),
    'MIA': (25.7814, -80.1870), 'MIL': (43.0451, -87.9174), 'MIN': (44.9795, -93.2761),
    'NOP': (29.9490, -90.0821), 'NYK': (40.7505, -73.9934), 'OKC': (35.4634, -97.5151),
    'ORL': (28.5392, -81.3839), 'PHI': (39.9012, -75.1720), 'PHX': (33.4457, -112.0712),
    'POR': (45.5316, -122.6668), 'SAC': (38.5802, -121.4997), 'SAS': (29.4270, -98.4375),
    'TOR': (43.6435, -79.3791), 'UTA': (40.7683, -111.9011), 'WAS': (38.8982, -77.0209)
}

# Bubble Parameters
BUBBLE_START = pd.Timestamp('2020-07-01')
BUBBLE_END   = pd.Timestamp('2020-10-15')
ORLANDO_COORDS = (28.5392, -81.3839)

def haversine(lat1, lon1, lat2, lon2):
    """Calculates distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def get_game_coords(row):
    """Determines the lat/lon of where the game was played."""
    # 1. Bubble Logic
    if BUBBLE_START <= row['GAME_DATE'] <= BUBBLE_END:
        return ORLANDO_COORDS

    # 2. Standard Logic
    matchup = row['MATCHUP']
    team_abbr = row['TEAM_ABBREVIATION']
    
    # If " @ ", the opponent is the home team
    if ' @ ' in matchup:
        home_team = matchup.split(' @ ')[1]
    else:
        # If " vs. ", the current player's team is home
        home_team = team_abbr
        
    return team_coords.get(home_team, (np.nan, np.nan))

# --- 2. EXECUTION ---

print("Loading data...")
# Make sure this matches your current filename
df = pd.read_csv('nba_all_games_2016_2024.csv') 

# Convert date column to datetime objects
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

print("Mapping stadium locations...")
# Apply the coordinate logic row by row
coords = df.apply(get_game_coords, axis=1)

# Create temporary Lat/Lon columns for the current game
df['LAT'] = [x[0] for x in coords]
df['LON'] = [x[1] for x in coords]

print("Calculating distances...")
# Sort by Player and Date to ensure chronological order
df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

# Shift columns to get the PREVIOUS game's location
df['PREV_LAT'] = df.groupby('PLAYER_ID')['LAT'].shift(1)
df['PREV_LON'] = df.groupby('PLAYER_ID')['LON'].shift(1)

# Calculate the distance
df['DISTANCE_TRAVELED_KM'] = haversine(
    df['PREV_LAT'], df['PREV_LON'],
    df['LAT'], df['LON']
)

# Calculate Days Rest (Difference in days between games)
df['DAYS_REST'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days

# --- 3. CLEANING ---
# If a player rested > 10 days (season break, injury, or covid hiatus), 
# set travel distance to 0.
df.loc[df['DAYS_REST'] > 10, 'DISTANCE_TRAVELED_KM'] = 0

# Fill remaining NaNs (first game of career/season) with 0
df['DISTANCE_TRAVELED_KM'] = df['DISTANCE_TRAVELED_KM'].fillna(0)
df['DAYS_REST'] = df['DAYS_REST'].fillna(100) # Assume long rest before first game

# Drop the temporary calculation columns to keep the CSV clean
df = df.drop(columns=['LAT', 'LON', 'PREV_LAT', 'PREV_LON'])

# --- 4. SAVE ---
output_filename = 'nba_games_with_travel.csv'
df.to_csv(output_filename, index=False)

print(f"Success! New file created: {output_filename}")
print(df[['GAME_DATE', 'MATCHUP', 'DISTANCE_TRAVELED_KM', 'DAYS_REST']].head())