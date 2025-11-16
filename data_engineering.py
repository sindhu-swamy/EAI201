import pandas as pd
import numpy as np

# Load the original dataset
df = pd.read_csv('fifa_merged_dataset.csv')

# --- 1. Clean the Data (Filter to match records for meaningful statistics) ---
df_cleaned = df.drop_duplicates().copy()
df_match_data = df_cleaned[df_cleaned['Key Id'].notnull()].copy()
median_jersey_number = df_match_data['team_jersey_number'].median()
df_match_data['team_position'].fillna('Unknown', inplace=True)
df_match_data['team_jersey_number'].fillna(median_jersey_number, inplace=True)

# --- 2. Feature Engineering ---

# A. Goal Differences
df_match_data['Goal_Difference'] = df_match_data['Home Team Score'] - df_match_data['Away Team Score']

# B. Team Average Age (Proxy for Player Experience)
# Note: Team Average Age is calculated as the mean age of ALL players in the dataset
# with a matching nationality, due to the random cross-join of player and match data.
df_player_info = df_cleaned.dropna(subset=['sofifa_id']).copy()
nationality_age = df_player_info.groupby('nationality')['age'].mean().reset_index()
nationality_age.rename(columns={'nationality': 'Country_Name', 'age': 'Team_Avg_Age'}, inplace=True)

df_match_data = pd.merge(
    df_match_data,
    nationality_age,
    left_on='Home Team Name',
    right_on='Country_Name',
    how='left'
).rename(columns={'Team_Avg_Age': 'Home_Team_Avg_Age'}).drop(columns=['Country_Name'])

df_match_data = pd.merge(
    df_match_data,
    nationality_age,
    left_on='Away Team Name',
    right_on='Country_Name',
    how='left'
).rename(columns={'Team_Avg_Age': 'Away_Team_Avg_Age'}).drop(columns=['Country_Name'])


# C. Team Win Rate
# Calculate Win Rate (Wins / Total Games) for each team across all 964 matches.
all_teams = pd.concat([df_match_data['Home Team Name'], df_match_data['Away Team Name']]).unique()
team_stats = {team: {'Wins': 0, 'Draws': 0, 'Losses': 0, 'Total_Games': 0} for team in all_teams}

for index, row in df_match_data.iterrows():
    home = row['Home Team Name']
    away = row['Away Team Name']
    
    # Update stats for both teams
    for team, win_col in [(home, 'Home Team Win'), (away, 'Away Team Win')]:
        team_stats[team]['Total_Games'] += 1
        if row[win_col] == 1:
            team_stats[team]['Wins'] += 1
        elif row['Draw'] == 1:
            team_stats[team]['Draws'] += 1
        else:
            team_stats[team]['Losses'] += 1

team_rate_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index().rename(columns={'index': 'Team_Name'})
team_rate_df['Win_Rate'] = np.where(team_rate_df['Total_Games'] > 0, 
                                   team_rate_df['Wins'] / team_rate_df['Total_Games'], 0)
team_rate_df = team_rate_df[['Team_Name', 'Win_Rate']]

# Merge Win Rate back to the main DataFrame
df_match_data = pd.merge(
    df_match_data,
    team_rate_df,
    left_on='Home Team Name',
    right_on='Team_Name',
    how='left'
).rename(columns={'Win_Rate': 'Home_Team_Win_Rate'}).drop(columns=['Team_Name'])

df_match_data = pd.merge(
    df_match_data,
    team_rate_df,
    left_on='Away Team Name',
    right_on='Team_Name',
    how='left'
).rename(columns={'Win_Rate': 'Away_Team_Win_Rate'}).drop(columns=['Team_Name'])

# Save the full engineered DataFrame
output_file_name = 'fifa_engineered_features.csv'
df_match_data.to_csv(output_file_name, index=False)