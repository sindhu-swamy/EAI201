import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load the original dataset
df = pd.read_csv('fifa_merged_dataset.csv')

# --- 1. Load and Clean the Data ---
df_cleaned = df.drop_duplicates().copy()
# Filter to 964 complete match records
df_match_data = df_cleaned[df_cleaned['Key Id'].notnull()].copy()
median_jersey_number = df_match_data['team_jersey_number'].median()
df_match_data['team_position'].fillna('Unknown', inplace=True)
df_match_data['team_jersey_number'].fillna(median_jersey_number, inplace=True)

# --- 2. Engineer Features (FIXED) ---

# A. Goal Differences
df_match_data['Goal_Difference'] = df_match_data['Home Team Score'] - df_match_data['Away Team Score']

# B. Team Average Age (Proxy)
df_player_info = df_cleaned.dropna(subset=['sofifa_id']).copy()
nationality_age = df_player_info.groupby('nationality')['age'].mean().reset_index()
nationality_age.rename(columns={'nationality': 'Country_Name', 'age': 'Team_Avg_Age'}, inplace=True)
df_match_data = pd.merge(df_match_data, nationality_age, left_on='Home Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Home_Team_Avg_Age'}).drop(columns=['Country_Name'])
df_match_data = pd.merge(df_match_data, nationality_age, left_on='Away Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Away_Team_Avg_Age'}).drop(columns=['Country_Name'])

# C. Team Win Rate
all_teams = pd.concat([df_match_data['Home Team Name'], df_match_data['Away Team Name']]).unique()
team_stats = {team: {'Wins': 0, 'Draws': 0, 'Losses': 0, 'Total_Games': 0} for team in all_teams}
for index, row in df_match_data.iterrows():
    home, away = row['Home Team Name'], row['Away Team Name']
    for team, win_col in [(home, 'Home Team Win'), (away, 'Away Team Win')]:
        team_stats[team]['Total_Games'] += 1
        if row[win_col] == 1: team_stats[team]['Wins'] += 1
        elif row['Draw'] == 1: team_stats[team]['Draws'] += 1
        else: team_stats[team]['Losses'] += 1

team_rate_df = pd.DataFrame.from_dict(team_stats, orient='index').reset_index().rename(columns={'index': 'Team_Name'})
team_rate_df['Win_Rate'] = np.where(team_rate_df['Total_Games'] > 0, team_rate_df['Wins'] / team_rate_df['Total_Games'], 0)
team_rate_df = team_rate_df[['Team_Name', 'Win_Rate']]

df_match_data = pd.merge(df_match_data, team_rate_df, left_on='Home Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Home_Team_Win_Rate'}).drop(columns=['Team_Name'])
df_match_data = pd.merge(df_match_data, team_rate_df, left_on='Away Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Away_Team_Win_Rate'}).drop(columns=['Team_Name'])

# --- 3. Final Data Preparation and Split ---

# Impute NaNs in Avg Age with global mean
global_avg_age = df_match_data[['Home_Team_Avg_Age', 'Away_Team_Avg_Age']].stack().mean()
df_match_data['Home_Team_Avg_Age'].fillna(global_avg_age, inplace=True)
df_match_data['Away_Team_Avg_Age'].fillna(global_avg_age, inplace=True)

features = [
    'Goal_Difference', 'Home_Team_Avg_Age', 'Away_Team_Avg_Age', 
    'Home_Team_Win_Rate', 'Away_Team_Win_Rate', 'Group Stage', 'Knockout Stage'
]

X = df_match_data[features]
y = df_match_data['Home Team Win'].astype(int)

# Split data (70% for training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 4. Create and Save Training Output CSV ---

# Combine training features (X_train) and target (y_train)
df_training_data = X_train.copy()
df_training_data['Target_Home_Win'] = y_train

# Include key match identifiers from the full dataset, using the training indices
match_identifiers = df_match_data.loc[X_train.index, ['Match Id', 'Home Team Name', 'Away Team Name']]
final_training_df = match_identifiers.join(df_training_data)

output_file_name = 'fifa_training.csv'
final_training_df.to_csv(output_file_name, index=False)

print(f"Training data CSV file saved: {output_file_name}")
print("\nFirst 5 rows of the training data:")
print(final_training_df.head())