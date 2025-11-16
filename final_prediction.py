import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# --- 1. Data Preparation (Final Fixed Pipeline - necessary for model training) ---
df = pd.read_csv('fifa_merged_dataset.csv')
df_cleaned = df.drop_duplicates().copy()
df_match_data = df_cleaned[df_cleaned['Key Id'].notnull()].copy()
median_jersey_number = df_match_data['team_jersey_number'].median()
df_match_data['team_position'] = df_match_data['team_position'].fillna('Unknown')
df_match_data['team_jersey_number'] = df_match_data['team_jersey_number'].fillna(median_jersey_number)

# Feature Engineering
df_match_data['Goal_Difference'] = df_match_data['Home Team Score'] - df_match_data['Away Team Score']
df_player_info = df_cleaned.dropna(subset=['sofifa_id']).copy()
nationality_age = df_player_info.groupby('nationality')['age'].mean().reset_index()
nationality_age.rename(columns={'nationality': 'Country_Name', 'age': 'Team_Avg_Age'}, inplace=True)

# Average Age Merge
df_match_data = pd.merge(df_match_data, nationality_age, left_on='Home Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Home_Team_Avg_Age'}).drop(columns=['Country_Name'])
df_match_data = pd.merge(df_match_data, nationality_age, left_on='Away Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Away_Team_Avg_Age'}).drop(columns=['Country_Name'])

# Team Win Rate
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

# Win Rate Merge
df_match_data = pd.merge(df_match_data, team_rate_df[['Team_Name', 'Win_Rate']], left_on='Home Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Home_Team_Win_Rate'}).drop(columns=['Team_Name'])
df_match_data = pd.merge(df_match_data, team_rate_df[['Team_Name', 'Win_Rate']], left_on='Away Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Away_Team_Win_Rate'}).drop(columns=['Team_Name'])

global_avg_age = df_match_data[['Home_Team_Avg_Age', 'Away_Team_Avg_Age']].stack().mean()
df_match_data['Home_Team_Avg_Age'] = df_match_data['Home_Team_Avg_Age'].fillna(global_avg_age)
df_match_data['Away_Team_Avg_Age'] = df_match_data['Away_Team_Avg_Age'].fillna(global_avg_age)

features = [
    'Goal_Difference', 'Home_Team_Avg_Age', 'Away_Team_Avg_Age',
    'Home_Team_Win_Rate', 'Away_Team_Win_Rate', 'Group Stage', 'Knockout Stage'
]
X = df_match_data[features]
y = df_match_data['Home Team Win'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Train Best Model (Random Forest) ---
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# --- 3. Simulate 2026 World Cup Final Data ---

# Finalists: France (Team A) vs. Brazil (Team B)
# Key Assumption: All historical features are updated post-2022 World Cup.
# Goal Difference is set to 0 for a tight, neutral final.
# Knockout Stage is 1.

# Hypothetical Profiles (based on historical trends for top teams):
# France: Higher age, slightly lower Win Rate.
# Brazil: Younger team, slightly higher historical Win Rate.

X_2026_final_scenarios = pd.DataFrame({
    'Scenario': ['France vs. Brazil (France as "Home")', 'Brazil vs. France (Brazil as "Home")'],
    'Goal_Difference': [0, 0], 
    'Home_Team_Avg_Age': [26.5, 25.0], 
    'Away_Team_Avg_Age': [25.0, 26.5],
    'Home_Team_Win_Rate': [0.75, 0.80], 
    'Away_Team_Win_Rate': [0.80, 0.75],
    'Group Stage': [0, 0], 
    'Knockout Stage': [1, 1]
})

# Isolate features for prediction
X_predict = X_2026_final_scenarios[features]

# --- 4. Prediction ---
# Predict the probability of the designated "Home Team" winning (Class 1)
probabilities = rf_pipeline.predict_proba(X_predict)[:, 1]

X_2026_final_scenarios['P(Home Win)'] = probabilities

# Determine overall winner by taking the maximum probability across both scenarios
max_prob_scenario = X_2026_final_scenarios.loc[X_2026_final_scenarios['P(Home Win)'].idxmax()]

predicted_winner = ""
if max_prob_scenario['Scenario'].startswith("France"):
    predicted_winner = "France"
else:
    predicted_winner = "Brazil"

print("\n=================================================")
print("  2026 WORLD CUP FINAL PREDICTION (SIMULATED)")
print("=================================================")
print("Simulated Scenarios and Predicted Probabilities:")
print(X_2026_final_scenarios[['Scenario', 'P(Home Win)']].to_markdown(index=False))

print("\n-------------------------------------------------")
print(f"Predicted Finalists: France vs. Brazil")
print(f"Predicted Winner (based on feature profile): {predicted_winner}")
print("-------------------------------------------------")