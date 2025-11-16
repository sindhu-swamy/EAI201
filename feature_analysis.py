import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# --- 1. Data Preparation (Final Fixed Pipeline) ---
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

# --- 2. Train Models ---

# A. Optimized Random Forest Pipeline
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42))
])
rf_pipeline.fit(X_train, y_train)

# B. Baseline Logistic Regression Pipeline
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(random_state=42, solver='liblinear'))
])
lr_pipeline.fit(X_train, y_train)

# --- 3. Feature Importance Extraction ---

# Random Forest Feature Importance
rf_selector_mask = rf_pipeline['feature_selection'].get_support()
rf_selected_features = X_train.columns[rf_selector_mask]
rf_importances = rf_pipeline['classifier'].feature_importances_
rf_importance_df = pd.DataFrame({
    'Feature': rf_selected_features, 
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)


# Logistic Regression Coefficients
lr_coefficients = lr_pipeline['logistic'].coef_[0]
lr_importance_df = pd.DataFrame({
    'Feature': X_train.columns, 
    'Coefficient': lr_coefficients,
    'Absolute Coefficient': np.abs(lr_coefficients)
}).sort_values(by='Absolute Coefficient', ascending=False).drop(columns=['Absolute Coefficient'])


print("\n=================================================")
print("  FEATURE IMPORTANCE RANKING (Random Forest)")
print("=================================================")
print(rf_importance_df.to_markdown(index=False))

print("\n=================================================")
print("  FEATURE COEFFICIENT RANKING (Logistic Regression)")
print("=================================================")
print(lr_importance_df.to_markdown(index=False))