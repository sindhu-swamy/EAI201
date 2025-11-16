import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# --- 1. Data Preparation (Reuse fixed and engineered data) ---
# NOTE: This section re-runs the data loading and feature engineering logic 
# to ensure the code block is self-contained and ready to execute.

df = pd.read_csv('fifa_merged_dataset.csv')
df_cleaned = df.drop_duplicates().copy()
df_match_data = df_cleaned[df_cleaned['Key Id'].notnull()].copy()
median_jersey_number = df_match_data['team_jersey_number'].median()
df_match_data['team_position'].fillna('Unknown', inplace=True)
df_match_data['team_jersey_number'].fillna(median_jersey_number, inplace=True)

# Feature Engineering
df_match_data['Goal_Difference'] = df_match_data['Home Team Score'] - df_match_data['Away Team Score']
df_player_info = df_cleaned.dropna(subset=['sofifa_id']).copy()
nationality_age = df_player_info.groupby('nationality')['age'].mean().reset_index()
nationality_age.rename(columns={'nationality': 'Country_Name', 'age': 'Team_Avg_Age'}, inplace=True)

df_match_data = pd.merge(df_match_data, nationality_age, left_on='Home Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Home_Team_Avg_Age'}).drop(columns=['Country Name'])
df_match_data = pd.merge(df_match_data, nationality_age, left_on='Away Team Name', right_on='Country_Name', how='left').rename(columns={'Team_Avg_Age': 'Away_Team_Avg_Age'}).drop(columns=['Country Name'])

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
df_match_data = pd.merge(df_match_data, team_rate_df[['Team_Name', 'Win_Rate']], left_on='Home Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Home_Team_Win_Rate'}).drop(columns=['Team_Name'])
df_match_data = pd.merge(df_match_data, team_rate_df[['Team_Name', 'Win_Rate']], left_on='Away Team Name', right_on='Team_Name', how='left').rename(columns={'Win_Rate': 'Away_Team_Win_Rate'}).drop(columns=['Team_Name'])

global_avg_age = df_match_data[['Home_Team_Avg_Age', 'Away_Team_Avg_Age']].stack().mean()
df_match_data['Home_Team_Avg_Age'].fillna(global_avg_age, inplace=True)
df_match_data['Away_Team_Avg_Age'].fillna(global_avg_age, inplace=True)

# Define Features and Target
features = [
    'Goal_Difference', 'Home_Team_Avg_Age', 'Away_Team_Avg_Age',
    'Home_Team_Win_Rate', 'Away_Team_Win_Rate', 'Group Stage', 'Knockout Stage'
]
X = df_match_data[features]
y = df_match_data['Home Team Win'].astype(int)

# Split data (70% Train, 30% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 2. Build Pipeline for Scaling, Selection, and Tuning ---

# The steps for the pipeline
# 1. Scaling: Standardize continuous features.
# 2. Feature Selection: Use a RandomForest model to determine important features.
# 3. Model: The final Random Forest Classifier for tuning.

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define the parameter grid for GridSearchCV (Hyperparameter Tuning)
param_grid = {
    # Parameters for the final Random Forest Classifier
    'classifier__n_estimators': [50, 100, 200],  # Number of trees
    'classifier__max_depth': [None, 5, 10],      # Max depth of the trees
    'classifier__min_samples_split': [2, 5],     # Minimum samples required to split a node

    # Parameter for the Feature Selection step
    # Select features that have importance > mean importance
    'feature_selection__threshold': ['mean', 'median']
}

# --- 3. Hyperparameter Tuning using k-fold Cross-Validation ---
# We use 5-fold cross-validation (cv=5) on the TRAINING data.
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    verbose=1, 
    n_jobs=-1
)

print("Starting k-fold Cross-Validation and Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)
print("Tuning complete.")

# --- 4. Evaluate the Best Model on the Test Set ---
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test)

print("\n------------------------------------------------------")
print(f"Best Hyperparameters found using GridSearchCV (5-fold): {grid_search.best_params_}")
print("------------------------------------------------------")

# Final Evaluation on the unseen Test Set
test_accuracy = accuracy_score(y_test, y_pred)
test_report = classification_report(y_test, y_pred, zero_division=0)

print(f"\nAccuracy of the Best Tuned Model on the Test Set: {test_accuracy:.4f}")
print("\nClassification Report (Test Set):\n", test_report)