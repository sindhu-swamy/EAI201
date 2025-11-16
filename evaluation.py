import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

# --- 1. Data Preparation (Final Fixed Pipeline) ---
# NOTE: This ensures the subsequent modeling is based on clean, correctly engineered data.
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

# Average Age Merge (Fixed typo: 'Country_Name')
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

# Win Rate Merge (Fixed right_on column)
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

# --- 2. Model Evaluation Function ---

def evaluate_model(model, X_test, y_test, model_name):
    """Calculates metrics, plots confusion matrix and ROC curve."""
    
    # 1. Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 2. Metrics (Accuracy, Precision, Recall, F1)
    print(f"\n===== {model_name} Metrics =====")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Win (0)', 'Home Win (1)'], 
                yticklabels=['No Win (0)', 'Home Win (1)'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 4. ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    plt.figure(figsize=(4, 3))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
    
    return {'Model': model_name, 'Accuracy': accuracy_score(y_test, y_pred), 'ROC-AUC': roc_auc}

# --- 3. Model Training and Hyperparameter Tuning ---

# A. Optimized Random Forest (using best params from previous step's output)
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='mean')),
    ('classifier', RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42))
])

print("Training Optimized Random Forest...")
rf_pipeline.fit(X_train, y_train)

# B. Baseline Logistic Regression (with Scaling for fair comparison)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(random_state=42, solver='liblinear'))
])

print("Training Baseline Logistic Regression...")
lr_pipeline.fit(X_train, y_train)

# --- 4. Evaluation and Comparison ---

results = []
results.append(evaluate_model(rf_pipeline, X_test, y_test, "Random Forest (Tuned)"))
results.append(evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression (Scaled)"))

print("\n\n--- Model Comparison Summary ---")
comparison_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print(comparison_df.to_markdown(index=False))