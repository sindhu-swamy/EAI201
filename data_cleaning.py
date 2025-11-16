import pandas as pd

# Load the dataset
df = pd.read_csv('fifa_merged_dataset.csv')

# --- 1. Remove Duplicates ---
df_no_duplicates = df.drop_duplicates()

# --- 2. Handle Missing Match Data ---
# Filter to keep only rows with complete match data (where 'Key Id' is not null)
df_cleaned = df_no_duplicates[df_no_duplicates['Key Id'].notnull()].copy()

# --- 3. Impute Missing Player Data (in the filtered set) ---
# 'team_position' (Categorical): Impute with 'Unknown'
df_cleaned['team_position'].fillna('Unknown', inplace=True)

# 'team_jersey_number' (Numerical): Impute with the median
median_jersey_number = df_cleaned['team_jersey_number'].median()
df_cleaned['team_jersey_number'].fillna(median_jersey_number, inplace=True)

# Save the cleaned DataFrame to a new CSV file
output_file_name = 'fifa_merged_dataset_cleaned.csv'
df_cleaned.to_csv(output_file_name, index=False)