import pandas as pd
import numpy as np

# --- 1. DATA SETUP ---
# Dataset from the homework problem
data = pd.DataFrame({
    'Student': [1, 2, 3, 4, 5],
    'Study Hours': [2, 4, 6, 8, 10],
    'Pass (1)/Fail (0)': [0, 0, 1, 1, 1]
})

feature = 'Study Hours'
target = 'Pass (1)/Fail (0)'
# Possible split points (midpoints between consecutive unique Study Hours)
split_points = [3, 5, 7, 9] 

# --- 2. GINI IMPURITY FUNCTIONS ---

def calculate_gini_impurity(data_subset, target_col):
    """Calculates the Gini impurity for a single node."""
    N = len(data_subset)
    if N == 0:
        return 0.0

    # Get class counts (0 and 1)
    class_counts = data_subset[target_col].value_counts()

    # Calculate probabilities and sum of squares
    sum_of_squares = 0
    for count in class_counts:
        p_i = count / N
        sum_of_squares += p_i**2

    # Gini = 1 - sum(p_i^2)
    gini = 1.0 - sum_of_squares
    return gini

def calculate_weighted_gini_split(data, target_col, feature_col, split_value):
    """Calculates the weighted Gini impurity for a split."""
    N = len(data)

    # Split the data: left <= split_value, right > split_value
    left_data = data[data[feature_col] <= split_value]
    right_data = data[data[feature_col] > split_value]

    N_left = len(left_data)
    N_right = len(right_data)

    # Calculate Gini for child nodes
    gini_left = calculate_gini_impurity(left_data, target_col)
    gini_right = calculate_gini_impurity(right_data, target_col)

    # Weighted Gini Impurity formula
    weighted_gini = (N_left / N) * gini_left + (N_right / N) * gini_right

    return weighted_gini, N_left, N_right, gini_left, gini_right

# --- 3. EXECUTION AND OUTPUT ---

print("--- 1. Dataset ---")
print(data.to_string(index=False))

# --- A. Root Node Impurity ---
root_gini = calculate_gini_impurity(data, target)
print(f"\n--- 2. Root Node Gini Impurity (Initial) ---")
print(f"Gini Impurity: {root_gini:.4f}")

# --- B. Evaluate All Splits ---
results = []
best_split = {'split_value': None, 'weighted_gini': float('inf')}

print("\n--- 3. Evaluating All Possible Splits ---")
for split_value in split_points:
    weighted_gini, n_left, n_right, gini_left, gini_right = calculate_weighted_gini_split(
        data, target, feature, split_value
    )
    
    # Gini Gain is used to compare splits; higher is better.
    gini_gain = root_gini - weighted_gini

    results.append({
        'Split (Study Hours)': f'<= {split_value}',
        'Weighted Gini': weighted_gini,
        'Gini Gain': gini_gain,
        'Gini Left': gini_left,
        'Gini Right': gini_right,
    })

    if weighted_gini < best_split['weighted_gini']:
        best_split['weighted_gini'] = weighted_gini
        best_split['split_value'] = split_value

# Output the Gini impurity at each step
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False, float_format="%.4f"))

# --- C. Chosen Split and Visualization ---
final_split_value = best_split['split_value']
final_wg = best_split['weighted_gini']
final_gain = root_gini - final_wg

# Get final node details for visualization
wg, nl, nr, gl, gr = calculate_weighted_gini_split(data, target, feature, final_split_value)
left_data = data[data[feature] <= final_split_value]
right_data = data[data[feature] > final_split_value]

print("\n" + "=" * 50)
print(f"--- 4. Chosen Split and First-Level Tree Visualization ---")
print(f"Chosen Split (Lowest Weighted Gini): **Study Hours <= {final_split_value}**")
print(f"Weighted Gini Impurity: {final_wg:.4f}")
print(f"Gini Gain: {final_gain:.4f}")
print("=" * 50)

# Text Visualization
print("\nROOT (N=5, Gini={:.4f})".format(root_gini))
print("  |")
print(f"  |--- IF Study Hours <= {final_split_value} (LEFT CHILD)")
print(f"  |       (N={nl}, Gini={gl:.4f})")
print(f"  |       Classes: Pass({left_data[target].sum()}), Fail({nl - left_data[target].sum()})")
print(f"  |       Prediction: {'Fail (0)' if left_data[target].mean() < 0.5 else 'Pass (1)'} (Pure Node: {gl == 0.0})")
print("  |")
print(f"  |--- ELSE (Study Hours > {final_split_value}) (RIGHT CHILD)")
print(f"          (N={nr}, Gini={gr:.4f})")
print(f"          Classes: Pass({right_data[target].sum()}), Fail({nr - right_data[target].sum()})")
print(f"          Prediction: {'Fail (0)' if right_data[target].mean() < 0.5 else 'Pass (1)'} (Pure Node: {gr == 0.0})")
