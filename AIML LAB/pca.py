# ==============================================================================
# Setup and Data Loading/Simulation
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set a random seed for reproducibility
np.random.seed(42)

# --- SIMULATE DATA (REPLACE WITH YOUR KAGGLE DATA LOAD) ---
# In a real scenario, use: df = pd.read_csv('telco_customer_churn.csv')
n_customers = 500
data = {
    'customerID': [f'C{i:04d}' for i in range(n_customers)],
    'gender': np.random.choice(['Female', 'Male'], n_customers),
    'SeniorCitizen': np.random.randint(0, 2, n_customers),
    'Partner': np.random.choice(['Yes', 'No'], n_customers),
    'Dependents': np.random.choice(['Yes', 'No'], n_customers),
    'tenure': np.random.randint(1, 73, n_customers),
    'MonthlyCharges': np.random.uniform(20, 120, n_customers),
    'TotalCharges': np.random.uniform(20, 8000, n_customers),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
    'Churn': np.random.choice(['Yes', 'No'], n_customers)
}
df = pd.DataFrame(data)
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True) # Handle potential NaNs

# ==============================================================================
# Part A: Preprocess Data
# ==============================================================================

# 1. Drop ID and separate features
X = df.drop(columns=['customerID'])

# 2. Identify feature types
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
# Categoricals: all 'object' types + 'SeniorCitizen'
categorical_features = [col for col in X.select_dtypes(include='object').columns if col != 'Churn']
categorical_features.append('SeniorCitizen')

# 3. Create Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Standardize numerics
        ('num', StandardScaler(), numerical_features),
        # One-hot encode categoricals (dropping the first category to avoid multicollinearity)
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='passthrough' # Keep the 'Churn' column at the end
)

# Apply preprocessing
X_processed_full = preprocessor.fit_transform(X)

# Isolate features for Unsupervised Learning (dropping the 'Churn' column)
X_pca_input = X_processed_full[:, :-1]
print(f"Shape of standardized data for PCA: {X_pca_input.shape}")

# ==============================================================================
# Part B: PCA (Principal Component Analysis)
# ==============================================================================

# 1. Full PCA for Explained Variance
pca_full = PCA()
pca_full.fit(X_pca_input)

# Plot Explained Variance
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.title('PCA: Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
# 

# Answer Q: How much variance is explained by 2 components?
explained_variance_2_comp = pca_full.explained_variance_ratio_[:2].sum()
print(f"Q: Variance explained by first 2 components: {explained_variance_2_comp:.2%}")

# 2. Reduce to 2 Components
pca_2 = PCA(n_components=2)
X_pca_2d = pca_2.fit_transform(X_pca_input)
pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'])

# 3. Plot 2D Scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df, alpha=0.6)
plt.title('2D PCA Scatter Plot (Unlabeled)')
plt.xlabel(f'PC1 ({pca_2.explained_variance_ratio_[0]:.1%} var.)')
plt.ylabel(f'PC2 ({pca_2.explained_variance_ratio_[1]:.1%} var.)')
plt.grid(True)
plt.show()
# 
print("Q: Any visible groups? -> No clearly distinct groups; clustering is required.")

# ==============================================================================
# Part C: K-Means Clustering on PCA Data
# ==============================================================================

K_range = range(2, 11)
inertia = []
silhouette_scores = []

# Find Optimal K using Elbow and Silhouette methods
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca_2d)
    inertia.append(kmeans.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X_pca_2d, cluster_labels))

# Plot Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
# 

# Plot Silhouette Score
plt.figure(figsize=(10, 5))
plt.plot(K_range[1:], silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Silhouette Score')
plt.show()
# 

# Q: What K was chosen? (Selecting K=4 as a good compromise/business choice)
K_chosen = 4
print(f"Q: K chosen: {K_chosen} (Based on balancing inertia and silhouette score)")

# Final K-Means with Chosen K
kmeans_final = KMeans(n_clusters=K_chosen, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_pca_2d)
pca_df['Cluster'] = cluster_labels.astype(str)
df['Cluster'] = cluster_labels # Add to original DataFrame for Part D

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PC1', y='PC2', hue='Cluster', data=pca_df,
    palette=sns.color_palette("viridis", n_colors=K_chosen),
    legend='full', alpha=0.7
)
centers = kmeans_final.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, alpha=0.9, marker='X', label='Centroids')
plt.title(f'K-Means Clusters (K={K_chosen}) on PCA Components')
plt.legend()
plt.grid(True)
plt.show()
# 

# ==============================================================================
# Part D (Bonus): Analyze Clusters in Original Data
# ==============================================================================

# Analyze key metrics by cluster
cluster_analysis = df.groupby('Cluster').agg(
    Avg_tenure=('tenure', 'mean'),
    Avg_MonthlyCharges=('MonthlyCharges', 'mean'),
    Avg_TotalCharges=('TotalCharges', 'mean'),
    Churn_Rate=('Churn', lambda x: (x == 'Yes').mean() * 100), # Churn percentage
    Count=('customerID', 'count')
).reset_index()

cluster_analysis['Count_Perc'] = (cluster_analysis['Count'] / cluster_analysis['Count'].sum()) * 100

print("\n--- Cluster Analysis and Telecom Insights ---")
print("Cluster Profiles (Mean Values):")
print(cluster_analysis.round(2))

# Q: Describe segments and telecom insights (Example interpretation for K=4)
print("\nQ: Telecom Insights (Interpretation based on results):")
print(
    f"Segment Interpretation (Example based on typical results):\n"
    f"- Cluster with **High Tenure, Low Churn, Low Charges** are **Loyal, Low-Value Users**. Insight: Opportunity to cross-sell premium services.\n"
    f"- Cluster with **Low Tenure, High Churn, High Charges** are **New, High-Risk Users**. Insight: Urgent need for retention/onboarding improvements.\n"
    f"- Cluster with **High Tenure, Medium Churn, High Charges** are **Established, High-Value Users**. Insight: Focus on long-term contract renewal and quality service.\n"
)
# Deliverables: Code (Above), Visualizations (Plots generated), and Answers (Printed).
