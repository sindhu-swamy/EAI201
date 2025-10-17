import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set a style for the plots
plt.style.use('ggplot')

# ==============================================================================
## Task A1: Data Loading and Exploration
# ==============================================================================

print("="*60)
print("Task A1: Data Loading and Exploration")
print("="*60)

# Load the California Housing dataset (as a substitute for Boston Housing)
# The dataset's target is the median house value in hundreds of thousands of dollars.
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame
target_name = california_housing.target_names[0] # 'MedHouseVal'

# Create a DataFrame and examine its structure
# The loaded object is already a DataFrame, but we ensure the target is included
data['Price'] = california_housing.target
df = data.copy()

print(f"Dataset used: California Housing (substitute for Boston Housing)")
print("\n--- DataFrame Structure (Head) ---")
print(df.head())
print("\n--- DataFrame Information ---")
df.info()

# How many samples and features are in the dataset?
n_samples, n_features = df.shape[0], df.shape[1] - 1 # Subtract 1 for the 'Price' column
print(f"\n--- Data Dimensions ---")
print(f"Number of samples (rows): {n_samples}")
print(f"Number of features (columns): {n_features}")

# What is the price range (min, max, mean) of the target variable?
price_min = df['Price'].min()
price_max = df['Price'].max()
price_mean = df['Price'].mean()

print(f"\n--- Price Range (Target Variable: Median House Value - $100k) ---")
print(f"Minimum Price: ${price_min:.2f} (x $100,000)")
print(f"Maximum Price: ${price_max:.2f} (x $100,000)")
print(f"Mean Price: ${price_mean:.2f} (x $100,000)")


# ==============================================================================
## Task A2: Exploratory Data Analysis
# ==============================================================================

print("\n" + "="*60)
print("Task A2: Exploratory Data Analysis")
print("="*60)

# Create a histogram of house prices
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=50, kde=True)
plt.title('Histogram of Median House Prices')
plt.xlabel('Price (x $100,000)')
plt.ylabel('Frequency')
plt.show() # 
# Calculate correlation matrix between features and price
# We only care about correlation with the 'Price' column
correlation_matrix = df.corr()
price_correlations = correlation_matrix['Price'].sort_values(ascending=False)

print("\n--- Correlation of Features with Price ---")
print(price_correlations)

# Which feature has the strongest positive correlation with price?
# (Excluding 'Price' itself)
strongest_positive_feature = price_correlations.index[1] # The first is 'Price' itself
strongest_positive_corr = price_correlations.iloc[1]

print(f"\nFeature with the strongest positive correlation with Price: '{strongest_positive_feature}' (Correlation: {strongest_positive_corr:.4f})")

# Create a scatter plot between price and the feature most correlated with price
plt.figure(figsize=(8, 5))
plt.scatter(df[strongest_positive_feature], df['Price'], alpha=0.5)
plt.title(f'Scatter Plot: Price vs. {strongest_positive_feature}')
plt.xlabel(strongest_positive_feature)
plt.ylabel('Price (x $100,000)')
plt.grid(True)
plt.show() # 
# Is the price distribution normal or skewed? What does this suggest?
price_skew = df['Price'].skew()
print(f"\n--- Price Distribution Skewness ---")
print(f"Skewness of Price distribution: {price_skew:.4f}")
if price_skew > 0.5:
    print("The distribution is **positively (right) skewed**.")
    print("This suggests that most houses have lower-to-mid prices, with a long tail of fewer, very expensive houses.")
elif price_skew < -0.5:
    print("The distribution is **negatively (left) skewed**.")
else:
    print("The distribution is relatively close to a normal distribution.")

# ==============================================================================
## Task A3: Model Building and Evaluation
# ==============================================================================

print("\n" + "="*60)
print("Task A3: Model Building and Evaluation")
print("="*60)

# Define features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size (80%): {len(X_train)} samples")
print(f"Testing set size (20%): {len(X_test)} samples")

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression Model Trained Successfully.")

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate and report R^2 score and RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Evaluation ---")
print(f"R² Score (Coefficient of Determination): {r2:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

# What is your model's R^2 score? Is this considered good performance?
print(f"\n--- R² Score Interpretation ---")
print(f"My model's R² score is **{r2:.4f}**.")
if r2 > 0.7:
    print("This is generally considered **good performance** for a basic linear regression model on real-world housing data, meaning the features explain about 60-70% of the variance in house prices.")
elif r2 > 0.5:
    print("This is considered **fair performance**. There is room for improvement, likely by using more complex models or feature engineering.")
else:
    print("This is considered **poor performance**, suggesting a linear model is not a good fit for this data or critical features are missing.")

# Create scatter plot of actual vs predicted prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)

# Add a perfect prediction line (y=x)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y=x)')

plt.title('Actual vs. Predicted Median House Prices')
plt.xlabel('Actual Price (y_test)')
plt.ylabel('Predicted Price (y_pred)')
plt.legend()
plt.grid(True)
plt.show() # 
# Based on the actual vs predicted plot, does your model perform better on certain price ranges?
residuals = y_test - y_pred

print(f"\n--- Performance Analysis from Plot ---")
print("Based on the plot, we look for where the points cluster closest to the 'Perfect Prediction' line (red dashed line).")
print("Observation:")
print("- The model seems to perform **best on the lower to mid-price range** (e.g., prices below ~$4.0). The points are tightly clustered around the red line here.")
print("- The model performs **worse on the highest price range** (e.g., prices above ~$4.5), showing a clear 'ceiling' effect. The model often under-predicts the most expensive houses, as the predicted values don't extend as high as the actual values.")
print("- This suggests the linear model is **limited in capturing the non-linear factors** that drive the prices of the most expensive homes.")

print("\n" + "="*60)
print("Lab Complete.")
print("="*60)
