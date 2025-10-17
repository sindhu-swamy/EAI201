import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set style for plots
sns.set_style('whitegrid')

# --- Tool 1: Load the Titanic dataset ---
print("## 1. Loading the Data\n")
try:
    # Assuming the common file name for the Titanic dataset
    df = pd.read_csv('Titanic-Dataset.csv')
    print("Data loaded successfully! First 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'Titanic-Dataset.csv' not found. Please ensure the file is in the correct directory.")
    # Exiting the script if the data isn't found
    exit()


# ==============================================================================
# --- Tool 2: Exploratory Data Analysis (EDA) ---
# ==============================================================================
print("\n" + "="*80)
print("## 2. Exploratory Data Analysis (EDA)\n")

### Summarize missing values and data types
print("### Missing Values and Data Types Summary")
df.info()

print("\n--- Missing Values Count ---\n")
print(df.isnull().sum())


# --- Visualize distributions of key features ---
print("\n### Feature Distributions\n")

# Distribution of Age
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title('Distribution of Age')
plt.show() # Use plt.show() to display the plot in a non-interactive script

# Survival by Sex (Count Plot)
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival Count by Sex')
plt.show()

# Survival by Pclass
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Pclass')
plt.show()


# ==============================================================================
# --- Tool 3: Data Cleaning and Imputation ---
# ==============================================================================
print("\n" + "="*80)
print("## 3. Data Cleaning and Imputation\n")

df_clean = df.copy()

# Age: Impute with the median
df_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)

# Embarked: Impute with the most frequent value (Mode)
most_frequent_embarked = df_clean['Embarked'].mode()[0]
df_clean['Embarked'].fillna(most_frequent_embarked, inplace=True)

# Fare: Impute with the median
df_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)

# Drop irrelevant columns
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df_clean.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns: {columns_to_drop}")


# ==============================================================================
# --- Tool 4: Feature Engineering ---
# ==============================================================================
print("\n" + "="*80)
print("## 4. Feature Engineering\n")

### Create a new feature FamilySize
df_clean['FamilySize'] = df_clean['SibSp'] + df_clean['Parch'] + 1
df_clean.drop(['SibSp', 'Parch'], axis=1, inplace=True)

### Extract Titles from the Name feature (using the original df for the 'Name' column)
df_clean['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
rare_titles = ['Dr', 'Rev', 'Major', 'Col', 'Capt', 'Don', 'Jonkheer', 'Countess', 'Lady', 'Sir', 'Dona']
df_clean['Title'] = df_clean['Title'].replace(rare_titles, 'Rare')
df_clean['Title'] = df_clean['Title'].replace(['Mlle', 'Ms'], 'Miss')
df_clean['Title'] = df_clean['Title'].replace('Mme', 'Mrs')

### Convert categorical features to numeric
print("\n### Converting Categorical Features to Numeric (One-Hot Encoding)\n")
df_model = pd.get_dummies(df_clean, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
df_model = pd.get_dummies(df_model, columns=['Pclass'], prefix='Pclass', drop_first=True)
print("Final DataFrame Info after Feature Engineering:")
df_model.info()

# ==============================================================================
# --- Tool 5: Prepare Data for Modeling ---
# ==============================================================================
print("\n" + "="*80)
print("## 5. Prepare Data for Modeling\n")

X = df_model.drop('Survived', axis=1)
y = df_model['Survived']

### Finalize features and split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size (80%): {X_train.shape[0]} samples")
print(f"Test set size (20%): {X_test.shape[0]} samples")
print("\n🎉 Data is ready for model training!")
