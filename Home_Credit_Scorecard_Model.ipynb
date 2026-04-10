## 1. Data Loading

import pandas as pd

# Load the main training dataset
try:
    df_train = pd.read_csv('application_train.csv')
    print("application_train.csv loaded successfully.")
except FileNotFoundError:
    print("Error: 'application_train.csv' not found. Please upload the file or ensure the path is correct.")
    df_train = None # Set to None to avoid further errors

## 2. Initial Data Exploration

if df_train is not None:
    print("\n--- First 5 rows of application_train.csv ---")
    display(df_train.head())

    print("\n--- Dataset Information (Data Types and Non-Null Counts) ---")
    df_train.info()

    print("\n--- Missing Values Count ---")
    missing_values = df_train.isnull().sum()
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    display(pd.DataFrame({'Missing Count': missing_values, 'Missing Percentage': (missing_values / len(df_train)) * 100}))

    print("\n--- Descriptive Statistics ---")
    display(df_train.describe())

    print("\n--- Target Variable Distribution ---")
    display(df_train['TARGET'].value_counts(normalize=True) * 100)

## 3. Data Preprocessing

### 3.1. Handling Missing Values

# Calculate the percentage of missing values for each column
missing_percentage = df_train.isnull().sum() / len(df_train) * 100

# Display columns with a missing value percentage above a threshold (e.g., 50%)
threshold = 50
columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()

print(f"Columns to be dropped (missing values > {threshold}%): {len(columns_to_drop)} columns")
print(columns_to_drop)

# Drop these columns from the DataFrame
df_train_interim = df_train.drop(columns=columns_to_drop)
print(f"Number of columns after dropping: {df_train_interim.shape[1]}")

# Update the list of columns that still have missing values
missing_values_after_drop = df_train_interim.isnull().sum()
missing_values_after_drop = missing_values_after_drop[missing_values_after_drop > 0].sort_values(ascending=False)

print("\n--- Missing Values After Column Dropping --- ")
display(pd.DataFrame({'Missing Count': missing_values_after_drop, 'Missing Percentage': (missing_values_after_drop / len(df_train_interim)) * 100}))

# Separate numerical and categorical columns that still have missing values
numerical_cols_with_missing = df_train_interim.select_dtypes(include=['int64', 'float64']).columns
categorical_cols_with_missing = df_train_interim.select_dtypes(include=['object']).columns

# Impute numerical columns with the median
for col in numerical_cols_with_missing:
    if df_train_interim[col].isnull().any():
        median_val = df_train_interim[col].median()
        df_train_interim[col] = df_train_interim[col].fillna(median_val)

# Impute categorical columns with the mode
for col in categorical_cols_with_missing:
    if df_train_interim[col].isnull().any():
        mode_val = df_train_interim[col].mode()[0]
        df_train_interim[col] = df_train_interim[col].fillna(mode_val)

print("\n--- Missing Values Verification After Imputation --- ")
missing_values_final = df_train_interim.isnull().sum()
missing_values_final = missing_values_final[missing_values_final > 0]

if missing_values_final.empty:
    print("No more missing values in the DataFrame.")
else:
    display(pd.DataFrame({'Missing Count': missing_values_final}))

### 3.2. Handling Categorical Variables (One-Hot Encoding)

# Identify categorical columns
categorical_cols = df_train_interim.select_dtypes(include=['object']).columns

print(f"Categorical columns to be encoded: {len(categorical_cols)} columns")
print(categorical_cols.tolist())

# Perform One-Hot Encoding
df_train_processed = pd.get_dummies(df_train_interim, columns=categorical_cols, dummy_na=False)

print(f"\nNumber of columns after One-Hot Encoding: {df_train_processed.shape[1]}")

# Display the first 5 rows of the DataFrame after encoding
print("\n--- First 5 Rows of DataFrame After One-Hot Encoding ---")
display(df_train_processed.head())

### 3.3. Preparing Data for Modeling (Splitting Features and Target)

from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df_train_processed.drop('TARGET', axis=1)
y = df_train_processed['TARGET']

# Handle any remaining non-numeric columns (e.g., booleans)
# Convert booleans to integers (0 or 1)
for col in X.select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)

# Drop the 'SK_ID_CURR' column as it's not used as a feature in modeling
X = X.drop('SK_ID_CURR', axis=1, errors='ignore')

print(f"Number of features (X) after preparation: {X.shape[1]}")
print(f"Size of target variable (y): {y.shape[0]}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data size (X_train): {X_train.shape}")
print(f"Testing data size (X_test): {X_test.shape}")
print(f"Target distribution in training data:\n{y_train.value_counts(normalize=True)}")
print(f"Target distribution in testing data:\n{y_test.value_counts(normalize=True)}")

## 4. Modeling

### 4.1. Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initialize Logistic Regression model
# Use `class_weight='balanced'` to handle class imbalance
log_reg_model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)

# Train the model on the training data
print("Training Logistic Regression model...")
log_reg_model.fit(X_train, y_train)
print("Logistic Regression model training completed.")

# Make probability predictions on the testing data
y_pred_log_reg = log_reg_model.predict_proba(X_test)[:, 1]

# Calculate and display the ROC-AUC score
roc_auc_log_reg = roc_auc_score(y_test, y_pred_log_reg)
print(f"ROC-AUC Score (Logistic Regression): {roc_auc_log_reg:.4f}")

### 4.2. XGBoost Classifier

import xgboost as xgb
from sklearn.metrics import roc_auc_score

# Initialize XGBoost Classifier model
# Use `scale_pos_weight` to handle class imbalance
# `scale_pos_weight` is calculated as (number_of_negative_cases / number_of_positive_cases)
positive_cases = y_train.sum()
negative_cases = len(y_train) - positive_cases
scale_pos_weight = negative_cases / positive_cases

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=200, # Initial number of estimators
    learning_rate=0.1,
    max_depth=5,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    scale_pos_weight=scale_pos_weight # Handle class imbalance
)

# Train the model on the training data
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("XGBoost model training completed.")

# Make probability predictions on the testing data
y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate and display the ROC-AUC score
roc_auc_xgb = roc_auc_score(y_test, y_pred_xgb)
print(f"ROC-AUC Score (XGBoost): {roc_auc_xgb:.4f}")

### 4.3. Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Initialize Decision Tree Classifier model
# Use `class_weight='balanced'` to handle class imbalance
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')

# Train the model on the training data
print("Training Decision Tree model...")
dt_model.fit(X_train, y_train)
print("Decision Tree model training completed.")

# Make probability predictions on the testing data
y_pred_dt = dt_model.predict_proba(X_test)[:, 1]

# Calculate and display the ROC-AUC score
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print(f"ROC-AUC Score (Decision Tree): {roc_auc_dt:.4f}")

## 5. Model Evaluation Summary

from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

print("Starting performance evaluation for each model...")

# --- Logistic Regression ---
# Predictions for Logistic Regression
y_train_pred_log_reg = log_reg_model.predict(X_train)
y_test_pred_log_reg = log_reg_model.predict(X_test)
y_test_pred_proba_log_reg = log_reg_model.predict_proba(X_test)[:, 1] # ROC-AUC uses probabilities

# Calculate metrics for Logistic Regression
train_accuracy_log_reg = accuracy_score(y_train, y_train_pred_log_reg) * 100
test_accuracy_log_reg = accuracy_score(y_test, y_test_pred_log_reg) * 100
roc_auc_log_reg_score = roc_auc_score(y_test, y_test_pred_proba_log_reg) * 100 # Convert to percentage

# --- XGBoost Classifier ---
# Predictions for XGBoost
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)
y_test_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1] # ROC-AUC uses probabilities

# Calculate metrics for XGBoost
train_accuracy_xgb = accuracy_score(y_train, y_train_pred_xgb) * 100
test_accuracy_xgb = accuracy_score(y_test, y_test_pred_xgb) * 100
roc_auc_xgb_score = roc_auc_score(y_test, y_test_pred_proba_xgb) * 100 # Convert to percentage

# --- Decision Tree Classifier ---
# Predictions for Decision Tree
y_train_pred_dt = dt_model.predict(X_train)
y_test_pred_dt = dt_model.predict(X_test)
y_test_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1] # ROC-AUC uses probabilities

# Calculate metrics for Decision Tree
train_accuracy_dt = accuracy_score(y_train, y_train_pred_dt) * 100
test_accuracy_dt = accuracy_score(y_test, y_test_pred_dt) * 100
roc_auc_dt_score = roc_auc_score(y_test, y_test_pred_proba_dt) * 100 # Convert to percentage

print("Performance evaluation completed.")

# Create a DataFrame to display the results neatly
results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'XGBoost', 'Decision Tree'],
    'Training Accuracy (%)': [train_accuracy_log_reg, train_accuracy_xgb, train_accuracy_dt],
    'Testing Accuracy (%)': [test_accuracy_log_reg, test_accuracy_xgb, test_accuracy_dt],
    'ROC-AUC Score (%)': [roc_auc_log_reg_score, roc_auc_xgb_score, roc_auc_dt_score]
})

display(results_df)

## 6. Feature Importance from XGBoost Model

import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importances from the XGBoost model
feature_importances = xgb_model.feature_importances_

# Create a DataFrame for visualization
features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

print("Top 20 Most Important Features according to XGBoost:")
display(features_df.head(20))

# Visualize Top 20 Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=features_df.head(20), hue='Feature', palette='viridis', legend=False)
plt.title('Top 20 Feature Importance from XGBoost Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

## 7. Visualizing Key Insights

### 7.1. Insight 1: External Credit Quality and Employment Status

First, let's look at the distribution of `EXT_SOURCE_2` and `EXT_SOURCE_3` (external source scores) compared to the target variable (`TARGET`). Lower scores on `EXT_SOURCE` generally indicate higher risk.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 6))

# Plot EXT_SOURCE_2 vs TARGET
plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
sns.violinplot(x='TARGET', y='EXT_SOURCE_2', data=df_train_processed, hue='TARGET', palette='viridis', legend=False)
plt.title('EXT_SOURCE_2 Distribution by Target')
plt.xlabel('Target (0: Repaid, 1: Default)')
plt.ylabel('EXT_SOURCE_2 Score')

# Plot EXT_SOURCE_3 vs TARGET
plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
sns.violinplot(x='TARGET', y='EXT_SOURCE_3', data=df_train_processed, hue='TARGET', palette='plasma', legend=False)
plt.title('EXT_SOURCE_3 Distribution by Target')
plt.xlabel('Target (0: Repaid, 1: Default)')
plt.ylabel('EXT_SOURCE_3 Score')

plt.tight_layout()
plt.show()

Next, let's visualize the relationship between `NAME_INCOME_TYPE` (income type) and `FLAG_EMP_PHONE` (ownership of a mobile phone registered as an office phone) with the probability of default (`TARGET`). `FLAG_EMP_PHONE` indicates employment stability.

plt.figure(figsize=(16, 6))

# Plot NAME_INCOME_TYPE vs TARGET (Mean of Target)
plt.subplot(1, 2, 1)
sns.barplot(x='NAME_INCOME_TYPE', y='TARGET', data=df_train_processed, hue='NAME_INCOME_TYPE', palette='coolwarm', legend=False)
plt.title('Default Rate by Income Type')
plt.xlabel('Income Type')
plt.ylabel('Default Rate (Mean Target)')
plt.xticks(rotation=45, ha='right')

# Plot FLAG_EMP_PHONE vs TARGET (Mean of Target)
plt.subplot(1, 2, 2)
sns.barplot(x='FLAG_EMP_PHONE', y='TARGET', data=df_train_processed, hue='FLAG_EMP_PHONE', palette='rocket', legend=False)
plt.title('Default Rate by Employee Phone Flag (1: Has Phone, 0: No Phone)')
plt.xlabel('FLAG_EMP_PHONE')
plt.ylabel('Default Rate (Mean Target)')
plt.xticks([0, 1], ['No Employee Phone', 'Has Employee Phone'])

plt.tight_layout()
plt.show()

### 7.2. Insight 2: Education and Consumer Demographics

Finally, let's visualize how education level, car ownership, and client region rating (`REGION_RATING_CLIENT_W_CITY`) are related to the probability of default (`TARGET`).

plt.figure(figsize=(18, 6))

# Plot NAME_EDUCATION_TYPE vs TARGET (Mean of Target)
plt.subplot(1, 3, 1)
sns.barplot(x='NAME_EDUCATION_TYPE', y='TARGET', data=df_train_processed, hue='NAME_EDUCATION_TYPE', palette='crest', legend=False)
plt.title('Default Rate by Education Type')
plt.xlabel('Education Type')
plt.ylabel('Default Rate (Mean Target)')
plt.xticks(rotation=45, ha='right')

# Plot FLAG_OWN_CAR vs TARGET (Mean of Target)
plt.subplot(1, 3, 2)
sns.barplot(x='FLAG_OWN_CAR', y='TARGET', data=df_train_processed, hue='FLAG_OWN_CAR', palette='mako', legend=False)
plt.title('Default Rate by Car Ownership')
plt.xlabel('Owns Car (N: No, Y: Yes)')
plt.ylabel('Default Rate (Mean Target)')

# Plot REGION_RATING_CLIENT_W_CITY vs TARGET (Mean of Target)
plt.subplot(1, 3, 3)
sns.barplot(x='REGION_RATING_CLIENT_W_CITY', y='TARGET', data=df_train_processed, hue='REGION_RATING_CLIENT_W_CITY', palette='rocket', legend=False)
plt.title('Default Rate by Region Rating (Client in City)')
plt.xlabel('Region Rating (Client in City)')
plt.ylabel('Default Rate (Mean Target)')

plt.tight_layout()
plt.show()
