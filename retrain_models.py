import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Create directory to save models
os.makedirs('saved_models', exist_ok=True)

# 1. Train Diabetes Model
diabetes_data = pd.read_csv('dataset/diabetes.csv')

X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

diabetes_model = RandomForestClassifier()
diabetes_model.fit(X_train, y_train)

# Save the trained Diabetes model
with open('saved_models/diabetes.pkl', 'wb') as f:
    pickle.dump(diabetes_model, f)

print("✅ Diabetes model saved!")

# 2. Train Heart Disease Model
heart_data = pd.read_csv('dataset/heart.csv')

X = heart_data.drop('target', axis=1)
y = heart_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

heart_model = RandomForestClassifier()
heart_model.fit(X_train, y_train)

# Save the trained Heart disease model
with open('saved_models/heart.pkl', 'wb') as f:
    pickle.dump(heart_model, f)

print("✅ Heart disease model saved!")

# 3. Train Kidney Disease Model
kidney_data = pd.read_csv('dataset/kidney_disease.csv')

# Check for missing values in the entire dataset before dropping
print("Missing values in each column before cleaning:")
print(kidney_data.isnull().sum())

# Ensure 'classification' does not contain NaN values before cleaning
if kidney_data['classification'].isnull().any():
    print("\nWarning: 'classification' column contains NaN values before cleaning.")

# Drop rows with missing target variable 'classification'
kidney_data = kidney_data.dropna(subset=['classification'])  # Drop rows where 'classification' (target) is NaN

# Check for NaN values in 'classification' after dropping rows
if kidney_data['classification'].isnull().any():
    raise ValueError("There are still NaN values in the 'classification' column after cleaning!")

# Drop all other rows with missing values
kidney_data = kidney_data.dropna()

# Check again for missing values after cleaning
print("\nMissing values in each column after cleaning:")
print(kidney_data.isnull().sum())

# Check unique values in 'classification' column before mapping
print("\nUnique values in 'classification' column before mapping:", kidney_data['classification'].unique())

# Handle any NaN values manually in 'classification' before encoding
kidney_data['classification'] = kidney_data['classification'].fillna('notckd')  # Default to 'notckd' if NaN

# Map 'classification' to numerical values
kidney_data['classification'] = kidney_data['classification'].map({'ckd': 1, 'notckd': 0})

# Verify the target variable doesn't have NaN after mapping
if kidney_data['classification'].isnull().any():
    raise ValueError("Target variable 'classification' still contains NaN values after mapping!")

# **Encoding all categorical features (like 'normal', 'abnormal', etc.)**
categorical_columns = kidney_data.select_dtypes(include=['object']).columns.tolist()

# Apply LabelEncoder to all categorical columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    kidney_data[col] = label_encoder.fit_transform(kidney_data[col])

# Separate features and target
X = kidney_data.drop('classification', axis=1)
y = kidney_data['classification']

# Ensure no empty target variable (y)
if len(y) == 0:
    raise ValueError("Target variable y is empty!")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
kidney_model = RandomForestClassifier()
kidney_model.fit(X_train, y_train)

# Save the trained Kidney disease model
with open('saved_models/kidney.pkl', 'wb') as f:
    pickle.dump(kidney_model, f)

print("✅ Kidney disease model saved!")

print("\n🎯 All models retrained and saved successfully!")
