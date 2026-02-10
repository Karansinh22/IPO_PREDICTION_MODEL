import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os

# 1. Load data
raw_data_path = 'dataset/raw_dataset/Initial Public Offering.xlsx'
df = pd.read_excel(raw_data_path)

# 2. Preprocessing
df_clean = df.drop(['IPO_Name', 'Date'], axis=1)
df_clean = df_clean.dropna(subset=['Listing Gain'])
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# Risk Categories
def classify_risk(gain):
    if gain > 25: return 'Low'
    elif gain >= 10: return 'Medium'
    else: return 'High'

df_clean['Risk_Category'] = df_clean['Listing Gain'].apply(classify_risk)

# 3. Advanced Feature Engineering
# Add some interaction terms
df_clean['QIB_Total_Ratio'] = df_clean['QIB'] / (df_clean['Total'] + 1e-5)
df_clean['HNI_Total_Ratio'] = df_clean['HNI'] / (df_clean['Total'] + 1e-5)
df_clean['RII_Total_Ratio'] = df_clean['RII'] / (df_clean['Total'] + 1e-5)
df_clean['Log_Issue_Size'] = np.log1p(df_clean['Issue_Size(crores)'])

features = ['Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price', 
            'QIB_Total_Ratio', 'HNI_Total_Ratio', 'RII_Total_Ratio', 'Log_Issue_Size']

X = df_clean[features]
y = df_clean['Risk_Category']

# Scale features (RobustScaler handles outliers better)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# 4. Stratified Split to ensure class balance
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Gradient Boosting (usually better than RF for high accuracy)
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
gbc.fit(X_train, y_train)

y_pred = gbc.predict(X_test)
print("--- GBC Initial Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 6. Hyperparameter Tuning for Gradient Boosting
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

print(f"\nBest accuracy from GridSearch: {grid_search.best_score_:.4f}")
print(f"Best params: {grid_search.best_params_}")

# Final check with best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print("\n--- Final Best Model Performance (Validation Set) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")

# Save the best model and scaler
if not os.path.exists('models_v2'):
    os.makedirs('models_v2')
joblib.dump(best_model, 'models_v2/fundamental_classifier_v2.pkl')
joblib.dump(scaler, 'models_v2/fundamental_scaler_v2.pkl')
joblib.dump(features, 'models_v2/fundamental_features_v2.pkl')
