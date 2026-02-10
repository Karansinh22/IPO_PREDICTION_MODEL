import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, r2_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ========== 1. LOAD & CLEAN ==========
df = pd.read_excel('dataset/raw_dataset/Initial Public Offering.xlsx')
df_clean = df.drop(['IPO_Name', 'Date'], axis=1)
df_clean = df_clean.dropna(subset=['Listing Gain'])
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

# ========== 2. RISK CATEGORIES ==========
def classify_risk(gain):
    if gain > 25: return 'Low'
    elif gain >= 10: return 'Medium'
    else: return 'High'

df_clean['Risk_Category'] = df_clean['Listing Gain'].apply(classify_risk)

# ========== 3. FEATURE ENGINEERING (Refined Top Features) ==========
df_clean['Subscription_Strength'] = df_clean['QIB'] + df_clean['HNI'] + df_clean['RII']
df_clean['Total_Sq'] = df_clean['Total'] ** 2
df_clean['High_Sub'] = (df_clean['Total'] > df_clean['Total'].median()).astype(int)
df_clean['QIB_Sq'] = df_clean['QIB'] ** 2
df_clean['HNI_Sq'] = df_clean['HNI'] ** 2
df_clean['HNI_x_RII'] = df_clean['HNI'] * df_clean['RII']
df_clean['Log_Issue_Size'] = np.log1p(df_clean['Issue_Size(crores)'].clip(lower=0))
df_clean['Issue_Per_Price'] = df_clean['Issue_Size(crores)'] / (df_clean['Offer Price'] + 1e-5)
df_clean['QIB_HNI_Ratio'] = df_clean['QIB'] / (df_clean['HNI'] + 1e-5)
df_clean['QIB_x_HNI'] = df_clean['QIB'] * df_clean['HNI']
df_clean['QIB_x_RII'] = df_clean['QIB'] * df_clean['RII']

features = [
    'Subscription_Strength', 'Total_Sq', 'High_Sub', 'Total', 'QIB_Sq', 
    'HNI_x_RII', 'HNI', 'HNI_Sq', 'QIB_x_HNI', 'QIB_x_RII', 
    'Issue_Size(crores)', 'Offer Price', 'Log_Issue_Size'
]

X = df_clean[features]
y_cls = df_clean['Risk_Category']
y_reg = df_clean['Listing Gain']

# ========== 4. ENCODE & SCALE ==========
le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ========== 5. SMOTETomek ==========
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_scaled, y_cls_encoded)

# ========== 6. ENSEMBLE MODEL (Refined XGBoosts) ==========
# XGB1 (Explorative)
xgb1 = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.03, 
    max_depth=9, 
    subsample=0.8, 
    colsample_bytree=0.7, 
    reg_alpha=0.2, 
    reg_lambda=2.0, 
    min_child_weight=2,
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

# XGB2 (Stable)
xgb2 = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.1, 
    max_depth=7, 
    subsample=0.9, 
    colsample_bytree=0.8, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)

ensemble = VotingClassifier(
    estimators=[('xgb1', xgb1), ('xgb2', xgb2)],
    voting='soft'
)

# Split (Small test set for better training on limited data, but with Cross-Val)
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.12, random_state=42, stratify=y_resampled)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Final Refined Accuracy: {accuracy*100:.2f}%")

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/fundamental_classifier.pkl')
joblib.dump(scaler, 'models/fundamental_scaler.pkl')
joblib.dump(features, 'models/fundamental_features.pkl')
joblib.dump(le, 'models/fundamental_label_encoder.pkl')

regressor = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
regressor.fit(X_scaled, y_reg)
joblib.dump(regressor, 'models/fundamental_regressor.pkl')

with open('model_results_final.txt', 'w') as f:
    f.write(f"Final Refined Accuracy: {accuracy*100:.2f}%\n")
    f.write(classification_report(y_test, y_pred, target_names=le.inverse_transform(sorted(np.unique(y_test)))))

print("Final models saved.")
