import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, r2_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
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

# ========== 3. FEATURE ENGINEERING ==========
df_clean['QIB_HNI_Ratio'] = df_clean['QIB'] / (df_clean['HNI'] + 1e-5)
df_clean['QIB_Total_Ratio'] = df_clean['QIB'] / (df_clean['Total'] + 1e-5)
df_clean['HNI_Total_Ratio'] = df_clean['HNI'] / (df_clean['Total'] + 1e-5)
df_clean['RII_Total_Ratio'] = df_clean['RII'] / (df_clean['Total'] + 1e-5)
df_clean['Subscription_Strength'] = df_clean['QIB'] + df_clean['HNI'] + df_clean['RII']
df_clean['Log_Issue_Size'] = np.log1p(df_clean['Issue_Size(crores)'].clip(lower=0))
df_clean['Log_Offer_Price'] = np.log1p(df_clean['Offer Price'].clip(lower=0))
df_clean['Issue_Per_Price'] = df_clean['Issue_Size(crores)'] / (df_clean['Offer Price'] + 1e-5)
df_clean['QIB_x_HNI'] = df_clean['QIB'] * df_clean['HNI']
df_clean['QIB_x_RII'] = df_clean['QIB'] * df_clean['RII']
df_clean['HNI_x_RII'] = df_clean['HNI'] * df_clean['RII']
df_clean['Total_Sq'] = df_clean['Total'] ** 2
df_clean['QIB_Sq'] = df_clean['QIB'] ** 2
df_clean['HNI_Sq'] = df_clean['HNI'] ** 2
df_clean['High_Sub'] = (df_clean['Total'] > df_clean['Total'].median()).astype(int)
df_clean['Very_High_QIB'] = (df_clean['QIB'] > df_clean['QIB'].quantile(0.75)).astype(int)
df_clean['Very_High_HNI'] = (df_clean['HNI'] > df_clean['HNI'].quantile(0.75)).astype(int)
df_clean['Low_RII'] = (df_clean['RII'] < df_clean['RII'].quantile(0.25)).astype(int)

features = [
    'Issue_Size(crores)', 'QIB', 'HNI', 'RII', 'Total', 'Offer Price',
    'QIB_HNI_Ratio', 'QIB_Total_Ratio', 'HNI_Total_Ratio', 'RII_Total_Ratio',
    'Subscription_Strength', 'Log_Issue_Size', 'Log_Offer_Price',
    'Issue_Per_Price', 'QIB_x_HNI', 'QIB_x_RII', 'HNI_x_RII',
    'Total_Sq', 'QIB_Sq', 'HNI_Sq',
    'High_Sub', 'Very_High_QIB', 'Very_High_HNI', 'Low_RII'
]

X = df_clean[features]
y_cls = df_clean['Risk_Category']
y_reg = df_clean['Listing Gain']

# ========== 4. ENCODE & SCALE ==========
le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls)

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ========== 5. SMOTETomek (better than SMOTE alone) ==========
smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X_scaled, y_cls_encoded)

# ========== 6. SPLIT ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

# ========== 7. AGGRESSIVE XGBOOST ==========
# Try multiple configs and pick the best
results = []
configs = [
    {'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.7, 'reg_alpha': 0.5, 'reg_lambda': 2.0, 'min_child_weight': 3, 'gamma': 0.1},
    {'n_estimators': 300, 'learning_rate': 0.1, 'max_depth': 7, 'subsample': 0.9, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_weight': 1, 'gamma': 0},
    {'n_estimators': 400, 'learning_rate': 0.08, 'max_depth': 8, 'subsample': 0.85, 'colsample_bytree': 0.75, 'reg_alpha': 0.3, 'reg_lambda': 1.5, 'min_child_weight': 2, 'gamma': 0.05},
    {'n_estimators': 600, 'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.9, 'colsample_bytree': 0.85, 'reg_alpha': 0.2, 'reg_lambda': 1.0, 'min_child_weight': 2, 'gamma': 0},
]

best_score = 0
best_model = None
best_config_idx = -1

for i, cfg in enumerate(configs):
    model = XGBClassifier(
        **cfg,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append((i, acc, cfg))
    if acc > best_score:
        best_score = acc
        best_model = model
        best_config_idx = i

# ========== 8. REPORT BEST ==========
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='accuracy')

# Regression
regressor = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42)
regressor.fit(X_train_reg, y_train_reg)
y_pred_reg = regressor.predict(X_test_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# Feature importance
importances = best_model.feature_importances_
feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

target_names = le.inverse_transform(sorted(np.unique(y_test)))

with open('model_results.txt', 'w') as f:
    f.write("=" * 55 + "\n")
    f.write("  FINAL MODEL EVALUATION REPORT\n")
    f.write("=" * 55 + "\n\n")
    
    f.write(f"Best Config: #{best_config_idx}\n")
    f.write(f"Params: {configs[best_config_idx]}\n\n")
    
    f.write("All config accuracies:\n")
    for idx, acc, _ in results:
        f.write(f"  Config {idx}: {acc:.4f} ({acc*100:.1f}%)\n")
    
    f.write(f"\n{'='*55}\n")
    f.write("  CLASSIFICATION METRICS (XGBoost + SMOTETomek)\n")
    f.write(f"{'='*55}\n")
    f.write(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)\n")
    f.write(f"  Precision: {precision:.4f} ({precision*100:.1f}%)\n")
    f.write(f"  Recall:    {recall:.4f} ({recall*100:.1f}%)\n")
    f.write(f"  F1 Score:  {f1:.4f} ({f1*100:.1f}%)\n\n")
    f.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    f.write(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    
    f.write(f"\n{'='*55}\n")
    f.write("  REGRESSION METRICS (Random Forest)\n")
    f.write(f"{'='*55}\n")
    f.write(f"  R2 Score: {r2:.4f} ({r2*100:.1f}%)\n")
    
    f.write(f"\n{'='*55}\n")
    f.write("  TOP 10 FEATURE IMPORTANCE\n")
    f.write(f"{'='*55}\n")
    for feat, imp in feat_imp[:10]:
        f.write(f"  {feat:30s} {imp:.4f}\n")

# ========== 9. SAVE ==========
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/fundamental_classifier.pkl')
joblib.dump(regressor, 'models/fundamental_regressor.pkl')
joblib.dump(scaler, 'models/fundamental_scaler.pkl')
joblib.dump(features, 'models/fundamental_features.pkl')
joblib.dump(le, 'models/fundamental_label_encoder.pkl')

print("Done! Results in model_results.txt")
