import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

# Load
df = pd.read_excel('dataset/raw_dataset/Initial Public Offering.xlsx')
df_clean = df.drop(['IPO_Name', 'Date'], axis=1)
df_clean = df_clean.dropna(subset=['Listing Gain'])
numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())

def classify_risk(gain):
    if gain > 25: return 'Low'
    elif gain >= 10: return 'Medium'
    else: return 'High'

df_clean['Risk_Category'] = df_clean['Listing Gain'].apply(classify_risk)

# Load saved models
classifier = joblib.load('models/fundamental_classifier.pkl')
regressor = joblib.load('models/fundamental_regressor.pkl')
scaler = joblib.load('models/fundamental_scaler.pkl')
features = joblib.load('models/fundamental_features.pkl')
le = joblib.load('models/fundamental_label_encoder.pkl')

# Recreate features
df_clean['QIB_HNI_Ratio'] = df_clean['QIB'] / (df_clean['HNI'] + 1e-5)
df_clean['QIB_Total_Ratio'] = df_clean['QIB'] / (df_clean['Total'] + 1e-5)
df_clean['HNI_Total_Ratio'] = df_clean['HNI'] / (df_clean['Total'] + 1e-5)
df_clean['RII_Total_Ratio'] = df_clean['RII'] / (df_clean['Total'] + 1e-5)
df_clean['Subscription_Strength'] = df_clean['QIB'] + df_clean['HNI'] + df_clean['RII']
df_clean['Log_Issue_Size'] = np.log1p(df_clean['Issue_Size(crores)'])
df_clean['Log_Offer_Price'] = np.log1p(df_clean['Offer Price'])
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

X = df_clean[features]
y_cls = df_clean['Risk_Category']
y_reg = df_clean['Listing Gain']

y_cls_encoded = le.transform(y_cls)
X_scaled = scaler.transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_cls_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

y_pred = classifier.predict(X_test)
y_pred_reg = regressor.predict(X_test_reg)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
r2 = r2_score(y_test_reg, y_pred_reg)

cv_scores = cross_val_score(classifier, X_resampled, y_resampled, cv=5, scoring='accuracy')

with open('model_results.txt', 'w') as f:
    f.write("=== CLASS DISTRIBUTION ===\n")
    f.write(str(df_clean['Risk_Category'].value_counts()) + "\n\n")
    f.write("=== CLASSIFICATION METRICS (XGBoost + SMOTE) ===\n")
    f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.1f}%)\n")
    f.write(f"Precision: {precision:.4f} ({precision*100:.1f}%)\n")
    f.write(f"Recall:    {recall:.4f} ({recall*100:.1f}%)\n")
    f.write(f"F1 Score:  {f1:.4f} ({f1*100:.1f}%)\n\n")
    target_names = le.inverse_transform(sorted(np.unique(y_test)))
    f.write(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
    f.write(f"\n5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    f.write(f"\n=== REGRESSION METRICS ===\n")
    f.write(f"R2 Score: {r2:.4f} ({r2*100:.1f}%)\n")

print("Results saved to model_results.txt")
