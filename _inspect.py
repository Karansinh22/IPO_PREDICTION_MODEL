import pandas as pd
import joblib
import os

BASE = r'd:\Desktop shortcuts\college\sem 6\MINI PROJECT\GITHUB\IPO_PREDICTION_MODEL'

# Fundamental
df_f = pd.read_csv(os.path.join(BASE, 'dataset/cleaned_dataset/cleaned_fundamental_data.csv'))
print("=== FUNDAMENTAL ===")
print("Shape:", df_f.shape)
print("Columns:", df_f.columns.tolist())
print("Target stats:")
print(df_f['Listing Gain'].describe())
print()

# GMP  
df_g = pd.read_csv(os.path.join(BASE, 'dataset/cleaned_dataset/cleaned_gmp_data.csv'))
print("=== GMP ===")
print("Shape:", df_g.shape)
print("Columns:", df_g.columns.tolist())
# find the target column
for c in df_g.columns:
    if 'gain' in c.lower() or 'listing' in c.lower():
        print(f"Target candidate '{c}':", df_g[c].describe())
print()

# Features used by models
f_feat = joblib.load(os.path.join(BASE, 'models/fundamental_features.pkl'))
print("Fund features:", f_feat)

g_feat = joblib.load(os.path.join(BASE, 'models/gmp/features.pkl'))
print("GMP features:", g_feat)
print()

# Current model types
f_reg = joblib.load(os.path.join(BASE, 'models/fundamental_regressor.pkl'))
f_cls = joblib.load(os.path.join(BASE, 'models/fundamental_classifier.pkl'))
g_reg = joblib.load(os.path.join(BASE, 'models/gmp/regressor.pkl'))
g_cls = joblib.load(os.path.join(BASE, 'models/gmp/classifier.pkl'))
print("Fund Regressor:", type(f_reg).__name__, getattr(f_reg, 'get_params', lambda: {})())
print("Fund Classifier:", type(f_cls).__name__, getattr(f_cls, 'get_params', lambda: {})())
print("GMP Regressor:", type(g_reg).__name__, getattr(g_reg, 'get_params', lambda: {})())
print("GMP Classifier:", type(g_cls).__name__, getattr(g_cls, 'get_params', lambda: {})())

# Risk category distribution
f_enc = joblib.load(os.path.join(BASE, 'models/fundamental_label_encoder.pkl'))
g_enc = joblib.load(os.path.join(BASE, 'models/gmp/label_encoder.pkl'))
print("\nFund label encoder classes:", f_enc.classes_)
print("GMP label encoder classes:", g_enc.classes_)

# Check risk dist in fundamental
for col in ['Risk_Category', 'Risk_Category_Encoded']:
    if col in df_f.columns:
        print(f"\nFund {col} dist:\n", df_f[col].value_counts())
for col in ['Risk_Category', 'Risk_Category_Encoded']:
    if col in df_g.columns:
        print(f"\nGMP {col} dist:\n", df_g[col].value_counts())
