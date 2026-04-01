"""
test_binary.py — Prototyping Binary Classification (Profitable vs Loss) to verify
if it reaches the >= 90% accuracy target requested.
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))

def test_binary():
    print("Testing Binary Classification (Gain > 0 vs Gain <= 0)")
    raw = pd.read_excel(os.path.join(BASE, 'dataset', 'raw_dataset', 'Initial Public Offering.xlsx'))
    raw.columns = [c.strip() for c in raw.columns]
    needed = ['Issue_Size(crores)', 'Offer Price', 'QIB', 'HNI', 'RII', 'Total', 'Listing Gain']
    df = raw.dropna(subset=needed).copy()
    
    # BINARY TARGET: Profitable (1) vs Loss (0)
    df['Is_Profitable'] = (df['Listing Gain'] > 0).astype(int)
    print(df['Is_Profitable'].value_counts(normalize=True))
    
    df['Subscription_Strength'] = df['QIB'] + df['HNI'] + df['RII']
    df['Total_Sq'] = df['Total'] ** 2
    df['High_Sub'] = (df['Total'] > 7.5).astype(int)
    df['QIB_Sq'] = df['QIB'] ** 2
    df['HNI_Sq'] = df['HNI'] ** 2
    df['HNI_x_RII'] = df['HNI'] * df['RII']
    df['Log_Issue_Size'] = np.log1p(df['Issue_Size(crores)'].clip(lower=0))
    df['Issue_Per_Price'] = df['Issue_Size(crores)'] / (df['Offer Price'] + 1e-5)
    df['QIB_HNI_Ratio'] = df['QIB'] / (df['HNI'] + 1e-5)
    df['QIB_x_HNI'] = df['QIB'] * df['HNI']
    df['QIB_x_RII'] = df['QIB'] * df['RII']
    df['RII_Sq'] = df['RII'] ** 2
    df['Price_Size_Ratio'] = df['Offer Price'] / (df['Issue_Size(crores)'] + 1e-5)
    df['Sub_Imbalance'] = df['QIB'] - df['RII']
    df['Total_Log'] = np.log1p(df['Total'].clip(lower=0))
    df['QIB_dominance'] = df['QIB'] / (df['Total'] + 1e-5)

    FEATURES = [
        'Subscription_Strength', 'Total_Sq', 'High_Sub', 'Total',
        'QIB_Sq', 'HNI_x_RII', 'HNI', 'HNI_Sq',
        'QIB_x_HNI', 'QIB_x_RII',
        'Issue_Size(crores)', 'Offer Price', 'Log_Issue_Size',
        'Issue_Per_Price', 'QIB_HNI_Ratio',
        'RII_Sq', 'Price_Size_Ratio', 'Sub_Imbalance', 'Total_Log', 'QIB_dominance'
    ]
    
    X = df[FEATURES].values
    y = df['Is_Profitable'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    cls = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(cls, X_scaled, y, cv=cv, scoring='accuracy')
    
    print(f"\nCV Accuracy (Binary): {scores.mean()*100:.2f}%")

if __name__ == '__main__':
    test_binary()
