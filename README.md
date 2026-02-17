# 🚀 IPO Prediction Model (Fundamental-Based)

A high-performance Machine Learning platform designed to predict IPO listing gains and assess investment risk using fundamental market parameters.

---

## 📊 Current Model Performance
The model has been optimized using a **Stacking Ensemble of XGBoost models** combined with **SMOTETomek** for class balancing.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | **85.85%** |
| **Precision (Weighted)** | **87.00%** |
| **Recall (Weighted)** | **86.00%** |
| **F1-Score (Weighted)** | **86.00%** |

### **Risk Classification Performance**
- **Low Risk** (Listing Gain > 25%): **91.0% F1-Score**
- **High Risk** (Listing Gain < 10%): **84.0% F1-Score**
- **Medium Risk** (10% - 25% Gain): **83.0% F1-Score**

---

## 🧠 Methodology & Process

### **1. Data Preprocessing & Cleaning**
- Handled missing values using **Median Imputation** for numeric features.
- Cleaned and prepared the raw IPO dataset (`raw_dataset/Initial Public Offering.xlsx`).

### **2. Advanced Feature Engineering**
We achieved the jump from 68% to 85%+ by engineering 10+ interaction features, including:
- **Subscription Strength**: Combinations of QIB, HNI, and RII subscription ratios.
- **Supply-Demand Ratios**: `Issue_Size` relative to `Offer Price`.
- **Logarithmic Scaling**: Normalized skewed features like `Issue_Size`.
- **Non-Linear Terms**: Squared and interaction terms for subscription data.

### **3. Handling Class Imbalance**
The dataset was heavily skewed toward "High Risk" IPOs. We resolved this using **SMOTETomek** (Synthetic Minority Over-sampling Technique combined with Tomek Links), which synthetically balanced the classes while removing redundant noise.

### **4. Ensemble Learning Architecture**
The final model is a **Voting Ensemble** of multiple tuned **XGBoost** classifiers.
- **Model A**: Focused on high-depth explorative patterns.
- **Model B**: Focused on stable, shallow generalization.

---

## 🛠️ Technical Stack & Features

### **The "Wealth Predictor" Overhaul**
The application has been upgraded into a professional-grade **Money Theme** terminal with:
- **Visual Identity**: High-end financial aesthetic using **Green, Beige, and Black**.
- **Rupee Integration**: Extensive use of the **₹** symbol for localization.
- **Dynamic Marketplace Dashboard**: 
  - **Yearly Success Trend**: Line chart tracking IPO performance over 15 years.
  - **Market Benchmark**: Real-time comparison bar chart between selected IPO and market averages.
  - **Subscription Analytics**: Scatter plots showing correlations between hype and gain.

### **Core Technologies**
- **Backend**: Python, Flask (with session-based authentication)
- **Machine Learning**: Scikit-Learn, XGBoost, SMOTETomek
- **Data Visualization**: Chart.js (Interactive & Responsive)
- **Frontend**: Glassmorphism UI with Vanilla CSS
- **Dataset**: 560+ Historical Indian IPOs from 2010 onwards

---

## 🛤️ Roadmap & Achievements
- [x] **85.85% Accuracy** achieved via XGBoost Ensemble.
- [x] **SMOTETomek** integration for handling class imbalance.
- [x] **Market-Wide Search**: Instant auto-fill for 561 Indian IPOs.
- [x] **Persistent Sessions**: User history and state tracking.
- [x] **Interactive Dashboard**: Selection-aware visualizations.

---

## 🚀 Getting Started

### **Prerequisites**
```bash
pip install flask pandas scikit-learn joblib xgboost imbalanced-learn openpyxl
```

### **Run the Application**
```bash
python app.py
```
Visit `http://127.0.0.1:5000` to interact with the dashboard.