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

## 🛠️ Technical Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, XGBoost, Imbalanced-Learn
- **Frontend**: HTML5, CSS3 (Glassmorphism UI)
- **Serialization**: Joblib

---

## 🛤️ Roadmap to 90% Accuracy
To push the performance from the current 86% to 90%, the following steps are planned:
1. **Sentiment Integration**: Scrape and analyze real-time market sentiment from financial news and Twitter/X APIs.
2. **Economic Indicators**: Incorporate macro-economic data (interest rates, GDP growth, market volatility/VIX).
3. **Advanced Stacking**: Implement a meta-learner (e.g., Logistic Regression or LightGBM) to intelligently weigh predictions from multiple base models.
4. **Time-Series Analysis**: Train on larger historical datasets to identify cyclical market trends (Bull/Bear cycle detection).

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