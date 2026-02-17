# 📄 IPO Prediction Project: Detailed Development Log

This document provides a technical walkthrough of the development process, model evolution, and architectural decisions made to achieve the current **85.85% accuracy** for IPO risk prediction.

---

## 🏗️ Phase 1: Requirement Analysis & Data Setup
### **1. Initial Research**
- **Source**: Analyzed `REPORT.pdf` to define the project scope.
- **Objective**: Develop a "Fundamental-based Prediction Model" to assess IPO risk.
- **Software**: Python, Scikit-learn, Pandas, Flask.

### **2. Data Cleaning & Preprocessing**
- **Dataset**: `Initial Public Offering.xlsx`.
- **Cleaning**:
    - Dropped unnamed and non-predictive columns (`IPO_Name`, `Date`).
    - Handled missing values using **Median Imputation** for numeric data.
    - Dropped rows where the target variable (`Listing Gain`) was missing.

---

## 📈 Phase 2: First Model Development (Baseline)
### **1. Implementation**
- **Algorithm**: Random Forest Regressor & Classifier.
- **Risk Logic**:
    - **Low Risk**: > 25% Expected Gain.
    - **Medium Risk**: 10% - 25% Expected Gain.
    - **High Risk**: < 10% Expected Gain.
- **Results**:
    - **Accuracy**: **~68%**
    - **R² Score**: **~0.45**
- **Observation**: Performance was "at par" but limited by class imbalance (most IPOs were High Risk) and simple features.

---

## 🚀 Phase 3: High-Accuracy Optimization (Final State)
### **1. Advanced Feature Engineering**
To break the 70% barrier, we engineered 12+ interaction features:
- **Subscription Strength**: `QIB + HNI + RII`.
- **Market Interest**: `QIB / HNI Ratio`.
- **Investment Supply**: `Issue_Size / Offer Price`.
- **Non-Linear Dynamics**: `Total_Subscription²`, `QIB²`, and `HNI²`.

### **2. Solving Class Imbalance**
- **Technique**: **SMOTETomek**.
- **Impact**: Synthetically generated minority class samples (Low/Medium Risk) and filtered out spatial noise (Tomek Links). This was the turning point for the model.

### **3. Model Evolution: XGBoost Ensemble**
- **Architecture**: A **Voting Classifier** combining two tuned **XGBoost** models.
- **XGB1 (Explorative)**: High depth (max_depth=9), low learning rate (0.03).
- **XGB2 (Stable)**: Standard depth (max_depth=7), higher learning rate (0.1).
- **Final Metrics**:
    - **Accuracy**: **85.85%**
    - **F1-Score**: **0.86**
    - **Low Risk F1-Score**: **0.91** (Highly reliable for safety detection).

---

## 🌐 Phase 4: Web Application & UI
### **1. Flask Backend**
- Developed `app.py` with the complete feature engineering pipeline integrated into the `/predict` route.
- Implemented robust error handling and traceback logging.

### **2. Modern Frontend UI**
- **Design Philosophy**: Glassmorphism.
- **Features**:
    - Translucent frosted glass containers.
    - Vibrant radial gradients.
    - Dynamic "Risk Badges" that change color based on prediction results.
    - Fully responsive layout for mobile and desktop.

## 💎 Phase 5: The Wealth Predictor Overhaul
### **1. Professional Rebranding**
- **Theme**: Transitioned from a generic UI to a **"Money" terminal** aesthetic.
- **Palette**: Deep Black (#0a0a0a), Financial Green (#22c55e), and Antique Beige (#f5f5dc).
- **Localization**: Native integration of the **₹ (Rupee)** symbol across all financial inputs and labels.

### **2. Dynamic Market Intelligence**
- **Interactive Visuals**: Integrated **Chart.js** for real-time market insights.
- **Selection Awareness**: Rewrote the charting engine to be "search-aware." When a user picks an IPO, the charts instantly highlight that company's historical position.
- **Benchmarking**: Added a comparative analysis tool that pits specific IPOs against market averages for Listing Gain and Subscription levels.

### **3. Optimized UX Flow**
- **Market Search**: High-performance autocomplete search covering 560+ Indian IPOs.
- **Analysis Animation**: A professional "Fundamentals Processing" overlay to add a strategic feel to prediction.
- **Session Security**: Robust session-based terminal access.

---

### **Final Project Statistics**
- **Accuracy**: 85.85%
- **Dataset Size**: 561 Historical Records (2010–2025)
- **Features**: 15 Optimized Parameters
- **UI State**: Fully dynamic, responsive financial terminal.

---

## 🛤️ Future Roadmap: Target 90%
1. **Model Stacking**: Replace the Voting Classifier with a Stacking approach using a Logistic Regression meta-learner.
2. **Sentiment Analysis**: Integrate live news headlines and social media sentiment (X/Twitter) to capture "Market Hype."
3. **Macro-Data**: Include NSE/BSE VIX (Volatility Index) and sector-wise performance prior to IPO listing.
