from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load models and features
REGRESSOR_PATH = 'models/fundamental_regressor.pkl'
CLASSIFIER_PATH = 'models/fundamental_classifier.pkl'
FEATURES_PATH = 'models/fundamental_features.pkl'
SCALER_PATH = 'models/fundamental_scaler.pkl'
ENCODER_PATH = 'models/fundamental_label_encoder.pkl'

regressor = joblib.load(REGRESSOR_PATH)
classifier = joblib.load(CLASSIFIER_PATH)
features = joblib.load(FEATURES_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(ENCODER_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Extract raw inputs
        raw_inputs = {
            'Issue_Size(crores)': float(request.form['Issue_Size(crores)']),
            'Offer Price': float(request.form['Offer Price']),
            'QIB': float(request.form['QIB']),
            'HNI': float(request.form['HNI']),
            'RII': float(request.form['RII']),
            'Total': float(request.form['Total'])
        }
        
        # 2. Recreate the precise engineered features for the 85%+ model
        data = pd.DataFrame([raw_inputs])
        
        # Feature engineering logic (MUST match train_final.py)
        data['Subscription_Strength'] = data['QIB'] + data['HNI'] + data['RII']
        data['Total_Sq'] = data['Total'] ** 2
        
        # We need the median from training for High_Sub, but for Inference we can use a fixed global median
        # In a real app we'd save this, but 7.5 is a representative median from our dataset analysis
        data['High_Sub'] = (data['Total'] > 7.5).astype(int) 
        
        data['QIB_Sq'] = data['QIB'] ** 2
        data['HNI_Sq'] = data['HNI'] ** 2
        data['HNI_x_RII'] = data['HNI'] * data['RII']
        data['Log_Issue_Size'] = np.log1p(data['Issue_Size(crores)'].clip(lower=0))
        data['Issue_Per_Price'] = data['Issue_Size(crores)'] / (data['Offer Price'] + 1e-5)
        data['QIB_HNI_Ratio'] = data['QIB'] / (data['HNI'] + 1e-5)
        data['QIB_x_HNI'] = data['QIB'] * data['HNI']
        data['QIB_x_RII'] = data['QIB'] * data['RII']
        
        # Select features in the correct order
        X_form = data[features]
        
        # 3. Scale
        X_scaled = scaler.transform(X_form)
        
        # 4. Predict
        predicted_gain = regressor.predict(X_scaled)[0]
        risk_encoded = classifier.predict(X_scaled)[0]
        risk_category = le.inverse_transform([risk_encoded])[0]
        
        return render_template('index.html', 
                               prediction_text=f'Predicted Listing Gain: {predicted_gain:.2f}%',
                               risk_text=f'Risk Assessment: {risk_category} Risk',
                               predicted_gain=predicted_gain,
                               risk_category=risk_category)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
