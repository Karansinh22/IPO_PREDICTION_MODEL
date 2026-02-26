from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import os
import uuid
from scraper import get_indian_market_ipos

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ipo_predictor_very_secret_key_123')

# Load Fundamental models
F_REGRESSOR = joblib.load('models/fundamental_regressor.pkl')
F_CLASSIFIER = joblib.load('models/fundamental_classifier.pkl')
F_FEATURES = joblib.load('models/fundamental_features.pkl')
F_SCALER = joblib.load('models/fundamental_scaler.pkl')
F_ENCODER = joblib.load('models/fundamental_label_encoder.pkl')

# Load GMP models
G_REGRESSOR = joblib.load('models/gmp/regressor.pkl')
G_CLASSIFIER = joblib.load('models/gmp/classifier.pkl')
G_FEATURES = joblib.load('models/gmp/features.pkl')
G_SCALER = joblib.load('models/gmp/scaler.pkl')
G_ENCODER = joblib.load('models/gmp/label_encoder.pkl')

# Global data for searching
ipo_data = pd.DataFrame()

@app.route('/api/market_ipos')
def get_market_ipos():
    try:
        data = get_indian_market_ipos()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_ipo_data():
    global ipo_data
    try:
        ipo_path = 'dataset/raw_dataset/Initial Public Offering.xlsx'
        if os.path.exists(ipo_path):
            ipo_data = pd.read_excel(ipo_path)
            ipo_data.columns = [c.strip() for c in ipo_data.columns]
            print(f"Loaded {len(ipo_data)} IPO records.")
        else:
            print("Warning: IPO dataset file not found.")
    except Exception as e:
        print(f"Error loading IPO dataset: {e}")

# Initial load
load_ipo_data()

@app.route('/')
def index():
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('index.html', history=session.get('history', []))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    if username:
        session['user_id'] = str(uuid.uuid4())
        session['username'] = username
        session['history'] = []
        return redirect(url_for('home'))
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/search')
def search():
    global ipo_data
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    if ipo_data.empty:
        load_ipo_data()

    if ipo_data.empty:
        return jsonify([{'name': 'Error: Data not loaded'}])

    # Search on IPO_Name
    results = ipo_data[ipo_data['IPO_Name'].str.lower().str.contains(query, na=False)]
    
    search_results = []
    for _, row in results.head(10).iterrows():
        try:
            # Handle Date to extract Year
            row_date = pd.to_datetime(row['Date'], errors='coerce')
            year = row_date.year if pd.notnull(row_date) else None

            search_results.append({
                'name': str(row['IPO_Name']),
                'issue_size': float(row.get('Issue_Size(crores)', 0)) if pd.notnull(row.get('Issue_Size(crores)')) else 0,
                'offer_price': float(row.get('Offer Price', 0)) if pd.notnull(row.get('Offer Price')) else 0,
                'qib': float(row.get('QIB', 0)) if pd.notnull(row.get('QIB')) else 0,
                'hni': float(row.get('HNI', 0)) if pd.notnull(row.get('HNI')) else 0,
                'rii': float(row.get('RII', 0)) if pd.notnull(row.get('RII')) else 0,
                'total': float(row.get('Total', 0)) if pd.notnull(row.get('Total')) else 0,
                'gain': float(row.get('Listing Gain', 0)) if pd.notnull(row.get('Listing Gain')) else 0,
                'year': year
            })
        except Exception:
            continue
            
    return jsonify(search_results)

@app.route('/get_market_insights')
def get_market_insights():
    global ipo_data
    if ipo_data.empty:
        load_ipo_data()
    
    if ipo_data.empty:
        return jsonify({})

    df = ipo_data.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year

    # 1. Yearly Listing Gain Trend
    yearly_trend = df.groupby('Year')['Listing Gain'].mean().fillna(0).to_dict()

    # 2. Subscription vs Listing Gain (Correlation)
    sub_gain_data = df[['Total', 'Listing Gain']].dropna()
    correlation_data = sub_gain_data.head(100).to_dict(orient='records') 

    # 3. Risk Distribution
    def simple_risk(gain):
        if gain > 25: return 'Low'
        elif gain >= 10: return 'Medium'
        else: return 'High'
    
    df['Risk'] = df['Listing Gain'].apply(simple_risk)
    risk_dist = df['Risk'].value_counts().to_dict()

    # 4. Market Averages (for benchmarking)
    market_averages = {
        'avg_gain': float(df['Listing Gain'].mean()),
        'avg_sub': float(df['Total'].mean()),
        'avg_size': float(df['Issue_Size(crores)'].mean())
    }

    return jsonify({
        'yearly_trend': yearly_trend,
        'correlation': correlation_data,
        'risk_dist': risk_dist,
        'averages': market_averages
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form.get('model_type', 'fundamental')
        
        if model_type == 'fundamental':
            raw_inputs = {
                'Issue_Size(crores)': float(request.form['Issue_Size(crores)']),
                'Offer Price': float(request.form['Offer Price']),
                'QIB': float(request.form['QIB']),
                'HNI': float(request.form['HNI']),
                'RII': float(request.form['RII']),
                'Total': float(request.form['Total'])
            }
            actual_gain = request.form.get('actual_gain')
            data = pd.DataFrame([raw_inputs])
            data['Subscription_Strength'] = data['QIB'] + data['HNI'] + data['RII']
            data['Total_Sq'] = data['Total'] ** 2
            data['High_Sub'] = (data['Total'] > 7.5).astype(int) 
            data['QIB_Sq'] = data['QIB'] ** 2
            data['HNI_Sq'] = data['HNI'] ** 2
            data['HNI_x_RII'] = data['HNI'] * data['RII']
            data['Log_Issue_Size'] = np.log1p(data['Issue_Size(crores)'].clip(lower=0))
            data['Issue_Per_Price'] = data['Issue_Size(crores)'] / (data['Offer Price'] + 1e-5)
            data['QIB_HNI_Ratio'] = data['QIB'] / (data['HNI'] + 1e-5)
            data['QIB_x_HNI'] = data['QIB'] * data['HNI']
            data['QIB_x_RII'] = data['QIB'] * data['RII']
            
            X_scaled = F_SCALER.transform(data[F_FEATURES])
            predicted_gain = F_REGRESSOR.predict(X_scaled)[0]
            risk_encoded = F_CLASSIFIER.predict(X_scaled)[0]
            risk_category = F_ENCODER.inverse_transform([risk_encoded])[0]
            
        else:
            # GMP logic
            raw_inputs = {
                'IPO_Size': float(request.form['IPO_Size']),
                'Subscription': float(request.form['Subscription']),
                'GMP': float(request.form['GMP']),
                'IPO Price': float(request.form['IPO Price']),
                'Listing_Year': pd.Timestamp.now().year,
                'Listing_Month': pd.Timestamp.now().month
            }
            data = pd.DataFrame([raw_inputs])
            data['GMP_to_Price_Ratio'] = data['GMP'] / (data['IPO Price'] + 1e-5)
            data['GMP_Subscription_Impact'] = data['GMP'] * data['Subscription']
            data['Log_IPO_Size'] = np.log1p(data['IPO_Size'].clip(lower=0))
            data['Subscription_Sq'] = data['Subscription'] ** 2
            data['GMP_Sq'] = data['GMP'] ** 2
            
            X_scaled = G_SCALER.transform(data[G_FEATURES])
            predicted_gain = G_REGRESSOR.predict(X_scaled)[0]
            risk_encoded = G_CLASSIFIER.predict(X_scaled)[0]
            risk_category = G_ENCODER.inverse_transform([risk_encoded])[0]

        # Save to History
        history_item = {
            'ipo_name': request.form.get('ipo_name', 'Manual Entry') or 'Manual Entry',
            'gain': f"{predicted_gain:.2f}%",
            'risk': risk_category,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': model_type,
            'inputs': raw_inputs
        }
        history = session.get('history', [])
        history.insert(0, history_item)
        session['history'] = history[:10]
        session.modified = True

        # Comparison Logic
        comparison_info = None
        if model_type == 'fundamental' and 'actual_gain' in locals() and actual_gain not in [None, '']:
            try:
                actual_val = float(actual_gain)
                diff = abs(actual_val - predicted_gain)
                comparison_info = {
                    "actual": actual_val,
                    "predicted": round(predicted_gain, 2),
                    "error": round(diff, 2)
                }
            except: pass

        return render_template('index.html', 
                               prediction_text=f'Predicted Listing Gain: {predicted_gain:.2f}%',
                               risk_text=f'Risk Assessment: {risk_category} Risk',
                               predicted_gain=predicted_gain,
                               risk_category=risk_category,
                               comparison_info=comparison_info,
                               model_type=model_type,
                               history=session['history'],
                               selected_ipo_data={
                                   'name': history_item['ipo_name'],
                                   'gain': predicted_gain,
                                   'total': float(request.form.get('Total', request.form.get('Subscription', 0))),
                                   'issue_size': float(request.form.get('Issue_Size(crores)', request.form.get('IPO_Size', 0)))
                               })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error_text=f"Error: {str(e)}", history=session.get('history', []))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
