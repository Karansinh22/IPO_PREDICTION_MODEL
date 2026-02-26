import pandas as pd
import json
import os
from datetime import datetime

# Paths
EXCEL_PATH = 'dataset/raw_dataset/Initial Public Offering.xlsx'
HISTORY_JSON = 'data/market_history.json'

def expand_history():
    print("Loading existing history...")
    try:
        with open(HISTORY_JSON, 'r') as f:
            existing_data = json.load(f)
    except Exception as e:
        print(f"Error loading {HISTORY_JSON}: {e}")
        existing_data = []

    existing_names = {item['name'] for item in existing_data}

    print(f"Reading {EXCEL_PATH}...")
    df = pd.read_excel(EXCEL_PATH)
    
    # Filter for past 5 years (from 2021 onwards)
    df['Date'] = pd.to_datetime(df['Date'])
    df_filtered = df[df['Date'] >= '2021-01-01'].copy()
    
    print(f"Found {len(df_filtered)} records from 2021 onwards.")

    new_records = []
    for _, row in df_filtered.iterrows():
        name = str(row['IPO_Name']).strip()
        if name in existing_names:
            continue
            
        # Map columns
        record = {
            "name": name,
            "full_name": name, # Excel doesn't distinguish short/full name
            "status": "Closed",
            "offer_price": float(row['Offer Price']) if pd.notnull(row['Offer Price']) else 0,
            "qib": float(row['QIB']) if pd.notnull(row['QIB']) else 0,
            "rii": float(row['RII']) if pd.notnull(row['RII']) else 0,
            "total": float(row['Total']) if pd.notnull(row['Total']) else 0,
            "gmp": 0, # Not in Excel
            "size": float(row['Issue_Size(crores)']) if pd.notnull(row['Issue_Size(crores)']) else 0,
            "actual_gain": float(row['Listing Gain']) if pd.notnull(row['Listing Gain']) else 0,
            "listing_date": row['Date'].strftime('%Y-%m-%d') if pd.notnull(row['Date']) else "Unknown"
        }
        new_records.append(record)
        existing_names.add(name)

    # Combine
    combined = existing_data + new_records
    
    # Sort by date descending
    combined.sort(key=lambda x: x.get('listing_date', '0000-00-00'), reverse=True)

    print(f"Writing {len(combined)} total records to {HISTORY_JSON}...")
    with open(HISTORY_JSON, 'w') as f:
        json.dump(combined, f, indent=2)
    print("Success!")

if __name__ == "__main__":
    expand_history()
