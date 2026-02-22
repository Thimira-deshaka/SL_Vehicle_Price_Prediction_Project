import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data(df):
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # 1. Feature Engineering
    current_year = 2026 # As per instructions
    # Check if 'year' column exists, if not try to find it
    if 'Manufacturer Year' in df.columns:
         df['year'] = df['Manufacturer Year']
    elif 'Year of Manufacture' in df.columns:
         df['year'] = df['Year of Manufacture']
    elif 'YOM' in df.columns:
         df['year'] = df['YOM']
         
    # Ensure year is numeric
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    df['vehicle_age'] = current_year - df['year']
    
    # Handle mileage
    if 'Mileage' in df.columns:
        df['mileage'] = pd.to_numeric(df['Mileage'].astype(str).str.replace(',', '').str.replace(' km', ''), errors='coerce')
    elif 'Millage(KM)' in df.columns:
        df['mileage'] = pd.to_numeric(df['Millage(KM)'], errors='coerce')

    df['mileage_per_year'] = df['mileage'] / (df['vehicle_age'] + 1)
    
    # 2. Handling Outliers (Prices in LKR)
    # Ensure price is numeric
    if 'Price' in df.columns:
        # Check if price is string or numeric first
        if df['Price'].dtype == object:
             df['price_lkr'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', '').str.replace('Rs. ', ''), errors='coerce')
        else:
             df['price_lkr'] = pd.to_numeric(df['Price'], errors='coerce')
        
        # Heuristic to detect if price is in Lakhs (e.g. 100.0 instead of 10,000,000)
        # If the median price is very small (e.g. < 1000), it's likely in Lakhs or Millions
        if df['price_lkr'].median() < 10000:
             # Assuming it's in Lakhs (100,000)
             print("Detected price in Lakhs. Multiplying by 100,000.")
             df['price_lkr'] = df['price_lkr'] * 100000

    # Filter out unrealistic prices and missing values
    # Remove rows where price, year, or mileage is NaN
    df = df.dropna(subset=['price_lkr', 'year', 'mileage'])
    
    # Filter reasonable price range (e.g., 5 Lakhs to 1000 Lakhs / 100 Million)
    df = df[(df['price_lkr'] > 500000) & (df['price_lkr'] < 150000000)]
    
    # 3. Log Transformation for Price (Target)
    df['price_log'] = np.log1p(df['price_lkr'])
    
    # 4. Encoding Categorical Variables
    # Map column names if they differ in the csv
    column_mapping = {
        'Brand': 'brand',
        'Model': 'model', 
        'Fuel Type': 'fuel_type',
        'Transmission': 'transmission',
        'Location': 'location',
        'Condition': 'condition',
        'Capacity': 'engine_cc',
        'Town': 'location',
        'Gear': 'transmission',
        'Engine (cc)': 'engine_cc'
    }
    df = df.rename(columns=column_mapping)
    
    cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
    encoders = {}
    
    # Ensure directory exists for saving encoders
    os.makedirs('models', exist_ok=True)

    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            # Convert to string to handle mixed types
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    # Save encoders
    joblib.dump(encoders, 'models/encoders.pkl')
        
    return df, encoders
