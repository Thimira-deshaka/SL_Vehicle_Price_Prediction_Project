import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def preprocess_data_for_app(csv_path):
    """
    Preprocess raw CSV data to match the trained model's expected format
    This replicates the preprocessing pipeline used during training
    """
    # Load raw data
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")
    
    # Column mapping from CSV to model format
    column_mapping = {
        'Brand': 'brand',
        'Model': 'model', 
        'YOM': 'year',
        'Engine (cc)': 'engine_cc',
        'Gear': 'transmission',
        'Fuel Type': 'fuel_type',
        'Millage(KM)': 'mileage',
        'Town': 'location',
        'Condition': 'condition',
        'Price': 'price'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Data cleaning and type conversion
    # 1. Clean price data
    if df['price'].dtype == object:
        df['price'] = df['price'].astype(str).str.replace(',', '').str.replace('Rs. ', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Convert price from Lakhs to actual LKR
    if df['price'].median() < 10000:
        print("Converting price from Lakhs to LKR")
        df['price'] = df['price'] * 100000
    
    # 2. Clean year data
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # 3. Clean mileage data
    if df['mileage'].dtype == object:
        df['mileage'] = df['mileage'].astype(str).str.replace(',', '').str.replace(' km', '')
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')
    
    # 4. Clean engine capacity
    df['engine_cc'] = pd.to_numeric(df['engine_cc'], errors='coerce')
    
    # 5. Feature Engineering
    current_year = 2026
    df['vehicle_age'] = current_year - df['year']
    df['vehicle_age'] = df['vehicle_age'].clip(lower=1)  # Avoid division by zero
    df['mileage_per_year'] = df['mileage'] / df['vehicle_age']
    
    # 6. Create log price for model (if price exists)
    df['price_log'] = np.log1p(df['price'])
    
    # 7. Clean categorical data
    categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    
    # 8. Handle missing values
    df = df.dropna(subset=['price', 'year', 'mileage', 'engine_cc'])
    
    # 9. Filter outliers
    # Remove unrealistic prices
    df = df[(df['price'] >= 100000) & (df['price'] <= 50000000)]  # 100K to 50M LKR
    
    # Remove unrealistic years
    df = df[(df['year'] >= 1990) & (df['year'] <= 2026)]
    
    # Remove unrealistic mileage
    df = df[df['mileage'] <= 1000000]  # Max 1M km
    
    print(f"Data shape after cleaning: {df.shape}")
    return df

def encode_categorical_features(df, encoders=None, fit_mode=True):
    """
    Encode categorical features using LabelEncoder
    If fit_mode=True, creates new encoders. If False, uses provided encoders.
    """
    categorical_features = ['brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
    
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    for feature in categorical_features:
        if feature in df.columns:
            if fit_mode:
                # Create new encoder
                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df[feature].astype(str))
                encoders[feature] = le
            else:
                # Use existing encoder with safe handling
                le = encoders.get(feature)
                if le is not None:
                    # Safe encoding for unseen labels
                    known_classes = set(le.classes_)
                    def safe_encode(x):
                        return le.transform([str(x)])[0] if str(x) in known_classes else 0
                    df_encoded[feature] = df[feature].astype(str).apply(safe_encode)
                else:
                    df_encoded[feature] = 0  # Default value if encoder not found
    
    return df_encoded, encoders

def prepare_model_features(df_encoded):
    """
    Prepare features in the exact order expected by the trained model
    """
    feature_columns = ['year', 'mileage', 'engine_cc', 'vehicle_age', 'mileage_per_year', 
                      'brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Default value for missing features
    
    X = df_encoded[feature_columns]
    
    return X

def create_eda_data():
    """
    Create processed data specifically for EDA visualizations
    """
    try:
        raw_df = preprocess_data_for_app('data/sri_lanka_car_price_dataset.csv')
        return raw_df
    except Exception as e:
        print(f"Error creating EDA data: {e}")
        return None

def create_model_data():
    """
    Create processed data for model performance analysis
    """
    try:
        # Load and preprocess data
        df = preprocess_data_for_app('data/sri_lanka_car_price_dataset.csv')
        
        # Load encoders
        encoders = joblib.load('models/encoders.pkl')
        
        # Encode categorical features
        df_encoded, _ = encode_categorical_features(df, encoders, fit_mode=False)
        
        # Prepare features
        X = prepare_model_features(df_encoded)
        y = df_encoded['price_log'] if 'price_log' in df_encoded.columns else np.log1p(df_encoded['price'])
        
        # Take a sample for performance (to avoid memory issues)
        sample_size = min(1000, len(X))
        sample_indices = np.random.choice(len(X), sample_size, replace=False)
        
        return X.iloc[sample_indices], y.iloc[sample_indices], df.iloc[sample_indices]
        
    except Exception as e:
        print(f"Error creating model data: {e}")
        return None, None, None