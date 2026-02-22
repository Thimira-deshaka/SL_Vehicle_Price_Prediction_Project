import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", page_icon="🚗")

@st.cache_resource
def load_assets():
    model_path = 'models/lgbm_model.pkl'
    encoder_path = 'models/encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders
    else:
        return None, None

model, encoders = load_assets()

st.title("Sri Lanka Used Vehicle Price Predictor")
st.write("Enter vehicle details to get the estimated market value.")

if model is None:
    st.error("Model not found! Please run the training script first (main.py).")
else:
    # Check what classes are available in encoders to populate dataset
    brands = encoders['brand'].classes_ if 'brand' in encoders else ["Toyota", "Suzuki", "Honda", "Mitsubishi", "Nissan"]
    fuel_types = encoders['fuel_type'].classes_ if 'fuel_type' in encoders else ["Petrol", "Hybrid", "Diesel"]
    transmissions = encoders['transmission'].classes_ if 'transmission' in encoders else ["Automatic", "Manual", "Tiptronic"]
    
    # User Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        brand = st.selectbox("Brand", brands)
        model_name = st.text_input("Model (e.g., Alto, Corolla)", "Aqua") # Ideally this should be a selectbox filtered by brand
        year = st.slider("Year of Manufacture", 2000, 2026, 2018)
        fuel = st.selectbox("Fuel Type", fuel_types)
    
    with col2:
        mileage = st.number_input("Mileage (km)", value=50000, step=1000)
        engine = st.number_input("Engine Capacity (cc)", value=1000, step=100)
        transmission = st.selectbox("Transmission", transmissions)
    
    if st.button("Predict Price"):
        # Preprocess features
        try:
            age = 2026 - year
            mileage_per_year = mileage / (age + 1)
            
            # Helper to safely encode
            def safe_encode(encoder, value):
                try:
                    return encoder.transform([str(value)])[0]
                except ValueError:
                    # Handle unseen labels typically by assigning a default or most common
                    # For simplicity, we assign 0 or handling needs to be more robust
                    return 0

            # Construct DataFrame with same feature set as training
            # We need to match the feature list from train.py: 
            # ['year', 'mileage', 'engine_cc', 'vehicle_age', 'mileage_per_year', 
            #  'brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
            
            # Note: User might not input all fields (Location/Condition), so we might need defaults
            # Let's assume defaults for missing fields or train without them if possible. 
            # For this demo, we'll try to provide dummy values for location/condition if not asked
            
            # Defaults
            default_location = encoders['location'].classes_[0] if 'location' in encoders else 'Colombo'
            default_condition = encoders['condition'].classes_[0] if 'condition' in encoders else 'Used'
            
            input_dict = {
                'year': year,
                'mileage': mileage,
                'engine_cc': engine,
                'vehicle_age': age,
                'mileage_per_year': mileage_per_year,
                'brand': safe_encode(encoders['brand'], brand),
                'model': safe_encode(encoders['model'], model_name), # Simple encoding
                'fuel_type': safe_encode(encoders['fuel_type'], fuel),
                'transmission': safe_encode(encoders['transmission'], transmission),
                'location': safe_encode(encoders['location'], default_location),
                'condition': safe_encode(encoders['condition'], default_condition)
            }
            
            input_df = pd.DataFrame([input_dict])
            
            # Predict
            prediction_log = model.predict(input_df)
            final_price = np.expm1(prediction_log)[0]
            
            st.success(f"Estimated Market Value: LKR {final_price:,.2f}")
            st.info("Note: Prices are based on current market trends and model accuracy.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
