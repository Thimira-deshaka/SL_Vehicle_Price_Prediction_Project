import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", page_icon="🚗")

# Custom CSS for attractive UI
st.markdown("""
    <style>
    /* Background image with gradient overlay */
    .stApp {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.45) 0%, rgba(118, 75, 162, 0.75) 50%, rgba(0, 0, 0, 0.6) 100%),
                    url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Main container styling - semi-transparent white */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.65);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        max-width: 1000px;
        margin: auto;
    }
    
    /* Title styling */
    h1 {
        color: #1e3a8a;
        text-align: center;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Input fields */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #1e293b !important;
        border: 2px solid rgba(100, 116, 139, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Number input container background */
    .stNumberInput > div {
        background-color: white !important;
        border-radius: 8px !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.5);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(102, 126, 234, 0.7);
    }
    
    /* Column styling */
    div[data-testid="column"] {
        background-color: rgba(255, 255, 255, 0.7);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Label styling - WHITE */
    label {
        font-weight: 600;
        color: white !important;
        font-size: 0.95rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

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
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2.5rem; 
                            border-radius: 18px; 
                            text-align: center;
                            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.6);
                            border: 2px solid rgba(255, 255, 255, 0.2);
                            margin-top: 1rem;'>
                    <div style='background: rgba(255, 255, 255, 0.15); 
                                padding: 0.5rem 1.5rem; 
                                border-radius: 50px; 
                                display: inline-block; 
                                margin-bottom: 1rem;'>
                        <span style='color: white; 
                                     font-size: 0.95rem; 
                                     font-weight: 600; 
                                     letter-spacing: 1.5px;'>
                            💰 ESTIMATED MARKET VALUE
                        </span>
                    </div>
                    <h1 style='color: #fcd34d; 
                               margin: 1rem 0; 
                               font-size: 3rem; 
                               font-weight: 800; 
                               text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                        LKR {final_price:,.2f}
                    </h1>
                    <div style='height: 2px; 
                                width: 80px; 
                                background: rgba(255, 255, 255, 0.5); 
                                margin: 1rem auto;'>
                    </div>
                    <p style='color: #e0e7ff; 
                              margin: 0; 
                              font-size: 0.95rem; 
                              font-weight: 400;'>
                        📊 Based on current market trends and ML analysis
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
