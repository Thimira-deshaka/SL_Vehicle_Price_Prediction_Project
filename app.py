import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our custom data processor
import sys
sys.path.append('src')
from data_processor import preprocess_data_for_app, create_eda_data, create_model_data

# Set page config
st.set_page_config(page_title="Sri Lanka Vehicle Analytics", page_icon="🚗", layout="wide")

# Custom CSS for attractive UI
st.markdown("""
    <style>
    /* Background styling */
    .stApp {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.45) 0%, rgba(118, 75, 162, 0.75) 50%, rgba(0, 0, 0, 0.6) 100%),
                    url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Main container styling */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.65);
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.4);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        max-width: 1400px;
        margin: auto;
    }
    
    /* Title styling */
    h1 {
        color: #e6eaf1;
        text-align: center;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 0.5rem;
    }
    
    /* Input field styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: white !important;
        color: #1e293b !important;
        border: 2px solid rgba(100, 116, 139, 0.3) !important;
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
    
    /* Label styling */
    label {
        font-weight: 600;
        color: white !important;
        font-size: 0.95rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #e0e7ff;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    """Load trained model and encoders"""
    model_path = 'models/lgbm_model.pkl'
    encoder_path = 'models/encoders.pkl'
    
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders
    else:
        return None, None

@st.cache_data
def load_eda_data():
    """Load and cache EDA data"""
    return create_eda_data()

@st.cache_data
def load_performance_data():
    """Load and cache model performance data"""
    return create_model_data()

def prediction_tab(model, encoders):
    """Vehicle price prediction interface"""
    st.subheader("🔮 Vehicle Price Prediction")
    st.write("Enter vehicle details to get an accurate market value prediction")
    
    # Get available options from encoders
    brands = sorted(encoders['brand'].classes_) if 'brand' in encoders else ["TOYOTA", "SUZUKI", "HONDA"]
    fuel_types = sorted(encoders['fuel_type'].classes_) if 'fuel_type' in encoders else ["PETROL", "HYBRID", "DIESEL"]
    transmissions = sorted(encoders['transmission'].classes_) if 'transmission' in encoders else ["AUTOMATIC", "MANUAL"]
    locations = sorted(encoders['location'].classes_) if 'location' in encoders else ["COLOMBO", "GAMPAHA", "KANDY"]
    conditions = sorted(encoders['condition'].classes_) if 'condition' in encoders else ["USED", "RECONDITIONED"]
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand = st.selectbox("Brand", brands)
        model_name = st.text_input("Model", "R8")
        year = st.slider("Year of Manufacture", 2000, 2026, 2018)
        fuel = st.selectbox("Fuel Type", fuel_types)
    
    with col2:
        mileage = st.number_input("Mileage (km)", value=50000, step=1000, min_value=0)
        engine = st.number_input("Engine Capacity (cc)", value=2000, step=50, min_value=600, max_value=5000)
        transmission = st.selectbox("Transmission", transmissions)
    
    with col3:
        location = st.selectbox("Location", locations)
        condition = st.selectbox("Condition", conditions)
        st.write("")  # Spacing
        predict_button = st.button("🔍 Predict Price", use_container_width=True)
    
    if predict_button:
        try:
            # Feature engineering
            age = 2026 - year
            mileage_per_year = mileage / (age + 1) if age > 0 else mileage
            
            # Safe encoding function
            def safe_encode(encoder, value):
                try:
                    return encoder.transform([str(value).upper()])[0]
                except (ValueError, AttributeError):
                    return 0
            
            # Create input features
            input_features = {
                'year': year,
                'mileage': mileage,
                'engine_cc': engine,
                'vehicle_age': age,
                'mileage_per_year': mileage_per_year,
                'brand': safe_encode(encoders['brand'], brand),
                'model': safe_encode(encoders['model'], model_name),
                'fuel_type': safe_encode(encoders['fuel_type'], fuel),
                'transmission': safe_encode(encoders['transmission'], transmission),
                'location': safe_encode(encoders['location'], location),
                'condition': safe_encode(encoders['condition'], condition)
            }
            
            input_df = pd.DataFrame([input_features])
            
            # Make prediction
            prediction_log = model.predict(input_df)
            final_price = np.expm1(prediction_log)[0]
            
            # Display result
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
                            💰 PREDICTED MARKET VALUE
                        </span>
                    </div>
                    <h1 style='color: #fcd34d; 
                               margin: 1rem 0; 
                               font-size: 3rem; 
                               font-weight: 800; 
                               text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                        LKR {final_price:,.0f}
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
                        📊 Based on {len(input_df)} features and ML analysis
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # SHAP Explanation for this specific prediction
            st.markdown("---")
            st.subheader("🧠 Why This Price? (AI Explanation)")
            st.info("🔍 **Understanding the prediction**: See exactly how each feature of your vehicle influences the predicted price.")
            
            try:
                # Generate SHAP explanation for the user's input
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 📊 Feature Impact Analysis")
                    
                    # Create feature contribution dataframe
                    feature_contributions = pd.DataFrame({
                        'Feature': input_df.columns,
                        'Your_Value': input_df.iloc[0].values,
                        'SHAP_Impact': shap_values[0],
                    }).sort_values('SHAP_Impact', key=abs, ascending=False)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        feature_contributions, 
                        x='SHAP_Impact', 
                        y='Feature',
                        orientation='h',
                        title="How Each Feature Affects Your Vehicle's Price",
                        color='SHAP_Impact',
                        color_continuous_scale='RdBu_r',
                        hover_data=['Your_Value']
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Explanation guide
                    st.info("📈 **How to read**: Red bars increase price, blue bars decrease price. Longer bars means, its a stronger impact.")
                
                with col2:
                    st.markdown("#### 📋 Detailed Breakdown")
                    
                    # Add user-friendly feature names
                    feature_names_map = {
                        'year': 'Manufacturing Year',
                        'mileage': 'Mileage (km)',
                        'engine_cc': 'Engine Size (cc)',
                        'vehicle_age': 'Vehicle Age (years)',
                        'mileage_per_year': 'Usage Rate (km/year)',
                        'brand': 'Brand',
                        'model': 'Model',
                        'fuel_type': 'Fuel Type',
                        'transmission': 'Transmission',
                        'location': 'Location',
                        'condition': 'Condition'
                    }
                    
                    # Enhanced display with interpretations
                    display_data = feature_contributions.copy()
                    display_data['Feature_Name'] = display_data['Feature'].map(feature_names_map).fillna(display_data['Feature'])
                    display_data['Impact_Direction'] = display_data['SHAP_Impact'].apply(
                        lambda x: "🔴 Increases Price" if x > 0 else "🔵 Decreases Price" if x < 0 else "➖ Neutral"
                    )
                    display_data['Impact_Strength'] = display_data['SHAP_Impact'].apply(
                        lambda x: "🔥 Strong" if abs(x) > 0.1 else "🔸 Moderate" if abs(x) > 0.05 else "💧 Weak"
                    )
                    
                    # Show the table
                    final_display = display_data[['Feature_Name', 'Your_Value', 'SHAP_Impact', 'Impact_Direction', 'Impact_Strength']]
                    st.dataframe(final_display, use_container_width=True, height=350)
                    
                    # Key insights
                    st.markdown("**🎯 Key Insights for Your Vehicle:**")
                    
                    top_positive = display_data[display_data['SHAP_Impact'] > 0].head(1)
                    top_negative = display_data[display_data['SHAP_Impact'] < 0].head(1)
                    
                    if not top_positive.empty:
                        feature = top_positive.iloc[0]['Feature_Name']
                        value = top_positive.iloc[0]['Your_Value']
                        impact = top_positive.iloc[0]['SHAP_Impact']
                        st.success(f"✅ **Best feature**: {feature} ({value}) adds +{impact:.3f} to price")
                    
                    if not top_negative.empty:
                        feature = top_negative.iloc[0]['Feature_Name']
                        value = top_negative.iloc[0]['Your_Value']
                        impact = abs(top_negative.iloc[0]['SHAP_Impact'])
                        st.warning(f"⚠️ **Price reducer**: {feature} ({value}) reduces price by -{impact:.3f}")
                    
                    # Overall explanation
                    base_value = explainer.expected_value
                    total_shap_impact = shap_values[0].sum()
                    st.info(f"📊 **Base price**: {np.expm1(base_value):,.0f} LKR → **Your price**: {final_price:,.0f} LKR")
            
            except Exception as shap_error:
                st.warning("⚠️ Advanced AI explanation temporarily unavailable")
                st.info("💡 The prediction is working perfectly! The detailed explanation feature will be available shortly.")
                
                # Basic feature summary as fallback
                st.subheader("📋 Your Vehicle Summary")
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.write(f"**🏷️ Brand & Model**: {brand} {model_name}")
                    st.write(f"**📅 Year**: {year} ({age} years old)")
                    st.write(f"**⛽ Fuel & Transmission**: {fuel}, {transmission}")
                
                with summary_col2:
                    st.write(f"**🚗 Engine**: {engine:,} cc")
                    st.write(f"**📍 Location**: {location}")
                    st.write(f"**🔧 Condition**: {condition}")
                    st.write(f"**📊 Usage**: {mileage:,} km ({mileage_per_year:,.0f} km/year)")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")


def eda_overview_tab():
    """EDA Overview with key statistics"""
    st.subheader("📊 Market Overview & Statistics")
    
    df = load_eda_data()
    if df is None:
        st.error("Unable to load data for analysis")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Average Price", f"LKR {df['price'].mean():,.0f}")
    with col3:
        st.metric("Price Range", f"LKR {df['price'].max() - df['price'].min():,.0f}")
    with col4:
        st.metric("Brands Available", df['brand'].nunique())
    
    st.markdown("---")
    
    # Price distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Distribution")
        fig = px.histogram(df, x='price', nbins=50, title="Vehicle Price Distribution")
        fig.update_layout(xaxis_title="Price (LKR)", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Price vs Year")
        fig = px.scatter(df.sample(min(1000, len(df))), x='year', y='price', 
                        title="Price vs Manufacturing Year", opacity=0.6)
        fig.update_layout(xaxis_title="Year", yaxis_title="Price (LKR)")
        st.plotly_chart(fig, use_container_width=True)


def eda_brand_analysis_tab():
    """Brand and model analysis"""
    st.subheader("🏢 Brand & Model Analysis")
    
    df = load_eda_data()
    if df is None:
        st.error("Unable to load data for analysis")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Brands by Average Price")
        brand_avg = df.groupby('brand')['price'].agg(['mean', 'count']).reset_index()
        brand_avg = brand_avg[brand_avg['count'] >= 10].sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(brand_avg, x='mean', y='brand', orientation='h',
                    title="Average Price by Brand")
        fig.update_layout(xaxis_title="Average Price (LKR)", yaxis_title="Brand")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Models by Average Price")
        model_avg = df.groupby('model')['price'].agg(['mean', 'count']).reset_index()
        model_avg = model_avg[model_avg['count'] >= 5].sort_values('mean', ascending=False).head(10)
        
        fig = px.bar(model_avg, x='mean', y='model', orientation='h',
                    title="Average Price by Model")
        fig.update_layout(xaxis_title="Average Price (LKR)", yaxis_title="Model")
        st.plotly_chart(fig, use_container_width=True)
    
    # Market share analysis
    st.subheader("Market Share Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_counts = df['brand'].value_counts().head(10)
        fig = px.pie(values=brand_counts.values, names=brand_counts.index,
                    title="Market Share by Brand (Top 10)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fuel_counts = df['fuel_type'].value_counts()
        fig = px.pie(values=fuel_counts.values, names=fuel_counts.index,
                    title="Market Share by Fuel Type")
        st.plotly_chart(fig, use_container_width=True)


def eda_feature_analysis_tab():
    """Feature correlation and analysis"""
    st.subheader("🔗 Feature Analysis & Correlations")
    
    df = load_eda_data()
    if df is None:
        st.error("Unable to load data for analysis")
        return
    
    # Correlation matrix
    st.subheader("Feature Correlation Matrix")
    
    # Select numeric columns for correlation
    numeric_cols = ['price', 'year', 'mileage', 'engine_cc', 'vehicle_age', 'mileage_per_year']
    corr_data = df[numeric_cols].corr()
    
    fig = px.imshow(corr_data, text_auto=True, aspect="auto",
                   title="Feature Correlation Heatmap", color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Engine Capacity vs Price")
        fig = px.scatter(df.sample(min(1000, len(df))), x='engine_cc', y='price',
                        title="Engine Capacity vs Price", opacity=0.6)
        fig.update_layout(xaxis_title="Engine Capacity (cc)", yaxis_title="Price (LKR)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Mileage vs Price")
        fig = px.scatter(df.sample(min(1000, len(df))), x='mileage', y='price',
                        title="Mileage vs Price", opacity=0.6)
        fig.update_layout(xaxis_title="Mileage (km)", yaxis_title="Price (LKR)")
        st.plotly_chart(fig, use_container_width=True)


def model_performance_tab(model):
    """Model performance analysis with proper data"""
    st.subheader("📈 Model Performance Analysis")
    
    X, y, df_sample = load_performance_data()
    
    if X is None or y is None:
        st.error("Unable to load model performance data. Please ensure the model is trained and data files exist.")
        return
    
    try:
        # Make predictions
        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y)
        
        # Calculate metrics
        r2 = r2_score(y, y_pred_log)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        # Display metrics
        st.subheader("🎯 Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination")
        with col2:
            st.metric("MAE", f"LKR {mae:,.0f}", help="Mean Absolute Error")
        with col3:
            st.metric("RMSE", f"LKR {rmse:,.0f}", help="Root Mean Squared Error")
        with col4:
            st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
        
        # Visualizations
        st.subheader("📊 Performance Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Actual vs Predicted
            fig = px.scatter(x=y_actual, y=y_pred, 
                           labels={'x': 'Actual Price (LKR)', 'y': 'Predicted Price (LKR)'},
                           title="Actual vs Predicted Prices")
            
            # Perfect prediction line
            min_val, max_val = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
            fig.add_shape(type="line", line=dict(dash='dash', color='red'),
                         x0=min_val, y0=min_val, x1=max_val, y1=max_val)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Residuals plot
            residuals = y_actual - y_pred
            fig = px.scatter(x=y_pred, y=residuals,
                           labels={'x': 'Predicted Price (LKR)', 'y': 'Residuals (LKR)'},
                           title="Residuals vs Predicted")
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        st.subheader("🔢 Feature Importance")
        feature_names = X.columns
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="LightGBM Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in performance analysis: {e}")


def shap_explainability_tab(model):
    """SHAP explainability analysis with proper data"""
    st.subheader("🧠 SHAP Model Explainability")
    
    X, y, df_sample = load_performance_data()
    
    if X is None:
        st.error("Unable to load data for SHAP analysis")
        return
    
    try:
        # Limit sample size for SHAP (computationally expensive)
        sample_size = min(100, len(X))
        X_shap = X.head(sample_size)
        
        # Create SHAP explainer
        with st.spinner("Calculating SHAP values... This may take a moment."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
        
        # SHAP Summary Plot (Feature Importance)
        st.subheader("📊 Global Feature Importance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**SHAP Feature Importance (Bar Plot)**")
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance (Bar Plot)")
            st.pyplot(plt.gcf())
            plt.close()
        
        with col2:
            st.write("**SHAP Summary Plot (Beeswarm)**")
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_values, X_shap, show=False)
            plt.title("SHAP Summary Plot (Beeswarm)")
            st.pyplot(plt.gcf())
            plt.close()
        
        # Feature importance values
        st.subheader("🔢 SHAP Importance Rankings")
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'Feature': X_shap.columns,
            'Mean_SHAP_Value': mean_shap_values
        }).sort_values('Mean_SHAP_Value', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.dataframe(shap_df, use_container_width=True)
        
        with col2:
            fig = px.bar(shap_df.head(10), x='Mean_SHAP_Value', y='Feature',
                        orientation='h', title="Top 10 Features by SHAP Importance")
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in SHAP analysis: {e}")
        st.info("💡 **Troubleshooting Tips:**")
        st.write("• Try upgrading SHAP: `pip install shap --upgrade`")
        st.write("• Or install a specific version: `pip install shap==0.41.0`")
        st.write("• Check if model and data are properly loaded")
        
        # Provide basic feature importance as fallback
        st.subheader("🔄 Fallback: LightGBM Feature Importance")
        try:
            X, _, _ = load_performance_data()
            if X is not None:
                feature_names = X.columns
                importances = model.feature_importances_
                
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title="LightGBM Feature Importance (Fallback)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as fallback_error:
            st.error(f"Fallback option also failed: {fallback_error}")


def main():
    """Main application"""
    st.title("🚗 Sri Lanka Vehicle Analytics Platform")
    st.write("Comprehensive vehicle price prediction and market analysis system")

    # Load model and encoders
    model, encoders = load_assets()

    if model is None:
        st.error("❌ Model not found! Please run the training script first (main.py).")
        st.info("Run: `python main.py` to train the model before using this application.")
        return

    # Create tabs
    tabs = st.tabs([
        "🔮 Price Prediction", 
        "📊 Market Overview", 
        "🏢 Brand Analysis", 
        "🔗 Feature Analysis",
        "📈 Model Performance", 
        "🧠 SHAP Explainability"
    ])
    
    with tabs[0]:
        prediction_tab(model, encoders)
    
    with tabs[1]:
        eda_overview_tab()
    
    with tabs[2]:
        eda_brand_analysis_tab()
    
    with tabs[3]:
        eda_feature_analysis_tab()
    
    with tabs[4]:
        model_performance_tab(model)
    
    with tabs[5]:
        shap_explainability_tab(model)

if __name__ == "__main__":
    main()