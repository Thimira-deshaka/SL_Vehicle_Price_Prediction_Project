import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import preprocess_data
from src.train import train_model
from src.explain import generate_explanations

def main():
    print("Starting Pipeline...")
    
    # 1. Load Data
    data_path = 'data/sri_lanka_car_price_dataset.csv'
    if not os.path.exists(data_path):
        # Create a dummy dataset for demonstration if file doesn't exist
        print(f"Warning: {data_path} not found. Creating a sample dataset.")
        os.makedirs('data', exist_ok=True)
        dummy_data = {
            'Brand': ['Toyota', 'Suzuki', 'Honda', 'Toyota', 'Nissan'] * 20,
            'Model': ['Corolla', 'Alto', 'Civic', 'Vitz', 'Sunny'] * 20,
            'Year of Manufacture': [2018, 2015, 2019, 2017, 2010] * 20,
            'Mileage': ['50,000 km', '30,000 km', '40,000 km', '60,000 km', '80,000 km'] * 20,
            'Fuel Type': ['Petrol', 'Petrol', 'Petrol', 'Hybrid', 'Diesel'] * 20,
            'Transmission': ['Automatic', 'Manual', 'Automatic', 'Automatic', 'Manual'] * 20,
            'Capacity': [1500, 800, 1800, 1000, 1300] * 20,
            'Price': ['Rs. 5,000,000', 'Rs. 2,500,000', 'Rs. 6,800,000', 'Rs. 4,500,000', 'Rs. 3,200,000'] * 20,
            'Location': ['Colombo', 'Gampaha', 'Kandy', 'Colombo', 'Galle'] * 20,
            'Condition': ['Used', 'Used', 'Used', 'Used', 'Used'] * 20
        }
        df = pd.DataFrame(dummy_data)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
    
    print("Data loaded. Shape:", df.shape)
    
    # 2. Preprocessing
    print("Preprocessing data...")
    df_processed, encoders = preprocess_data(df)
    print("Preprocessing complete. Processed shape:", df_processed.shape)
    
    # 3. Training
    print("Training LightGBM model...")
    model, X_test = train_model(df_processed)
    print("Training complete.")
    
    # 4. Explainability
    print("Generating SHAP explanations...")
    try:
        generate_explanations(model, X_test)
        print("Explanations generated.")
    except Exception as e:
        print(f"Skipping SHAP explanation due to error (common with version mismatches): {e}")

    print("Pipeline finished successfully. Run 'streamlit run app.py' to see the dashboard.")

if __name__ == "__main__":
    main()
