import lightgbm as lgb
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(df):
    # Select features usually available - ensure they match preprocessing
    features = ['year', 'mileage', 'engine_cc', 'vehicle_age', 'mileage_per_year', 
                'brand', 'model', 'fuel_type', 'transmission', 'location', 'condition']
    
    # Ensure columns exist
    available_features = [col for col in features if col in df.columns]
    
    X = df[available_features]
    y = df['price_log']
    
    # Step 1: Split 70% Train and 30% "Other" (Validation + Test)
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=42)
    
    # Step 2: Split the 30% into half Validation (15%) and half Test (15%)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=42)

    print(f"Data Splits - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # LightGBM Model
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        objective='regression',
        random_state=42
    )
    
    # Fit the model with Early Stopping on Validation set
    model.fit(
        X_train, 
        y_train, 
        eval_set=[(X_val, y_val)], 
        eval_metric='l1',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lgbm_model.pkl')
    
    # Evaluation on Test Set
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    rmse = np.sqrt(mean_absolute_error(np.expm1(y_test), np.expm1(preds))) # Approximation for demonstration or calculate properly
    # Actually let's calculate standard RMSE on the original scale
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(np.expm1(y_test), np.expm1(preds))
    rmse = np.sqrt(mse)

    print("\nModel Performance on Test Set:")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE (LKR): {mae:,.2f}")
    print(f"RMSE (LKR): {rmse:,.2f}")
    
    return model, X_test
