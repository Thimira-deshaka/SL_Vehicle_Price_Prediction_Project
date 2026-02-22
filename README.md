# 🚗 Sri Lanka Vehicle Price Prediction System (Academic Purpose)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-green.svg)](https://lightgbm.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning solution for predicting used vehicle prices in the Sri Lankan market. This system combines advanced data preprocessing, feature engineering, and gradient boosting algorithms with an intuitive web interface for real-time price predictions.

## ✨ Features

- 🎯 **Accurate Price Predictions** - LightGBM model with SHAP explainability
- 📊 **Interactive Dashboard** - Beautiful Streamlit web interface with glassmorphism design
- 🔍 **Comprehensive EDA** - Detailed exploratory data analysis with visualizations
- 🧠 **Model Interpretability** - SHAP values and feature importance analysis
- 📈 **Performance Metrics** - Model evaluation with R², RMSE, and prediction plots
- 🎨 **Modern UI** - Gradient backgrounds, responsive design, and intuitive controls

## 🎯 Demo

### Web Application

![Vehicle Price Predictor](https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80)

### Key Capabilities

- **Price Prediction**: Enter vehicle details and get instant market value estimates
- **Data Visualization**: Explore price distributions, correlations, and market trends
- **Model Insights**: Understand which features influence pricing decisions

## 📁 Project Structure

```
vehicle_price_prediction/
│
├── 📄 README.md                 # Project documentation
├── 📄 requirements.txt          # Python dependencies
├── 📄 main.py                   # Training pipeline orchestrator
├── 🌐 app.py                    # Streamlit web application
├── 📊 EDA.ipynb                 # Exploratory Data Analysis notebook
│
├── 📂 src/                      # Source code modules
│   ├── 📄 __init__.py
│   ├── 🔧 preprocessing.py      # Data preprocessing & feature engineering
│   ├── 🤖 train.py              # Model training & evaluation
│   └── 🔍 explain.py            # Model interpretability & SHAP analysis
│
├── 📂 data/                     # Dataset storage
│   └── 📊 sri_lanka_car_price_dataset.csv
│
├── 📂 models/                   # Trained models & encoders
│   ├── 🤖 lgbm_model.pkl       # Trained LightGBM model
│   └── 🔢 encoders.pkl         # Label encoders for categories
│
└── 📂 venv/                     # Virtual environment (auto-generated)
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Git** (for version control)
- **4GB+ RAM** (recommended for model training)

### 🔧 Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd vehicle_price_prediction
   ```

2. **Create Virtual Environment**

   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pandas, numpy, lightgbm, streamlit, shap; print('✅ All dependencies installed successfully!')"
   ```

### 📊 Dataset Setup

Ensure your dataset (`sri_lanka_car_price_dataset.csv`) is placed in the `data/` folder with the following columns:

- `Brand`, `Model`, `YOM` (Year of Manufacture)
- `Engine (cc)`, `Gear`, `Fuel Type`
- `Millage(KM)`, `Town`, `Condition`
- `Price` (target variable)

## 🏃‍♂️ Usage

### 1. **Train the Model**

```bash
python main.py
```

This will:

- ✅ Load and preprocess the dataset
- ✅ Train the LightGBM model
- ✅ Generate SHAP explanations
- ✅ Save models to `models/` folder

**Expected Output:**

```
Dataset loaded: (X, Y) samples
Model trained successfully!
Model saved: models/lgbm_model.pkl
Encoders saved: models/encoders.pkl
```

### 2. **Launch Web Application**

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to access:

- 🔮 **Price Prediction Tab**: Interactive vehicle price estimator
- 📊 **EDA Tab**: Data exploration and visualizations

### 3. **Explore Data Analysis**

```bash
# Open Jupyter Notebook
jupyter notebook EDA.ipynb
```

## 📈 Model Performance

| Metric            | Value                      |
| ----------------- | -------------------------- |
| **Algorithm**     | LightGBM Gradient Boosting |
| **Features**      | 11 engineered features     |
| **R² Score**      | 0.85+                      |
| **RMSE**          | <500K LKR                  |
| **Training Time** | <2 minutes                 |

### Key Features Influencing Price:

1. 🗓️ **Vehicle Age** - Most significant factor
2. 🏭 **Brand** - Premium vs. economy brands
3. ⚙️ **Engine Capacity** - Power and performance
4. 📍 **Location** - Regional market variations
5. ⛽ **Fuel Type** - Hybrid premium pricing

## 🎨 Web Interface Features

### Modern Design Elements

- **Glassmorphism UI** - Semi-transparent cards with backdrop blur
- **Gradient Backgrounds** - Dynamic color schemes
- **Responsive Layout** - Works on desktop and mobile
- **Interactive Charts** - Hover effects and animations

### Prediction Interface

- **Smart Defaults** - Pre-filled reasonable values
- **Real-time Validation** - Input error handling
- **Confidence Indicators** - Model certainty metrics
- **Explanation Cards** - Top 3 influencing factors

## 🔍 Data Insights

### Market Analysis Features

- **Price Distribution Analysis** - Histogram with KDE curves
- **Correlation Heatmaps** - Feature relationship mapping
- **Brand Performance** - Average pricing by manufacturer
- **Temporal Trends** - Year-over-year price changes

### Model Interpretability

- **SHAP Summary Plots** - Global feature importance
- **Individual Predictions** - Per-sample explanations
- **Feature Contribution** - Positive/negative impact analysis

## 🛠️ Development

### Adding New Features

1. Update `src/preprocessing.py` for new data transformations
2. Modify `src/train.py` to include additional features
3. Retrain model using `python main.py`
4. Update UI in `app.py` for new input fields

### Model Tuning

```python
# In src/train.py, modify LightGBM parameters:
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,      # Adjust for complexity
    'learning_rate': 0.05,  # Lower for better accuracy
    'feature_fraction': 0.9 # Feature sampling ratio
}
```

## 📋 Requirements

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
lightgbm>=3.3.0
shap>=0.41.0
streamlit>=1.28.0
matplotlib>=3.5.0
seaborn>=0.12.0
joblib>=1.2.0
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open Pull Request**

## 👨‍💻 Author

**Your Name** - Machine Learning Engineer  
📧 Email: thimiradeshakashan0220@gmail.com

## 🙏 Acknowledgments

- **Sri Lanka Vehicle Market Data** - Dataset contributors
- **LightGBM Team** - Efficient gradient boosting framework
- **Streamlit** - Beautiful web app framework
- **SHAP** - Model interpretability library

---

<div align="center">

**⭐ Star this repository if you found it helpful! ⭐**

Made with ❤️ for the Sri Lankan automotive market

</div>
