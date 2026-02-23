import shap
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_explanations(model, X_test, save_plots=True):
    """
    Generate SHAP explanations for the model
    
    Args:
        model: Trained LightGBM model
        X_test: Test features DataFrame
        save_plots: Whether to save plots to disk
    
    Returns:
        explainer, shap_values for further analysis
    """
    # LightGBM is tree-based, so use TreeExplainer is often faster/better
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if save_plots:
        os.makedirs('models', exist_ok=True)
        
        # Plot Global Importance (Bar plot)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.savefig('models/shap_importance.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot Global Importance (Beeswarm plot)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('models/shap_summary.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        print("SHAP plots saved to models/ directory")
    
    return explainer, shap_values

def explain_single_prediction(explainer, shap_values, X_test, sample_index=0, save_plot=True):
    """
    Generate explanation for a single prediction
    
    Args:
        explainer: SHAP explainer object
        shap_values: Computed SHAP values
        X_test: Test features DataFrame
        sample_index: Index of sample to explain
        save_plot: Whether to save the waterfall plot
    
    Returns:
        Dictionary with explanation details
    """
    if sample_index >= len(X_test):
        raise ValueError(f"Sample index {sample_index} out of bounds for dataset of size {len(X_test)}")
    
    sample = X_test.iloc[sample_index]
    sample_shap_values = shap_values[sample_index]
    
    if save_plot:
        os.makedirs('models', exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explainer.expected_value, sample_shap_values, sample, show=False)
        plt.savefig(f'models/explanation_sample_{sample_index}.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Individual explanation plot saved for sample {sample_index}")
    
    # Return structured explanation data
    explanation = {
        'expected_value': explainer.expected_value,
        'shap_values': sample_shap_values,
        'feature_values': sample.to_dict(),
        'feature_contributions': dict(zip(X_test.columns, sample_shap_values)),
        'total_impact': sample_shap_values.sum(),
        'final_prediction': explainer.expected_value + sample_shap_values.sum()
    }
    
    return explanation

def get_feature_importance_ranking(shap_values, feature_names):
    """
    Get feature importance ranking based on mean absolute SHAP values
    
    Args:
        shap_values: SHAP values array
        feature_names: List of feature names
    
    Returns:
        Sorted list of (feature_name, importance_score) tuples
    """
    importance_scores = np.abs(shap_values).mean(axis=0)
    feature_importance = list(zip(feature_names, importance_scores))
    return sorted(feature_importance, key=lambda x: x[1], reverse=True)
