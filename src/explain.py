import shap
import matplotlib.pyplot as plt
import os

def generate_explanations(model, X_test):
    # LightGBM is tree-based, so use TreeExplainer is often faster/better
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Plot Global Importance (Beeswarm plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/shap_summary.png', bbox_inches='tight')
    print("SHAP summary plot saved to models/shap_summary.png")
