import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def train_and_explain():
    # Load sample data
    X, y = shap.datasets.adult()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Model accuracy: {model.score(X_test, y_test):.4f}")
    
    # Explain predictions
    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Generate plots
    plt.figure()
    shap.summary_plot(shap_values[1], X_test, show=False)
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()
    
    print("SHAP summary plot saved to shap_summary.png")

if __name__ == "__main__":
    train_and_explain()
