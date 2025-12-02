import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

def plot_confusion_matrix(model_name, model, X_test, y_test):
    # Confusion Matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

def plot_roc_curve(model_name, model, X_test, y_test):
    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"{model_name} - ROC Curve")
    plt.show()

def actual_vs_pred(y_test, y_pred, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values for regression models.

    Args:
        y_test: True target values from the test set
        y_pred: Predicted values from the model
        title: Title for the plot
    """
    # Scatter plot of actual vs predicted values
    plt.scatter(y_test, y_pred, alpha=0.6, s=20, label="Predicted Points")
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    
    
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle="--",
             color="black",
             linewidth=2,
             label="Perfect Prediction Line")

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def residuals_plot(y_test, y_pred, title="Residuals Plot"):
    """
    Plot residuals for regression models.

    Args:
        y_test: True target values from the test set
        y_pred: Predicted values from the model
        title: Title for the plot
    """
    # Calculate residuals
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.show()
        
    
def feature_importance_plot(importances, feature_names, title="Feature Importances"):
    """
    Plot feature importances for models that provide them.

    Args:
        importances: Array of feature importances
        feature_names: List of feature names corresponding to the importances
        title: Title for the plot
    """
    # Sort importances and feature names
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_feature_names = [feature_names[i] for i in indices]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importances)), sorted_importances, align='center')
    plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=90)
    
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()

