import matplotlib.pyplot as plt
import numpy as np

def actual_vs_pred(y_test, y_pred, title="Actual vs Predicted"):
    """
    Plot actual vs predicted values for regression models.

    Args:
        y_test: True target values from the test set
        y_pred: Predicted values from the model
        title: Title for the plot
    """
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
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.grid(True)
    plt.show()
        
    
# def feature_importance_plot():

