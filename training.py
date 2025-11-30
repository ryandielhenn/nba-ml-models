import numpy as np
import pandas as pd
import data_utils
from sklearn.model_selection import train_test_split
from models import classification_configs_baseline, regression_configs_baseline
from analysis_visuals import actual_vs_pred, residuals_plot, feature_importance_plot
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    # Classification metrics
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_models(X_train, 
                              X_test, 
                              y_train, 
                              y_test, 
                              feature_cols, 
                              target_name, 
                              classification=False):
    """
    Train and evaluate all models for a given target variable.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        feature_cols: List of feature column names
        target_name: Name of target variable (for display)

    Returns:
        dict: Results for each model
    """
    print("\n" + "=" * 80)
    print(f"MODEL TRAINING FOR {target_name}")
    print("=" * 80)

    models = (
        classification_configs_baseline()
        if classification
        else regression_configs_baseline()
    )
    results = {}

    # Initialize scaler once for all models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for model_name, config in models.items():
        print("\n" + "-" * 80)
        print(f"Model: {model_name}")
        print("-" * 80)

        # Select scaled or unscaled data based on model requirements
        X_train_use = X_train_scaled if config["use_scaled"] else X_train
        X_test_use = X_test_scaled if config["use_scaled"] else X_test

        # Train model
        model = config["model"]
        model.fit(X_train_use, y_train)
        y_pred = model.predict(X_test_use)
        
        # Visualize results for regression models
        actual_vs_pred(y_test, y_pred, title=f"{model_name} - Actual vs Predicted")
        # Visualize residuals for regression models
        residuals_plot(y_test, y_pred, title=f"{model_name} - Residuals Plot")
        # Visualize feature importances if available
        if hasattr(model, "feature_importances_"):
            feature_importance_plot(model.feature_importances_, feature_cols, title=f"{model_name} - Feature Importances")
        
        
        # Calculate metrics based on task type
        if classification:
            # Classification metrics
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            results[model_name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1": f1_score(y_test, y_pred),
                "ROC-AUC": roc_auc_score(y_test, y_pred_proba),
            }

            # Print metrics
            for metric, value in results[model_name].items():
                print(f"{metric}: {value:.4f}")

            # Print confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print("\nConfusion Matrix:")
            print("  Predicted:  Loss | Win")
            print(f"Actual Loss:  {cm[0][0]:3d} | {cm[0][1]:3d}")
            print(f"Actual Win:   {cm[1][0]:3d} | {cm[1][1]:3d}")

        else:
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[model_name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

            # Print metrics
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")

        # Print coefficients or feature importances (same for both)
        if config["has_coef"] and hasattr(model, "coef_"):
            print("\nFeature Coefficients:")
            coefs = model.coef_[0] if classification else model.coef_
            for feature, coef in zip(feature_cols, coefs):
                print(f"  {feature}: {coef:.4f}")
        elif hasattr(model, "feature_importances_"):
            print("\nFeature Importances:")
            for feature, importance in zip(feature_cols, model.feature_importances_):
                print(f"  {feature}: {importance:.4f}")
    return results


def summarize_results(results, target_name):
    """
    Print summary of model results.
    Auto-detects classification vs regression based on metrics.

    Args:
        results: Dictionary of model results
        target_name: Name of target variable
    """
    print("\n" + "=" * 80)
    print(f"SUMMARY OF RESULTS FOR {target_name}")
    print("=" * 80)

    results_df = pd.DataFrame(results).T

    # Auto-detect task type based on metrics present
    is_classification = "Accuracy" in results_df.columns

    if is_classification:
        # Print classification metrics
        print(
            f"\n{'':<20s}{'Accuracy':>10s}{'Precision':>12s}{'Recall':>10s}{'F1':>10s}{'ROC-AUC':>10s}"
        )
        for model_name, row in results_df.iterrows():
            print(
                f"{model_name:<20s}{row['Accuracy']:>10.4f}{row['Precision']:>12.4f}{row['Recall']:>10.4f}{row['F1']:>10.4f}{row['ROC-AUC']:>10.4f}"
            )

        # Identify best models
        best_roc = results_df["ROC-AUC"].idxmax()
        best_f1 = results_df["F1"].idxmax()
        best_acc = results_df["Accuracy"].idxmax()

        print(f"\nBest Model (by ROC-AUC): {best_roc}")
        print(f"Best Model (by F1-Score): {best_f1}")
        print(f"Best Model (by Accuracy): {best_acc}")
    else:
        # Print regression metrics
        print(f"\n{'':<20s}{'RMSE':>10s}{'MAE':>12s}{'R²':>10s}")
        for model_name, row in results_df.iterrows():
            print(
                f"{model_name:<20s}{row['RMSE']:>10.6f}{row['MAE']:>12.6f}{row['R²']:>10.6f}"
            )

        # Identify best models
        best_model_r2 = results_df["R²"].idxmax()
        best_model_rmse = results_df["RMSE"].idxmin()
        best_model_mae = results_df["MAE"].idxmin()

        print(f"\nBest Model (by R²): {best_model_r2}")
        print(f"Best Model (by RMSE): {best_model_rmse}")
        print(f"Best Model (by MAE): {best_model_mae}")

    return results_df


def predict_target(df, 
                   target_col, 
                   test_size=0.4, 
                   random_state=42, 
                   classification=False):
    """
    Complete pipeline for predicting a target variable.

    Args:
        df: DataFrame with data
        target_col: Target variable to predict
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        classification: Boolean, True for classification, False for regression

    Returns:
        tuple: (results dict, results DataFrame)
    """
    print("\n" + "#" * 80)
    print(f"# PREDICTION PIPELINE FOR: {target_col}")
    print("#" * 80)

    # Prepare features
    X, y, feature_cols = data_utils.prepare_features(df, target_col)

    # Split data (stratify for classification to maintain class balance)
    stratify = y if classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    if classification:
        print(f"Train win rate: {y_train.mean():.2%}")
        print(f"Test win rate: {y_test.mean():.2%}")

    # Train and evaluate models
    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, feature_cols, target_col, classification
    )

    # Summarize results (auto-detects task type)
    results_df = summarize_results(results, target_col)

    return results, results_df
