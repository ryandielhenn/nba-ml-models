import numpy as np
import pandas as pd
import data_utils
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from models import classification_configs_baseline, classification_configs_tuned, regression_configs_baseline, regression_configs_tuned
from analysis_visuals import actual_vs_pred, residuals_plot, feature_importance_plot, plot_roc_curve, plot_confusion_matrix
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

    # Initialize scaler once for all models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    results = {}
    #Add cross-validation strategy to regression and classification configs
    if classification:
        # StratifiedKFold for classification to maintain class balance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        # KFold for regression for continuous targets
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
  
    
    #if classification, use classification configs
    if classification:
        models = {
            **classification_configs_tuned(X_train, y_train, X_train_scaled),
            **classification_configs_baseline()
    }
    #else, use regression configs
    else:
        models = {
            **regression_configs_tuned(X_train, y_train),
            **regression_configs_baseline(),
        }
      
    results = {}



    for model_name, config in models.items():
        print("\n" + "-" * 80)
        print(f"Model: {model_name}")
        print("-" * 80)
        
  
        # Select scaled or unscaled data based on model requirements
        X_train_use = X_train_scaled if config["use_scaled"] else X_train
        X_test_use = X_test_scaled if config["use_scaled"] else X_test

        # Train model
        model = config["model"]
        #Cross-validation on the training set. scores will be accuracy for classification and neg_mean_squared_error for regression
        if classification:
            scoring = {
                'accuracy': 'accuracy',
                'precision': 'precision',
                'recall': 'recall',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
        else:
            scoring = {
                'neg_rmse': 'neg_root_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
        
        cv_results = cross_validate(
            model, 
            X_train_use, 
            y_train, 
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        
        results[model_name] = {}
        
        if classification:
            results[model_name] = {
                "CV_Accuracy": cv_results['test_accuracy'].mean(),
                "CV_Precision": cv_results['test_precision'].mean(),
                "CV_Recall": cv_results['test_recall'].mean(),
                "CV_F1": cv_results['test_f1'].mean(),
                "CV_ROC-AUC": cv_results['test_roc_auc'].mean(),
            }
            print(f"Cross-Validation Accuracy: {results[model_name]['CV_Accuracy']:.4f}")
            print(f"Cross-Validation Precision: {results[model_name]['CV_Precision']:.4f}")
            print(f"Cross-Validation Recall: {results[model_name]['CV_Recall']:.4f}")
            print(f"Cross-Validation F1: {results[model_name]['CV_F1']:.4f}")
            print(f"Cross-Validation ROC-AUC: {results[model_name]['CV_ROC-AUC']:.4f}")
        else:
            results[model_name] = {
                "CV_RMSE": -cv_results['test_neg_rmse'].mean(),
                "CV_MAE": -cv_results['test_neg_mae'].mean(),
                "CV_R²": cv_results['test_r2'].mean(),
            }
            print(f"Cross-Validation RMSE: {results[model_name]['CV_RMSE']:.4f}")
            print(f"Cross-Validation MAE: {results[model_name]['CV_MAE']:.4f}")
            print(f"Cross-Validation R²: {results[model_name]['CV_R²']:.4f}")            
            
        model.fit(X_train_use, y_train)
        y_pred = model.predict(X_test_use)
        
        # Calculate metrics based on task type
        if classification:
            # Classification metrics
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            plot_roc_curve(model_name, model, X_test_use, y_test)
            plot_confusion_matrix(model_name, model, X_test_use, y_test)
            
            
            
            #calculate classification metrics
            #store in results dict
            classification_metrics = {
                "TEST_Accuracy": accuracy_score(y_test, y_pred),
                "TEST_Precision": precision_score(y_test, y_pred),
                "TEST_Recall": recall_score(y_test, y_pred),
                "TEST_F1": f1_score(y_test, y_pred),
                "TEST_ROC-AUC": roc_auc_score(y_test, y_pred_proba),
            }
            #update results dict
            results[model_name].update(classification_metrics)
            
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
            # Visualize results for regression models
            actual_vs_pred(y_test, y_pred, title=f"{model_name} - Actual vs Predicted")
            # Visualize residuals for regression models
            residuals_plot(y_test, y_pred, title=f"{model_name} - Residuals Plot")
            # Visualize feature importances if available
            if hasattr(model, "feature_importances_"):
                feature_importance_plot(model.feature_importances_, feature_cols, title=f"{model_name} - Feature Importances")
        
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)


            #regression metrics dict
            regression_metrics = {"TEST_RMSE": rmse, "TEST_MAE": mae, "TEST_R²": r2}
            #update results dict
            results[model_name].update(regression_metrics)

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
    is_classification = "TEST_Accuracy" in results_df.columns

    if is_classification:
        # Print classification metrics
        print(
            f"\n{'':<20s}{'TEST_Accuracy':>10s}{'TEST_Precision':>12s}{'TEST_Recall':>10s}{'F1':>10s}{'ROC-AUC':>10s}"
        )
        for model_name, row in results_df.iterrows():
            print(
                f"{model_name:<20s}{row['TEST_Accuracy']:>10.4f}{row['TEST_Precision']:>12.4f}{row['TEST_Recall']:>10.4f}{row['TEST_F1']:>10.4f}{row['TEST_ROC-AUC']:>10.4f}"
            )

        # Identify best models
        best_roc = results_df["TEST_ROC-AUC"].idxmax()
        best_f1 = results_df["TEST_F1"].idxmax()
        best_acc = results_df["TEST_Accuracy"].idxmax()

        print(f"\nBest Model (by ROC-AUC): {best_roc}")
        print(f"Best Model (by F1-Score): {best_f1}")
        print(f"Best Model (by Accuracy): {best_acc}")
    else:
        # Print regression metrics
        print(f"\n{'':<20s}{'TEST_RMSE':>10s}{'TEST_MAE':>12s}{'TEST_R²':>10s}")
        for model_name, row in results_df.iterrows():
            print(
                f"{model_name:<20s}{row['TEST_RMSE']:>10.6f}{row['TEST_MAE']:>12.6f}{row['TEST_R²']:>10.6f}"
            )

        # Identify best models
        best_model_r2 = results_df["TEST_R²"].idxmax()
        best_model_rmse = results_df["TEST_RMSE"].idxmin()
        best_model_mae = results_df["TEST_MAE"].idxmin()

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
        if y.dtype == "O" or not np.issubdtype(y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            print("\nClass encoding:")
            for cls, code in zip(le.classes_, le.transform(le.classes_)):
                print(f"  {cls} -> {code}")

            # replace y with numeric encoded version
            y = pd.Series(y_encoded, index=y.index, name=y.name)

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
