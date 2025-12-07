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

    # Initialize scaler once for all models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
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
        
        if classification:
            results[model_name] = {
                "CV_Accuracy": cv_results['test_accuracy'].mean(),
                "CV_Precision": cv_results['test_precision'].mean(),
                "CV_Recall": cv_results['test_recall'].mean(),
                "CV_F1": cv_results['test_f1'].mean(),
                "CV_ROC-AUC": cv_results['test_roc_auc'].mean(),
            }
        else:
            results[model_name] = {
                "CV_RMSE": -cv_results['test_neg_rmse'].mean(),
                "CV_MAE": -cv_results['test_neg_mae'].mean(),
                "CV_R²": cv_results['test_r2'].mean(),
            }
            
        model.fit(X_train_use, y_train)
        y_pred = model.predict(X_test_use)
        
        # Calculate metrics based on task type
        if classification:
            # Classification metrics
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            
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
            

        else:
        
            # Regression metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)


            #regression metrics dict
            regression_metrics = {"TEST_RMSE": rmse, "TEST_MAE": mae, "TEST_R²": r2}
            #update results dict
            results[model_name].update(regression_metrics)

    return results


def summarize_results(results, target_name):
    """
    Print summary of model results including both CV and test metrics.
    Auto-detects classification vs regression based on metrics.
    
    Args:
        results: Dictionary of model results
        target_name: Name of target variable
        
    Returns:
        pd.DataFrame: Results organized as a DataFrame
    """
    print("\n" + "=" * 80)
    print(f"SUMMARY OF RESULTS FOR {target_name}")
    print("=" * 80)
    
    results_df = pd.DataFrame(results).T
    
    # Auto-detect task type based on metrics present
    is_classification = "TEST_Accuracy" in results_df.columns
    
    if is_classification:
        # Print CV metrics
        print("\nCross-Validation Performance:")
        print(f"{'Model':<35s}{'CV Acc':>10s}{'CV F1':>10s}{'CV AUC':>10s}")
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<35s}{row['CV_Accuracy']:>10.4f}{row['CV_F1']:>10.4f}{row['CV_ROC-AUC']:>10.4f}")
        
        # Print test metrics
        print("\nTest Set Performance:")
        print(f"{'Model':<35s}{'Test Acc':>10s}{'Test Prec':>12s}{'Test Rec':>10s}{'Test F1':>10s}{'Test AUC':>10s}")
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<35s}{row['TEST_Accuracy']:>10.4f}{row['TEST_Precision']:>12.4f}{row['TEST_Recall']:>10.4f}{row['TEST_F1']:>10.4f}{row['TEST_ROC-AUC']:>10.4f}")
        
        # Identify best models (using CV metrics for fair comparison)
        best_roc = results_df["CV_ROC-AUC"].idxmax()
        best_f1 = results_df["CV_F1"].idxmax()
        best_acc = results_df["CV_Accuracy"].idxmax()
        
        print(f"\nBest Model (by CV ROC-AUC): {best_roc}")
        print(f"Best Model (by CV F1-Score): {best_f1}")
        print(f"Best Model (by CV Accuracy): {best_acc}")
        
    else:
        # Print CV metrics
        print("\nCross-Validation Performance:")
        print(f"{'Model':<35s}{'CV RMSE':>12s}{'CV MAE':>12s}{'CV R²':>10s}")
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<35s}{row['CV_RMSE']:>12.6f}{row['CV_MAE']:>12.6f}{row['CV_R²']:>10.6f}")
        
        # Print test metrics
        print("\nTest Set Performance:")
        print(f"{'Model':<35s}{'Test RMSE':>12s}{'Test MAE':>12s}{'Test R²':>10s}")
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<35s}{row['TEST_RMSE']:>12.6f}{row['TEST_MAE']:>12.6f}{row['TEST_R²']:>10.6f}")
        
        # Identify best models (using CV metrics for fair comparison)
        best_model_r2 = results_df["CV_R²"].idxmax()
        best_model_rmse = results_df["CV_RMSE"].idxmin()
        best_model_mae = results_df["CV_MAE"].idxmin()
        
        print(f"\nBest Model (by CV R²): {best_model_r2}")
        print(f"Best Model (by CV RMSE): {best_model_rmse}")
        print(f"Best Model (by CV MAE): {best_model_mae}")
    
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
        X_train, X_test, y_train, y_test, classification
    )

    # Summarize results (auto-detects task type)
    summarize_results(results, target_col)
