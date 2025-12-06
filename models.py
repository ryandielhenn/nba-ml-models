from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


def tune_hyperparameters(X_train, y_train, model_type: str = "random_forest"):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    Cross-validation is handled automatically by these methods.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')

    Returns:
        RandomizedSearchCV object containing:
            - best_estimator_: The best trained model
            - best_params_: The best parameter combination
            - best_score_: The best cross-validation score
            - cv_results_: Detailed results for all parameter combinations

    Usage:
        randomized_search = tune_hyperparameters(X_train, y_train, 'random_forest')
        best_model = randomized_search.best_estimator_
        best_params = randomized_search.best_params_
        y_pred = best_model.predict(X_test)
    """

    models = {
        'random_forest': RandomForestClassifier(random_state=42),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42),
        'xgboost': XGBClassifier(random_state=42),
        'lightgbm': LGBMClassifier(random_state=42, verbose=-1)
    }

    # Use RandomizedSearchCV because it's faster. 
    # Fixed number of tries instead of all possible parameter combinations.
    param_distributions = {
        'random_forest': {
            'n_estimators': randint(50, 500),
            'max_depth': randint(10, 50),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', None]
        },
        'gradient_boosting': {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'subsample': uniform(0.6, 0.4)
        },
        'logistic_regression': {
            'C': uniform(0.01, 10),
            'penalty': ['l1', 'l2'],
            'solver': ['saga'],
            'max_iter': [2000],
            'class_weight': ['balanced', None]
        },
        'xgboost': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        },
        'lightgbm': {
            'n_estimators': randint(50, 500),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }   
    }

    random_search = RandomizedSearchCV(
        estimator=models[model_type],
        param_distributions=param_distributions[model_type],
        n_iter=10,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search

def tune_hyperparameters_regression(X_train, y_train, model_type: str = "random_forest"):
    """Hyperparameter tuning for REGRESSION models."""
    models = {
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "xgboost": XGBRegressor(random_state=42),
        "lightgbm": LGBMRegressor(random_state=42, verbose=-1),
    }

    # Same distributions as classification, but used with regressors
    param_distributions = {
        "random_forest": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(10, 50),
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        },
        "gradient_boosting": {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.05, 0.25),
            "max_depth": randint(3, 10),
            "min_samples_split": randint(2, 20),
            "subsample": uniform(0.7, 0.3),
        },
        "xgboost": {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.05, 0.25),
            "max_depth": randint(3, 10),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        },
        "lightgbm": {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.05, 0.25),
            "max_depth": randint(3, 12),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        },
    }
    random_search = RandomizedSearchCV(
        estimator=models[model_type],
        param_distributions=param_distributions[model_type],
        n_iter=10,
        cv=5,
        scoring="neg_root_mean_squared_error",  # regression metric
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)
    return random_search


def regression_configs_baseline():
    """
    Baseline regression model configurations with default hyperparameters.

    Returns:
        dict: Regression model configurations (untuned)
    """
    models = {
        "Linear Regression": {
            "model": LinearRegression(),
            "use_scaled": True,
            "has_coef": True,
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        
        "XGBoost": {
            "model": XGBRegressor(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        "LightGBM": {
            "model": LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            "use_scaled": False,
            "has_coef": False,
        },
    }
    return models


def regression_configs_tuned(X_train, y_train):
    """
    Tuned regression model configurations with optimized hyperparameters.

    

    Returns:
        dict: Regression model configurations (tuned)
    """

    rf_search = tune_hyperparameters_regression(X_train, y_train, 'random_forest')
    gb_search = tune_hyperparameters_regression(X_train, y_train, 'gradient_boosting')
    xgb_search = tune_hyperparameters_regression(X_train, y_train, 'xgboost')
    lgbm_search = tune_hyperparameters_regression(X_train, y_train, 'lightgbm')

    models = {
        "Random Forest (Tuned)": {
            "model": RandomForestRegressor(
                **rf_search.best_params_,
                random_state=42,
            ),
            "use_scaled": False,
            "has_coef": False,
        },
        "Gradient Boosting (Tuned)": {
            "model": GradientBoostingRegressor(
                **gb_search.best_params_,
                random_state=42,
            ),
            "use_scaled": False,
            "has_coef": False,
        },
       
        'XGBoost (Tuned)': {
            'model': XGBRegressor(
                **xgb_search.best_params_,
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
      
        'LightGBM (Tuned)': {
            'model': LGBMRegressor(
                **lgbm_search.best_params_,
                random_state=42,
                verbose=-1
            ),
            'use_scaled': False,
            'has_coef': False
        }
    }
    return models


def classification_configs_baseline():
    """
    Baseline classification model configurations with default hyperparameters.

    Returns:
        dict: Classification model configurations (untuned)
    """
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "use_scaled": True,
            "has_coef": True,
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, random_state=42),
            "use_scaled": False,
            "has_coef": False,
        },
        "LightGBM": {
            "model": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            "use_scaled": False,
            "has_coef": False,
        },
    }
    return models


def classification_configs_tuned(X_train, y_train, X_train_scaled):
    """
    Tuned classification model configurations with optimized hyperparameters.

    

    Returns:
        dict: Classification model configurations (tuned)
    """
    # Run tuning for each model
    logreg_search = tune_hyperparameters(X_train_scaled, y_train, 'logistic_regression')
    rf_search = tune_hyperparameters(X_train, y_train, 'random_forest')
    gb_search = tune_hyperparameters(X_train, y_train, 'gradient_boosting')
    xgb_search = tune_hyperparameters(X_train, y_train, 'xgboost')
    lgbm_search = tune_hyperparameters(X_train, y_train, 'lightgbm')

    models = {
        "Logistic Regression (Tuned)": {
            "model": LogisticRegression(
                **logreg_search.best_params_,
                random_state=42
                ),
            "use_scaled": True,
            "has_coef": True,
        },
        "Random Forest (Tuned)": {
            "model": RandomForestClassifier(
                **rf_search.best_params_,
                random_state=42
            ),
            "use_scaled": False,
            "has_coef": False,
        },
        "Gradient Boosting (Tuned)": {
            "model": GradientBoostingClassifier(
                **gb_search.best_params_,
                random_state=42
            ),
            "use_scaled": False,
            "has_coef": False,
        },
       
        'XGBoost (Tuned)': {
            'model': XGBClassifier(
                
                n_estimators=xgb_search.best_params_['n_estimators'],      
                learning_rate=xgb_search.best_params_['learning_rate'],    
                max_depth=xgb_search.best_params_['max_depth'],           
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
       
        'LightGBM (Tuned)': {
            'model': LGBMClassifier(
                n_estimators=lgbm_search.best_params_['n_estimators'],      
                learning_rate=lgbm_search.best_params_['learning_rate'],     
                max_depth=lgbm_search.best_params_['max_depth'],          
                verbose=-1
            ),
        
            'use_scaled': False,
            'has_coef': False
        },
    }
    return models
