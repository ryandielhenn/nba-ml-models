from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
# TODO (Jesus): Add these imports after implementing XGBoost/LightGBM
# from xgboost import XGBRegressor, XGBClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier

def tune_hyperparameters(X_train, y_train, model_type='random_forest'):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    Cross-validation is handled automatically by these methods.

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to tune ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')

    Returns:
        GridSearchCV object containing:
            - best_estimator_: The best trained model
            - best_params_: The best parameter combination
            - best_score_: The best cross-validation score
            - cv_results_: Detailed results for all parameter combinations

    Usage:
        grid_search = tune_hyperparameters(X_train, y_train, 'random_forest')
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        y_pred = best_model.predict(X_test)

    TODO (Jesus): Implement this function with:
        1. Define param_grid for each model_type
        2. Create GridSearchCV with cv=5
        3. Fit and return the grid_search object
    """
    # TODO: Add imports at top of notebook if needed
    # from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    # TODO: Define parameter grids for different models
    # param_grids = {
    #     'random_forest': {
    #         'n_estimators': [50, 100, 200],
    #         'max_depth': [10, 20, None],
    #         'min_samples_split': [2, 5, 10]
    #     },
    #     'gradient_boosting': {
    #         'n_estimators': [50, 100, 200],
    #         'learning_rate': [0.01, 0.1, 0.2],
    #         'max_depth': [3, 5, 7]
    #     }
    # }

    # TODO: Create GridSearchCV and fit
    # grid_search = GridSearchCV(
    #     estimator=...,
    #     param_grid=param_grids[model_type],
    #     cv=5,  # 5-fold cross-validation (automatic)
    #     scoring='r2',
    #     n_jobs=-1,  # Use all CPU cores
    #     verbose=1
    # )
    # grid_search.fit(X_train, y_train)
    # return grid_search

    raise NotImplementedError("Jesus: Implement hyperparameter tuning here!")


def regression_configs_baseline():
    """
    Baseline regression model configurations with default hyperparameters.
    
    Returns:
        dict: Regression model configurations (untuned)
    """
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'use_scaled': True,
            'has_coef': True
        },
        'Random Forest': {
            'model': RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        }
    }
    return models


def regression_configs_tuned():
    """
    Tuned regression model configurations with optimized hyperparameters.
    
    TODO (Jesus): After running tune_hyperparameters() on regression models:
    1. Run: grid_search = tune_hyperparameters(X_train, y_train, 'random_forest')
    2. Print: grid_search.best_params_
    3. Update the parameters below with best_params_ values
    4. Repeat for gradient_boosting, xgboost, lightgbm
    5. Uncomment XGBoost and LightGBM sections after adding them
    
    Returns:
        dict: Regression model configurations (tuned)
    """
    models = {
        'Linear Regression': {
            'model': LinearRegression(),
            'use_scaled': True,
            'has_coef': True
        },
        'Random Forest (Tuned)': {
            'model': RandomForestRegressor(
                n_estimators=100,      # TODO: Replace with best_params_['n_estimators']
                max_depth=None,        # TODO: Replace with best_params_['max_depth']
                min_samples_split=2,   # TODO: Replace with best_params_['min_samples_split']
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
        'Gradient Boosting (Tuned)': {
            'model': GradientBoostingRegressor(
                n_estimators=100,      # TODO: Replace with best_params_['n_estimators']
                learning_rate=0.1,     # TODO: Replace with best_params_['learning_rate']
                max_depth=3,           # TODO: Replace with best_params_['max_depth']
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        }
        # TODO (Jesus): Uncomment after tuning XGBoost
        # 'XGBoost (Tuned)': {
        #     'model': XGBRegressor(
        #         n_estimators=100,      # TODO: Replace with best_params_
        #         learning_rate=0.1,     # TODO: Replace with best_params_
        #         max_depth=6,           # TODO: Replace with best_params_
        #         random_state=42
        #     ),
        #     'use_scaled': False,
        #     'has_coef': False
        # },
        # TODO (Jesus): Uncomment after tuning LightGBM
        # 'LightGBM (Tuned)': {
        #     'model': LGBMRegressor(
        #         n_estimators=100,      # TODO: Replace with best_params_
        #         learning_rate=0.1,     # TODO: Replace with best_params_
        #         max_depth=6,           # TODO: Replace with best_params_
        #         random_state=42,
        #         verbose=-1
        #     ),
        #     'use_scaled': False,
        #     'has_coef': False
        # }
    }
    return models


def classification_configs_baseline():
    """
    Baseline classification model configurations with default hyperparameters.
    
    Returns:
        dict: Classification model configurations (untuned)
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'use_scaled': True,
            'has_coef': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        }
    }
    return models


def classification_configs_tuned():
    """
    Tuned classification model configurations with optimized hyperparameters.
    
    TODO (Jesus): After running tune_hyperparameters() on classification models:
    1. Run: grid_search = tune_hyperparameters(X_train, y_train, 'random_forest')
    2. Print: grid_search.best_params_
    3. Update the parameters below with best_params_ values
    4. Repeat for gradient_boosting, xgboost, lightgbm
    5. Uncomment XGBoost and LightGBM sections after adding them
    
    Returns:
        dict: Classification model configurations (tuned)
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42),
            'use_scaled': True,
            'has_coef': True
        },
        'Random Forest (Tuned)': {
            'model': RandomForestClassifier(
                n_estimators=100,      # TODO: Replace with best_params_['n_estimators']
                max_depth=None,        # TODO: Replace with best_params_['max_depth']
                min_samples_split=2,   # TODO: Replace with best_params_['min_samples_split']
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        },
        'Gradient Boosting (Tuned)': {
            'model': GradientBoostingClassifier(
                n_estimators=100,      # TODO: Replace with best_params_['n_estimators']
                learning_rate=0.1,     # TODO: Replace with best_params_['learning_rate']
                max_depth=3,           # TODO: Replace with best_params_['max_depth']
                random_state=42
            ),
            'use_scaled': False,
            'has_coef': False
        }
        # TODO (Jesus): Uncomment after tuning XGBoost
        # 'XGBoost (Tuned)': {
        #     'model': XGBClassifier(
        #         n_estimators=100,      # TODO: Replace with best_params_
        #         learning_rate=0.1,     # TODO: Replace with best_params_
        #         max_depth=6,           # TODO: Replace with best_params_
        #         random_state=42
        #     ),
        #     'use_scaled': False,
        #     'has_coef': False
        # },
        # TODO (Jesus): Uncomment after tuning LightGBM
        # 'LightGBM (Tuned)': {
        #     'model': LGBMClassifier(
        #         n_estimators=100,      # TODO: Replace with best_params_
        #         learning_rate=0.1,     # TODO: Replace with best_params_
        #         max_depth=6,           # TODO: Replace with best_params_
        #         random_state=42,
        #         verbose=-1
        #     ),
        #     'use_scaled': False,
        #     'has_coef': False
        # }
    }
    return models
