import os
import kagglehub
import pandas as pd
import numpy as np

def load_nba_data():
    """
    Download and load NBA player stats dataset from Kaggle.
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    print("Downloading dataset...")
    path = kagglehub.dataset_download("eduardopalmieri/nba-player-stats-season-2425")
    print(f"Path to dataset files: {path}")
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"\nAvailable CSV files: {csv_files}")
    
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    
    print("\n" + "="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
    else:
        print("\nNo missing values found!")
    
    return df


def prepare_features(df, target_col, exclude_cols=None):
    """
    Prepare features and target variable for modeling.
    
    Args:
        df: DataFrame with raw data
        target_col: Name of target variable column
        exclude_cols: List of columns to exclude (default: auto-detected)
    
    Returns:
        tuple: (X, y, feature_names)
    """
    if exclude_cols is None:
        # Auto-detect columns to exclude
        exclude_cols = [target_col, 'Player', 'Data', 'FG%', 'PTS', 'GmSc']
        
        # Conditionally exclude based on target
        if target_col in ['FG', 'FGA']:
            exclude_cols.extend(['PTS', 'GmSc'])
        elif target_col == 'PTS':
            exclude_cols.extend(['GmSc', 'FG'])
        elif target_col == 'GmSc':
            exclude_cols.extend(['PTS'])
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select only numeric features for now
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nTarget variable: {target_col}")
    print(f"Feature variables ({len(numeric_cols)} total): {numeric_cols}")
    
    # Create feature matrix and target vector
    X = df[numeric_cols].copy()
    y = df[target_col].copy()
    
    # Clean data
    valid_indices = X.notna().all(axis=1) & y.notna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    y = y[X.index]
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    
    return X, y, numeric_cols

def aggregate_team_game_stats(df):
    """
    Aggregate player-level stats to team-game level.
    
    Args:
        df: DataFrame with player-level data
    
    Returns:
        DataFrame with one row per team per game
    """
    # Aggregate statistics by team, opponent, date, and result
    team_stats = df.groupby(['Tm', 'Opp', 'Data', 'Res']).agg({
        # Shooting stats - sum across all players
        'FG': 'sum',
        'FGA': 'sum',
        '3P': 'sum',
        '3PA': 'sum',
        'FT': 'sum',
        'FTA': 'sum',
        'PTS': 'sum',
        
        # Rebounding - sum
        'ORB': 'sum',
        'DRB': 'sum',
        'TRB': 'sum',
        
        # Other stats - sum
        'AST': 'sum',
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'PF': 'sum',
        
        # Minutes - could use sum or mean depending on interpretation
        'MP': 'mean'  # Average minutes per player
    }).reset_index()
    
    # Calculate derived features
    team_stats['team_fg_pct'] = team_stats['FG'] / team_stats['FGA']
    team_stats['team_3p_pct'] = team_stats['3P'] / team_stats['3PA']
    team_stats['team_ft_pct'] = team_stats['FT'] / team_stats['FTA']
    
    # Create binary target variable (1 = Win, 0 = Loss)
    team_stats['win'] = (team_stats['Res'] == 'W').astype(int)
    
    return team_stats
