import os
from typing import List, Optional
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

    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    print(f"\nAvailable CSV files: {csv_files}")

    df = pd.read_csv(os.path.join(path, csv_files[0]))

    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
    else:
        print("\nNo missing values found!")

    return df


def prepare_features(
    df: pd.DataFrame, target_col: str, exclude_cols: Optional[List[str]] = None
):
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
        exclude_cols = [target_col, "Player", "Data", "FG%", "PTS", "GmSc"]

        # Conditionally exclude based on target
        if target_col in ["FG", "FGA"]:
            exclude_cols.extend(["PTS", "GmSc"])
        elif target_col == "PTS":
            exclude_cols.extend(["GmSc", "FG"])
        elif target_col == "GmSc":
            exclude_cols.extend(["PTS"])

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


def aggregate_team_game_stats(df: pd.DataFrame):
    """
    Aggregate player-level stats to team-game level.

    Args:
        df: DataFrame with player-level data

    Returns:
        DataFrame with one row per team per game
    """
    # Aggregate statistics by team, opponent, date, and result
    team_stats = (
        df.groupby(["Tm", "Opp", "Data", "Res"])
        .agg(
            {
                # Shooting stats - sum across all players
                "FG": "sum",
                "FGA": "sum",
                "3P": "sum",
                "3PA": "sum",
                "FT": "sum",
                "FTA": "sum",
                "PTS": "sum",
                # Rebounding - sum
                "ORB": "sum",
                "DRB": "sum",
                "TRB": "sum",
                # Other stats - sum
                "AST": "sum",
                "STL": "sum",
                "BLK": "sum",
                "TOV": "sum",
                "PF": "sum",
                # Minutes - could use sum or mean depending on interpretation
                "MP": "mean",  # Average minutes per player
            }
        )
        .reset_index()
    )

    # Calculate derived features
    team_stats["team_fg_pct"] = team_stats["FG"] / team_stats["FGA"]
    team_stats["team_3p_pct"] = team_stats["3P"] / team_stats["3PA"]
    team_stats["team_ft_pct"] = team_stats["FT"] / team_stats["FTA"]

    # Create binary target variable (1 = Win, 0 = Loss)
    team_stats["win"] = (team_stats["Res"] == "W").astype(int)

    return team_stats
def plot_player_pts_distribution(player_stats_df):
    plt.figure()
    plt.hist(player_stats_df["PTS"], bins=30, edgecolor="black")
    plt.title("Distribution of Player PTS per Game")
    plt.xlabel("PTS")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_team_stat_distributions(team_stats_df):
    candidate_team_cols = ["PTS", "FG", "FGA", "3P", "3PA", "FT", "FTA", "TRB", "AST", "STL","BLK", "TOV",]    
    team_dist_cols = [c for c in candidate_team_cols if c in team_stats_df.columns]
    n = len(team_dist_cols)
    rows = 2
    cols = int(np.ceil(n / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, team_dist_cols):
        ax.hist(team_stats_df[col], bins=30, edgecolor="black")
        ax.set_title(col)
        
    for j in range(len(team_dist_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Team-level Stat Distributions", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_player_pts_distribution(player_stats_df: pd.DataFrame) -> None:

    plt.figure()
    plt.hist(player_stats_df["PTS"], bins=30, edgecolor="black")
    plt.title("Distribution of Player PTS per Game")
    plt.xlabel("PTS")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_team_stat_distributions(team_stats_df):

    candidate_team_cols = [
        "PTS", "FG", "FGA", "3P", "3PA",
        "FT", "FTA", "TRB", "AST", "STL",
        "BLK", "TOV",
    ]
    team_dist_cols = [c for c in candidate_team_cols if c in team_stats_df.columns]


    print("Team-level columns plotted:", team_dist_cols)

    n = len(team_dist_cols)
    rows = 2
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, team_dist_cols):
        ax.hist(team_stats_df[col], bins=30, edgecolor="black")
        ax.set_title(col)

    for j in range(len(team_dist_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Team-level Stat Distributions", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(player_stats_df, team_stats_df):
    
    player_cols = ["PTS", "FG", "FGA", "3P", "3PA", "FT", "FTA","TRB", "AST", "STL", "BLK", "TOV", "MP",]
    player_cols = [c for c in player_cols if c in player_stats_df.columns]

    corr = player_stats_df[player_cols].corr()
    fig, ax = plt.subplots(figsize=(0.8 * len(player_cols) + 3,0.8 * len(player_cols) + 3))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(player_cols)))
    ax.set_xticklabels(player_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(player_cols)))
    ax.set_yticklabels(player_cols)
    ax.set_title("Player-Level Correlation Heatmap")
    
    plt.tight_layout()
    plt.show()


    team_cols = ["PTS", "FG", "FGA", "3P", "3PA","FT", "FTA", "TRB", "AST","STL", "BLK", "TOV", "team_fg_pct","team_3p_pct", "team_ft_pct", "win",]
    team_cols = [c for c in team_cols if c in team_stats_df.columns]
    

    corr = team_stats_df[team_cols].corr()
    fig, ax = plt.subplots(figsize=(0.8 * len(team_cols) + 3,
                                    0.8 * len(team_cols) + 3))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(team_cols)))
    ax.set_xticklabels(team_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(team_cols)))
    ax.set_yticklabels(team_cols)
    ax.set_title("Team-Level Correlation Heatmap")
    
    plt.tight_layout()
    plt.show()


def plot_win_loss_feature_comparisons(team_stats_df):
    
    df = team_stats_df.copy()
    df["result_label"] = df["win"].map({1: "Win", 0: "Loss"})

    win_loss_features = ["PTS", "team_fg_pct", "team_3p_pct", "team_ft_pct","TRB", "AST", "TOV",]
    win_loss_features = [c for c in win_loss_features if c in df.columns]
    summary = (df.groupby("result_label")[win_loss_features].mean().round(2))

    print(summary)

    n = len(win_loss_features)
    rows = 2
    cols = int(np.ceil(n / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, win_loss_features):
        win_vals = df.loc[df["result_label"] == "Win", col]
        loss_vals = df.loc[df["result_label"] == "Loss", col]
         
        ax.boxplot([loss_vals.dropna(), win_vals.dropna()],labels=["Loss", "Win"],)
        ax.set_title(f"{col} by Result")
        ax.set_ylabel(col)
    for j in range(len(win_loss_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_temporal_trends(team_stats_df):
    df = team_stats_df.copy()
    df["game_date"] = pd.to_datetime(df["Data"])
    df = df.sort_values("game_date")

    mean_3pa_aggregation = None

    if "3PA" in df.columns:
        mean_3pa_aggregation = ("3PA", "mean")
    else:
        mean_3pa_aggregation = ("PTS", "mean")

    grouped = df.groupby("game_date")
    trend_df = grouped.agg(mean_pts=("PTS", "mean"),mean_3pa=mean_3pa_aggregation,win_rate=("win", "mean"),).reset_index()
    print(trend_df.head())
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(trend_df["game_date"], trend_df["mean_pts"])
    axes[0].set_ylabel("Avg PTS")
    axes[0].set_title("Temporal Trend: League Avg Points")

    axes[1].plot(trend_df["game_date"], trend_df["mean_3pa"])
    axes[1].set_ylabel("Avg 3PA")
    axes[1].set_title("Temporal Trend: League Avg 3PA")

    axes[2].plot(trend_df["game_date"], trend_df["win_rate"])
    axes[2].set_ylabel("Win Rate")
    axes[2].set_title("Temporal Trend: Avg Win Rate")
    axes[2].set_xlabel("Date")

    plt.tight_layout()
    plt.show()
