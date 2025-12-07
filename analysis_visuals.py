import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


