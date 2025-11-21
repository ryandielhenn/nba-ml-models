# Project Progress Report

**Team Members:** Ryan Dielhenn, Harshil Patel, [Name 3]  
**Project Lead:** [Name]  
**Date:** Friday, November 21, 2025

---

## Project Overview

**Title:** NBA Player Stats

This project aims to predict NBA game outcomes and player shooting performance by analyzing the 2024-25 season data. We aim to develop machine learning models to forecast both game results (win/loss) and individual player field goal performance (FG/FGA) based on player statistics and game context. By analyzing patterns in shooting efficiency, minutes played, rebounds, assists, and other performance metrics, we seek to identify which factors most strongly correlate with team success and individual shooting performance. This dual-prediction approach provides insights for sports analytics, team strategy optimization, and understanding the relationship between individual player performance and team outcomes.

---

## Team Member Responsibilities

- **Ryan Dielhenn:** Feature engineering, dimensionality reduction (PCA), model training
- **Momoka Aung:** Model training and analysis.
- **Angel:** Data Visualization
- **Jesus:** Hyperparameter Tuning
- **Harshil:** Performance metrics analysis, project documentation and reporting

---

## Data Information

- **Data Source(s):** Kaggle - NBA Player Stats Season 2024-25 (https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425)
- **Data Format:** CSV (fetched via KaggleHub)
- **Dataset Size:** Version 37, 1.51 MB
- **Key Features:** Player performance statistics including FG (field goals), FGA (field goal attempts), FG%, 3P/3PA/3P% (three-point statistics), FT/FTA/FT% (free throw statistics), rebounds (ORB, DRB, TRB), AST (assists), STL (steals), BLK (blocks), TOV (turnovers), PF (personal fouls), PTS (points), MP (minutes played), and GmSc (game score). Additional contextual features include Player name, team (Tm), opponent (Opp), game result (Res), and date.
- **Data Challenges:** [Any issues with quality, missing values, etc.]

---

## Project Status and Progress

We are currently focused on predicting FG/FGA (field goals/field goal attempts) using multiple machine learning approaches. So far, we have implemented and evaluated three different models: Linear Regression, Random Forest, and Gradient Boosting. Our initial results show that Linear Regression performed best across all metrics with an R² of 0.8875, RMSE of 1.0871, and MAE of 0.7942. Feature importance analysis revealed that opposing team (Opp) is the strongest predictor, followed by game result (Res) and minutes played (MP). The next phase of our project will focus on predicting game results (win/loss outcomes). We are exploring two approaches: predicting results at the individual player level using per-game statistics, or aggregating player performance metrics by team and date to make team-level predictions. Both approaches have potential merit—individual player predictions could identify key performance thresholds that correlate with wins, while team-level aggregation may capture collective team dynamics more effectively.

---
