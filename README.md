# NBA Player Stats Model Training and Evaluation

## Installing dependencies and opening jupyter
`pip install -r requirements.txt`

`jupyter notebook`

At this point all cells in our main notebook, `nba_stats_prediction.ipynb` should run with no issues (venv recommended).

## Contents

This section provides an overview of the contents of this Report

`nba_stats_prediction.ipynb` is a notebook that loads the [NBA - Player Stats - Season 24/25](https://www.kaggle.com/datasets/eduardopalmieri/nba-player-stats-season-2425) dataset, performs exploratory data analysis including visualizations, and then runs a full machine learning pipeline to predict player PTS per game (points scored) and team win/loss.

`analysis_visuals.py` is a python file with functions for generating data visualizations including distributiions, ROC curve, and correlation heatmaps etc.

`training.py` is a python file with funtions for model training, evaluation, and predictions.

`models.py` is a python file with configurations for all baseline models that are used in the training pipeline in addition to functions for generating their tuned variants.

`data_utils.py` is a python file with functions for loading the dataset and preparing features such as data cleaning and using individual player stats to engineer team-game level features.

`doc` contains both the progress and final reports for this project. 
