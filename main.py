# main.py

from data_utils import load_nba_data
from training import predict_target


def main():
    # 1. Load data
    df = load_nba_data()

    # =============================
    # A) REGRESSION: Predict points (PTS)
    # =============================
    print("\n" + "#" * 80)
    print("# RUNNING REGRESSION: Predicting PTS")
    print("#" * 80)

    results_pts, results_df_pts = predict_target(
        df,
        target_col="PTS",        # numeric target
        classification=False     # REGRESSION
    )

    print("\nRegression results for PTS:")
    print(results_df_pts)

    # =============================
    # B) CLASSIFICATION: Predict win/loss (Res)
    # =============================
    print("\n" + "#" * 80)
    print("# RUNNING CLASSIFICATION: Predicting Res (W/L)")
    print("#" * 80)

    # This assumes target column "Res" exists with values like 'W' and 'L'
    #encode 'W'/'L' to 'w'/'l' for consistency to 0,1 encoding in models
    results_res, results_df_res = predict_target(
        df,
        target_col="Res",       # classification target (W/L)
        classification=True     # CLASSIFICATION
    )

    print("\nClassification results for Res (W/L):")
    print(results_df_res)


if __name__ == "__main__":
    main()
   
