import numpy as np
import pandas as pd
import pickle
import json
import sys
from pathlib import Path

MODEL_PATH = Path("model")
DATA_PATH = Path("data")


def get_n_best_players(df, pred, n):
    ind = np.argpartition(pred, -n)[-n:]
    ind = ind[np.argsort(pred[ind])][::-1]
    df = df.iloc[ind].reset_index(drop=True)
    return pd.concat([df, pd.DataFrame(pred[ind], columns=["PRED"])], axis=1)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <json_path>")
        sys.exit(1)

    with open(MODEL_PATH / "regr_all_nba.pkl", "rb") as f:
        regr_all_nba = pickle.load(f)

    with open(MODEL_PATH / "regr_all_rookie.pkl", "rb") as f:
        regr_all_rookie = pickle.load(f)

    combined_all_nba_df = pd.read_csv(DATA_PATH / "combined_all_nba.csv", header=[0, 1])
    combined_all_rookie_df = pd.read_csv(
        DATA_PATH / "combined_all_rookie.csv", header=[0, 1]
    )

    season_all_nba_df = combined_all_nba_df["2023-24"].dropna()
    season_all_rookie_df = combined_all_rookie_df["2023-24"].dropna()

    X_all_nba_df = season_all_nba_df.drop(columns=["PLAYER", "AWARD"])
    X_all_rookie_df = season_all_rookie_df.drop(columns=["PLAYER", "AWARD"])

    predicted_all_nba = regr_all_nba.predict(X_all_nba_df)
    predicted_all_rookie = regr_all_rookie.predict(X_all_rookie_df)

    best_players_all_nba_df = get_n_best_players(
        season_all_nba_df, predicted_all_nba, 15
    )
    best_players_all_rookie_df = get_n_best_players(
        season_all_rookie_df, predicted_all_rookie, 10
    )

    best_players_json = {
        "first all-nba team": best_players_all_nba_df["PLAYER"].to_list()[:5],
        "second all-nba team": best_players_all_nba_df["PLAYER"].to_list()[5:10],
        "third all-nba team": best_players_all_nba_df["PLAYER"].to_list()[10:15],
        "first rookie all-nba team": best_players_all_rookie_df["PLAYER"].to_list()[:5],
        "second rookie all-nba team": best_players_all_rookie_df["PLAYER"].to_list()[
            5:10
        ],
    }
    with open(sys.argv[1], "w") as f:
        json.dump(best_players_json, f)


if __name__ == "__main__":
    main()
