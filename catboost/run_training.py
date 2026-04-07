import argparse
import json
from pathlib import Path
import gc
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import mlflow

import torch

from modules import CatBoostImputer


def main(dataset: str):
    DATASET_NAME = dataset

    print(f"Running for {DATASET_NAME}")

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print("Device:", device)

    num_threads = torch.get_num_threads()
    print("Num threads:", num_threads)

    OUTPUT_PATH = Path(DATASET_NAME)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    DATA_PATH = Path("../data")

    TRAIN_PATH = DATA_PATH / DATASET_NAME / "train.csv"

    RANDOM_STATE = 0
    VALID_SPLIT = 0.2

    ITERATIONS = 1000
    LEARNING_RATE = 0.01
    DEPTH = 3

    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(RANDOM_STATE)

    df = pd.read_csv(TRAIN_PATH)
    df.head()

    COLUMNS_NAMES = df.columns.to_list()
    N_COLUMNS = len(COLUMNS_NAMES)

    with open("./metadata.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_metadata = config[DATASET_NAME]

    NUM_FEATURES = np.array(dataset_metadata["num_features"], dtype=np.int64)
    CAT_FEATURES = np.array(dataset_metadata["cat_features"], dtype=np.int64)
    MISSING_TOKENS = config["missing_tokens"]

    print("Names:", *COLUMNS_NAMES)
    print("Numeric features:", *NUM_FEATURES)
    print("Categorical features:", *CAT_FEATURES)

    def load_inputs():
        train_df = pd.read_csv(TRAIN_PATH)
        train_df.replace(MISSING_TOKENS, np.nan, inplace=True)
        train_array = train_df.to_numpy(dtype=object)

        train_array, valid_array = train_test_split(
            train_array, test_size=VALID_SPLIT, random_state=RANDOM_STATE
        )

        return train_array, valid_array

    train_array, valid_array = load_inputs()

    gc.collect()
    torch.cuda.empty_cache()

    model = CatBoostImputer(
        n_columns=N_COLUMNS,
        num_features=NUM_FEATURES,
        cat_features=CAT_FEATURES,
        columns_names=COLUMNS_NAMES,
        random_state=RANDOM_STATE,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        task_type=device,
    )

    mlflow.set_experiment(f"{DATASET_NAME}-catboost")
    with mlflow.start_run():
        model.fit(train_array, valid_array, verbose=ITERATIONS // 5)
        joblib.dump(
            model, OUTPUT_PATH / f"{model.__class__.__name__}_{DATASET_NAME}.joblib"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset)
