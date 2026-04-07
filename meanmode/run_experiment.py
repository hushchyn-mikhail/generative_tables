import argparse
import json

import subprocess
import joblib
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

from modules import MeanModeImputer


def run_evaluation(dataset_name: str, prediction_path: str, output_dir: str):
    subprocess.run(
        [
            "python",
            "test.py",
            "evaluate",
            "--dataset",
            dataset_name,
            "--prediction",
            prediction_path,
            "--output-dir",
            output_dir,
        ],
        cwd="..",
        check=True,
    )


def main(dataset: str):
    DATASET_NAME = dataset

    print(f"Running for {DATASET_NAME}")

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print("Device:", device)

    num_threads = torch.get_num_threads()
    print("Num threads:", num_threads)

    DATA_PATH = Path("../data")

    TEST_PATH = DATA_PATH / DATASET_NAME / "test.csv"

    RANDOM_STATE = 0
    MISSING_RATE = 0.2

    with open("./metadata.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    MISSING_TOKENS = config["missing_tokens"]

    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(RANDOM_STATE)

    def load_test():
        test_df = pd.read_csv(TEST_PATH)
        test_df.replace(MISSING_TOKENS, np.nan, inplace=True)
        test_array = test_df.to_numpy(dtype=object)

        return test_array

    test_array = load_test()

    model: MeanModeImputer = joblib.load(
        f"{DATASET_NAME}/MeanModeImputer_{DATASET_NAME}.joblib"
    )

    def make_loco_batch(x: np.ndarray):
        assert x.ndim == 1

        n_columns = x.shape[0]
        batch = np.repeat(np.expand_dims(x, axis=0), n_columns, 0)
        idx = np.arange(n_columns)
        batch[idx, idx] = np.nan
        return batch

    def evaluate_loco(X: np.ndarray, model: MeanModeImputer):
        out = []

        for row in tqdm(X, desc="Evaluating LOCO", leave=False):
            batch = make_loco_batch(row)

            pred = model.predict(batch)

            row = row.copy()
            row[np.arange(len(row))] = np.diag(pred)

            out.append(row)

        out = np.stack(out, axis=0)

        columns_names = model.columns_names
        if columns_names is None:
            columns_names = [f"{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(out, columns=columns_names)
        return df

    loco_df = evaluate_loco(test_array, model)
    loco_df.to_csv(f"{DATASET_NAME}/meanmode_{DATASET_NAME}_loco.csv", index=False)

    def evaluate_mcar(
        X: np.ndarray,
        model: MeanModeImputer,
        missing_rate: float = 0.2,
        random_state: int = 0,
    ):
        rng = np.random.default_rng(random_state)
        mask = rng.random(X.shape) < missing_rate

        cat_cols = model.cat_features
        num_cols = model.num_features

        print("Total masked:", mask.sum())
        print("Masked numeric:", mask[:, num_cols].sum())
        print("Masked categorical:", mask[:, cat_cols].sum())

        X_corrupted = X.astype(object).copy()
        X_corrupted[mask] = np.nan

        pred = model.predict(X_corrupted)

        out = np.full(X.shape, fill_value=None, dtype=object)
        out[mask] = pred[mask]

        for col in model.cat_features:
            col_mask = ~mask[:, col]
            out[col_mask, col] = "?"

        columns_names = model.columns_names
        if columns_names is None:
            columns_names = [f"{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(out, columns=columns_names)
        return df

    mcar_df = evaluate_mcar(
        test_array, model, missing_rate=MISSING_RATE, random_state=RANDOM_STATE
    )
    mcar_df.to_csv(f"{DATASET_NAME}/meanmode_{DATASET_NAME}_mcar.csv", index=False)

    loco_pred_path = f"meanmode/{DATASET_NAME}/meanmode_{DATASET_NAME}_loco.csv"
    loco_out_dir = f"meanmode/{DATASET_NAME}/result_meanmode_{DATASET_NAME}_loco"
    run_evaluation(DATASET_NAME, loco_pred_path, loco_out_dir)

    mcar_pred_path = f"meanmode/{DATASET_NAME}/meanmode_{DATASET_NAME}_mcar.csv"
    mcar_out_dir = f"meanmode/{DATASET_NAME}/result_meanmode_{DATASET_NAME}_mcar"
    run_evaluation(DATASET_NAME, mcar_pred_path, mcar_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset)
