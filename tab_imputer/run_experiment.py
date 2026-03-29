import argparse
import subprocess

from pathlib import Path
from tqdm import tqdm
import joblib

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from modules import TabImputer, MLP, ResNet
from utils import TabularDataset


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_threads = torch.get_num_threads()
    print("Num threads:", num_threads)

    DATA_PATH = Path("../data")

    TEST_PATH = DATA_PATH / DATASET_NAME / "test.csv"

    RANDOM_STATE = 0
    MISSING_RATE = 0.2

    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(RANDOM_STATE)

    def load_test():
        test_df = pd.read_csv(TEST_PATH)
        test_df.replace([" ?", "?"], np.nan, inplace=True)
        test_array = test_df.to_numpy(dtype=object)

        encoder = joblib.load(f"{DATASET_NAME}/{DATASET_NAME}_encoder.joblib")

        test_dataset = TabularDataset(test_array, encoder)

        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=1, num_workers=num_threads
        )

        return test_loader

    test_loader = load_test()

    model = TabImputer(
        n_columns=test_loader.dataset.encoder.n_columns,
        num_features=torch.as_tensor(test_loader.dataset.encoder.num_features),
        cat_features=torch.as_tensor(test_loader.dataset.encoder.cat_features),
        classes_count=torch.as_tensor(
            test_loader.dataset.encoder.get_classes_count(with_unseen=True)
        ),
        d_model=32,
        embed_dim=16,
        encoder_output_dim=None,
        encoder_layers=4,
        head_layers=4,
        dropout=0.1,
        backbone=MLP,
    )
    model.load_state_dict(
        torch.load(
            f"{DATASET_NAME}/{model.__class__.__name__}_{DATASET_NAME}.pth",
            map_location=device,
        )
    )

    def make_loco_batch(x: torch.Tensor, mask: torch.Tensor):
        assert x.dim() == 1
        assert mask.dim() == 1
        assert x.shape == mask.shape

        n_columns = x.shape[0]

        x_batch = x.unsqueeze(0).repeat(n_columns, 1)
        mask_batch = mask.unsqueeze(0).repeat(n_columns, 1)

        idx = torch.arange(n_columns, device=x.device)
        mask_batch[idx, idx] = 0

        x_batch = x_batch * mask_batch
        return x_batch, mask_batch

    def evaluate_loco(loader: DataLoader, model: TabImputer, device: torch.device):
        assert loader.batch_size == 1

        model = model.to(device)

        out = []

        for x, mask in tqdm(loader, desc="Evaluating LOCO", leave=False):
            x = x.to(device)
            mask = mask.to(device)

            x = x[0]
            mask = mask[0]
            x_batch, mask_batch = make_loco_batch(x, mask)

            pred = model.predict(x_batch, mask_batch)
            pred = pred.detach().cpu().numpy()

            row = x.detach().cpu().numpy().copy()
            row[np.arange(len(row))] = np.diag(pred)

            out.append(row)

        out = np.stack(out, axis=0)
        out = loader.dataset.encoder.inverse_transform(out)

        columns_names = loader.dataset.encoder.columns_names
        if columns_names is None:
            columns_names = [f"{i}" for i in range(loader.dataset.encoder.n_columns)]

        df = pd.DataFrame(out, columns=columns_names)
        return df

    loco_df = evaluate_loco(test_loader, model, device)
    loco_df.to_csv(f"{DATASET_NAME}/tab_imputer_{DATASET_NAME}_loco.csv", index=False)

    def evaluate_mcar(
        loader: DataLoader,
        model: TabImputer,
        device: torch.device,
        missing_rate: float = 0.2,
        random_state: int = 0,
    ):
        model = model.to(device)

        rng = np.random.default_rng(random_state)
        mcar_mask = rng.random(loader.dataset.table.shape) < missing_rate

        out = []
        start = 0

        for x, mask in tqdm(loader, desc="Evaluating MCAR", leave=False):
            x = x.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]
            end = start + batch_size

            batch_mcar_mask_np = mcar_mask[start:end]
            batch_mcar_mask = torch.from_numpy(batch_mcar_mask_np).to(
                device=device, dtype=torch.bool
            )

            corrupted_mask = mask.clone()
            corrupted_mask[batch_mcar_mask] = 0

            x_input = x * corrupted_mask

            pred = model.predict(x_input, corrupted_mask)
            pred = pred.detach().cpu().numpy()

            batch_out = np.full(x.shape, fill_value=np.nan, dtype=np.float32)
            batch_out[batch_mcar_mask_np] = pred[batch_mcar_mask_np]

            out.append(batch_out)
            start = end

        out = np.concat(out, axis=0)
        out = loader.dataset.encoder.inverse_transform(out)

        columns_names = loader.dataset.encoder.columns_names
        if columns_names is None:
            columns_names = [f"{i}" for i in range(loader.dataset.encoder.n_columns)]

        df = pd.DataFrame(out, columns=columns_names)
        return df

    mcar_df = evaluate_mcar(test_loader, model, device, MISSING_RATE, RANDOM_STATE)
    mcar_df.to_csv(f"{DATASET_NAME}/tab_imputer_{DATASET_NAME}_mcar.csv", index=False)

    loco_pred_path = f"tab_imputer/{DATASET_NAME}/tab_imputer_{DATASET_NAME}_loco.csv"
    loco_out_dir = f"tab_imputer/{DATASET_NAME}/result_tab_imputer_{DATASET_NAME}_loco"
    run_evaluation(DATASET_NAME, loco_pred_path, loco_out_dir)

    mcar_pred_path = f"tab_imputer/{DATASET_NAME}/tab_imputer_{DATASET_NAME}_mcar.csv"
    mcar_out_dir = f"tab_imputer/{DATASET_NAME}/result_tab_imputer_{DATASET_NAME}_mcar"
    run_evaluation(DATASET_NAME, mcar_pred_path, mcar_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset)
