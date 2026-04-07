import argparse

from pathlib import Path
import json
from tqdm import tqdm
import gc
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader

from modules import TabImputer, MLP, ResNet
from utils import TabularEncoder, TabularDataset

import mlflow


def main(dataset: str):
    DATASET_NAME = dataset

    print(f"Running for {DATASET_NAME}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_threads = torch.get_num_threads()
    print("Num threads:", num_threads)

    OUTPUT_PATH = Path(DATASET_NAME)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    DATA_PATH = Path("../data")

    TRAIN_PATH = DATA_PATH / DATASET_NAME / "train.csv"

    RANDOM_STATE = 0
    VALID_SPLIT = 0.2
    BATCH_SIZE = 2048

    def seed_everything(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    seed_everything(RANDOM_STATE)

    adult_df = pd.read_csv(TRAIN_PATH)
    adult_df.head()

    COLUMNS_NAMES = adult_df.columns.to_list()
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

        encoder = TabularEncoder(N_COLUMNS, NUM_FEATURES, CAT_FEATURES, COLUMNS_NAMES)
        encoder.fit(train_array)
        joblib.dump(encoder, OUTPUT_PATH / f"{DATASET_NAME}_encoder.joblib")

        train_dataset = TabularDataset(train_array, encoder)
        valid_dataset = TabularDataset(
            valid_array, encoder, is_valid=True, random_state=RANDOM_STATE
        )

        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=num_threads
        )
        valid_loader = DataLoader(
            valid_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=num_threads
        )

        return train_loader, valid_loader

    train_loader, valid_loader = load_inputs()

    def training_epoch(
        model: TabImputer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        loader: DataLoader,
        tqdm_desc: str,
    ):

        device = next(model.parameters()).device
        train_losses = []

        model.train()
        for x, mask in tqdm(loader, desc=tqdm_desc, leave=False):
            x = x.to(device)
            mask = mask.to(device)

            x_input, mask_input, dropped = model.transform(x, mask)

            optimizer.zero_grad()
            output = model(x_input, mask_input)
            loss = model.loss(output, x, dropped)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            mlflow.log_metric(
                "lr", float(scheduler.get_last_lr()[0]), step=scheduler.last_epoch
            )

            train_losses.append(loss.item())

        return np.mean(train_losses)

    @torch.no_grad()
    def validation_epoch(model: TabImputer, loader: DataLoader, tqdm_desc: str):
        assert loader.dataset.is_valid

        device = next(model.parameters()).device
        valid_losses = []

        model.eval()
        for x, mask, dropped in tqdm(loader, desc=tqdm_desc, leave=False):
            x = x.to(device)
            mask = mask.to(device)
            dropped = dropped.to(device)

            x_input = x * mask

            output = model(x_input, mask)
            loss = model.loss(output, x, dropped)

            valid_losses.append(loss.item())

        return np.mean(valid_losses)

    def train_model(
        model: TabImputer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        device: torch.device,
    ):

        mlflow.set_experiment(f"{DATASET_NAME}-tab-imputer")
        with mlflow.start_run():

            model = model.to(device)

            mlflow.log_params(
                {"model_class": model.__class__.__name__, "model_repr": model.repr}
            )

            mlflow.log_params(
                {
                    "model": model.__class__.__name__,
                    "num_epochs": num_epochs,
                    "optimizer": optimizer.__class__.__name__,
                    "scheduler": scheduler.__class__.__name__,
                    "device": str(device),
                }
            )

            best_loss = float("inf")
            best_path = OUTPUT_PATH / f"{model.__class__.__name__}_{DATASET_NAME}.pth"

            for epoch in tqdm(range(1, num_epochs + 1), desc="Epoch", leave=False):
                train_loss = training_epoch(
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    tqdm_desc=f"Training {epoch}/{num_epochs}",
                )
                valid_loss = validation_epoch(
                    model, valid_loader, tqdm_desc=f"Validating {epoch}/{num_epochs}"
                )

                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("valid_loss", valid_loss, step=epoch)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), best_path)

                mlflow.log_metric("best_valid_loss", best_loss)

        return model.state_dict()

    model = TabImputer(
        n_columns=train_loader.dataset.encoder.n_columns,
        num_features=torch.as_tensor(train_loader.dataset.encoder.num_features),
        cat_features=torch.as_tensor(train_loader.dataset.encoder.cat_features),
        classes_count=torch.as_tensor(
            train_loader.dataset.encoder.get_classes_count(with_unseen=True)
        ),
        d_model=32,
        embed_dim=16,
        encoder_output_dim=None,
        encoder_layers=4,
        head_layers=4,
        dropout=0.1,
        backbone=MLP,
    )

    gc.collect()
    torch.cuda.empty_cache()

    num_epochs = 600
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )

    history = train_model(
        model,
        optimizer,
        scheduler,
        train_loader,
        valid_loader,
        num_epochs=num_epochs,
        device=device,
    )

    print("Training is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    main(args.dataset)
