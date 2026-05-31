import argparse
import glob
import json
import pickle
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import src
from tabdiff.metrics import TabMetrics
from tabdiff.modules.main_modules import Model, UniModMLP
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from tabdiff.trainer import Trainer, recover_data, split_num_cat_target
from utils_train import TabDiffDataset


class DummyLogger:
    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


def build_trainer(
    repo_root: Path,
    dataname: str,
    device: str,
    exp_name: str | None,
    ckpt_path: str | None,
):
    curr_dir = repo_root / "tabdiff"
    info_path = repo_root / "data" / dataname / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    if exp_name is None:
        exp_name = "learnable_schedule"

    if ckpt_path is None:
        ckpt_candidates = sorted(
            glob.glob(str(curr_dir / "ckpt" / dataname / exp_name / "best_ema_model*"))
        )
        if not ckpt_candidates:
            raise FileNotFoundError(
                f"Checkpoint not found in {curr_dir / 'ckpt' / dataname / exp_name}. "
                "Train the model first or pass --ckpt_path explicitly."
            )
        ckpt_path = ckpt_candidates[-1]

    ckpt_path = str(Path(ckpt_path).resolve())
    config_path = Path(ckpt_path).parent / "config.pkl"
    if not config_path.exists():
        raise FileNotFoundError(f"config.pkl not found рядом с чекпоинтом: {config_path}")

    with open(config_path, "rb") as f:
        raw_config = pickle.load(f)

    data_dir = repo_root / "data" / dataname

    train_data = TabDiffDataset(
        dataname,
        str(data_dir),
        info,
        y_only=False,
        isTrain=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    test_data = TabDiffDataset(
        dataname,
        str(data_dir),
        info,
        y_only=False,
        isTrain=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=raw_config["train"]["main"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    d_numerical = train_data.d_numerical
    categories = train_data.categories

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()

    if raw_config["diffusion_params"]["scheduler"] != "power_mean_per_column":
        raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
        raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    model = Model(backbone, **raw_config["diffusion_params"]["edm_params"])
    model.to(device)

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **raw_config["diffusion_params"],
        device=device,
    )
    diffusion.to(device)

    synthetic_dir = repo_root / "synthetic" / dataname
    metrics = TabMetrics(
        str(synthetic_dir / "real.csv"),
        str(synthetic_dir / "test.csv"),
        str(synthetic_dir / "val.csv") if (synthetic_dir / "val.csv").exists() else None,
        info,
        device,
        metric_list=["density"],
    )

    trainer = Trainer(
        diffusion,
        train_loader,
        train_data,
        test_data,
        metrics,
        DummyLogger(),
        **raw_config["train"]["main"],
        sample_batch_size=raw_config["sample"]["batch_size"],
        num_samples_to_generate=None,
        model_save_path=str(Path(ckpt_path).parent),
        result_save_path=str(Path(ckpt_path).parent),
        device=device,
        ckpt_path=ckpt_path,
        y_only=False,
    )
    trainer.diffusion.eval()
    return trainer, info


def build_local_feature_maps(info: dict):
    """
    В TabDiff внутренний порядок не такой, как просто сортировка по idx_mapping.

    Для regression:
        x_num = [target] + numerical_features
        x_cat = categorical_features

    Для binclass / multiclass:
        x_num = numerical_features
        x_cat = [target] + categorical_features
    """
    task_type = info["task_type"]

    num_cols = list(info["num_col_idx"])
    cat_cols = list(info["cat_col_idx"])
    target_cols = list(info["target_col_idx"])

    if task_type == "regression":
        num_order = target_cols + num_cols
        cat_order = cat_cols
    else:
        num_order = num_cols
        cat_order = target_cols + cat_cols

    num_local_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(num_order)}
    cat_local_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(cat_order)}
    return num_local_map, cat_local_map


def impute_one_column(
    trainer: Trainer,
    info: dict,
    original_col_idx: int,
    resample_rounds: int,
    impute_condition: str,
    debug: bool = False,
):
    device = trainer.device
    dataset = trainer.dataset
    test_dataset = trainer.test_dataset

    d_numerical = dataset.d_numerical
    categories = dataset.categories

    num_local_map, cat_local_map = build_local_feature_maps(info)

    num_mask_idx = []
    cat_mask_idx = []

    if original_col_idx in num_local_map:
        num_mask_idx = [num_local_map[original_col_idx]]
        masked_block = "num"
    elif original_col_idx in cat_local_map:
        cat_mask_idx = [cat_local_map[original_col_idx]]
        masked_block = "cat"
    else:
        raise ValueError(
            f"Column {original_col_idx} not found in reconstructed num/cat/target maps"
        )

    X_train = dataset.X.to(device)
    X_test = deepcopy(test_dataset.X).to(device)

    x_num_train = X_train[:, :d_numerical]
    x_cat_train = X_train[:, d_numerical:].long()
    x_num_test = X_test[:, :d_numerical]
    x_cat_test = X_test[:, d_numerical:].long()

    if debug:
        print(
            f"[DEBUG] original_col_idx={original_col_idx}, "
            f"masked_block={masked_block}, "
            f"num_mask_idx={num_mask_idx}, cat_mask_idx={cat_mask_idx}"
        )

    # Маскируем только один столбец
    if num_mask_idx:
        avg = x_num_train[:, num_mask_idx].mean(0)
        x_num_test[:, num_mask_idx] = avg

    if cat_mask_idx:
        mask_token = torch.tensor(categories, dtype=x_cat_test.dtype, device=device)[cat_mask_idx]
        x_cat_test[:, cat_mask_idx] = mask_token
        if debug:
            print(
                "[DEBUG] first 10 masked categorical values:",
                x_cat_test[:10, cat_mask_idx[0]].detach().cpu().tolist(),
            )

    with torch.no_grad():
        syn_data = trainer.diffusion.sample_impute(
            x_num_test,
            x_cat_test,
            num_mask_idx=num_mask_idx,
            cat_mask_idx=cat_mask_idx,
            resample_rounds=resample_rounds,
            impute_condition=impute_condition,
            w_num=0.0,
            w_cat=0.0,
        )

    # Ограничиваем категориальные id допустимым диапазоном train
    if x_cat_train.shape[1] > 0 and syn_data[:, d_numerical:].shape[1] > 0:
        syn_cat_part = syn_data[:, d_numerical:].long()
        train_cat_part = x_cat_train.long()

        if syn_cat_part.shape[1] == train_cat_part.shape[1]:
            max_train = train_cat_part.max(dim=0).values.to(syn_cat_part.device)
            min_train = train_cat_part.min(dim=0).values.to(syn_cat_part.device)
            syn_cat_part = torch.maximum(syn_cat_part, min_train.unsqueeze(0))
            syn_cat_part = torch.minimum(syn_cat_part, max_train.unsqueeze(0))
            syn_data[:, d_numerical:] = syn_cat_part

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data,
        info,
        dataset.num_inverse,
        dataset.int_inverse,
        dataset.cat_inverse,
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Column-wise imputation on test.csv with a trained TabDiff model"
    )
    parser.add_argument("--dataname", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--exp_name", default="learnable_schedule")
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resample_rounds", type=int, default=1)
    parser.add_argument("--impute_condition", choices=["x_t", "x_0"], default="x_t")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu != -1 and torch.cuda.is_available() else "cpu"
    repo_root = Path(args.repo_root).resolve()

    trainer, info = build_trainer(
        repo_root=repo_root,
        dataname=args.dataname,
        device=device,
        exp_name=args.exp_name,
        ckpt_path=args.ckpt_path,
    )

    test_csv = repo_root / "data" / args.dataname / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Не найден test.csv: {test_csv}")

    prediction_df = pd.read_csv(test_csv)
    ordered_columns = list(prediction_df.columns)

    for col_idx, col_name in enumerate(ordered_columns):
        print(f"Imputing column {col_idx}: {col_name}")
        syn_df = impute_one_column(
            trainer=trainer,
            info=info,
            original_col_idx=col_idx,
            resample_rounds=args.resample_rounds,
            impute_condition=args.impute_condition,
            debug=args.debug,
        )
        prediction_df[col_name] = syn_df[col_name].values

        if args.debug:
            same_ratio = (
                prediction_df[col_name].astype(str).values
                == pd.read_csv(test_csv)[col_name].astype(str).values
            ).mean()
            print(f"[DEBUG] same_ratio for {col_name}: {same_ratio:.6f}")

    if args.output is None:
        out_path = repo_root / "impute" / args.dataname / args.exp_name / "prediction.csv"
    else:
        out_path = Path(args.output).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(out_path, index=False)
    print(f"Saved prediction file to: {out_path}")


if __name__ == "__main__":
    main()