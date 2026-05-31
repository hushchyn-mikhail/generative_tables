import os
import glob
import json
from copy import deepcopy
import numpy as np
import pandas as pd
import torch

import src  # из submodules/tabdiff/src
from utils_train import preprocess  # из submodules/tabdiff/utils_train.py

from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion


def _set_device(gpu: int) -> str:
    if gpu != -1 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def _load_info(tabsyn_root: str, dataname: str) -> dict:
    info_path = os.path.join(tabsyn_root, "data", dataname, "info.json")
    with open(info_path, "r") as f:
        return json.load(f)


def _default_exp_name(non_learnable_schedule: bool) -> str:
    return "non_learnable_schedule" if non_learnable_schedule else "learnable_schedule"


def _infer_ckpt_path(tabdiff_root: str, dataname: str, exp_name: str) -> str:
    ckpt_parent = os.path.join(tabdiff_root, "tabdiff", "ckpt", dataname, exp_name)
    cand = glob.glob(os.path.join(ckpt_parent, "best_ema_model*.pt"))
    if not cand:
        # иногда могут сохранять просто model_*.pt
        cand = glob.glob(os.path.join(ckpt_parent, "model_*.pt"))
    if not cand:
        raise FileNotFoundError(
            f"Cannot find checkpoint in {ckpt_parent}. "
            f"Train TabDiff first (or pass --ckpt_path explicitly)."
        )
    cand.sort()
    return cand[-1]


def _build_tabdiff_model(
    tabdiff_root: str,
    info: dict,
    dataset_dir: str,
    device: str,
    non_learnable_schedule: bool,
    config_pkl_path: str | None,
):

    # 1) load config
    if config_pkl_path is not None and os.path.exists(config_pkl_path):
        import pickle
        with open(config_pkl_path, "rb") as f:
            raw_config = pickle.load(f)
    else:
        config_path = os.path.join(tabdiff_root, "tabdiff", "configs", "tabdiff_configs.toml")
        raw_config = src.load_config(config_path)

    # 2) preprocess -> чтобы узнать d_numerical и categories 
    X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse = preprocess(
        dataset_dir,
        task_type=info["task_type"],
        inverse=True,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    categories = np.array(categories)

    # 3) build backbone + diffusion model
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()  # +1 for [MASK] category

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    denoise_fn = Model(backbone, **raw_config["diffusion_params"]["edm_params"]).to(device)

    # learnable schedules 
    if (not non_learnable_schedule):
        raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
        raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=denoise_fn,
        y_only_model=None,
        device=device,
        **raw_config["diffusion_params"],
    ).to(device)

    (X_train_num, X_test_num), (X_train_cat, X_test_cat) = X_num, X_cat

    bundle = {
        "raw_config": raw_config,
        "diffusion": diffusion,
        "d_numerical": d_numerical,
        "categories": categories,
        "num_inverse": num_inverse,
        "int_inverse": int_inverse,
        "cat_inverse": cat_inverse,
        "X_train_num": np.asarray(X_train_num),
        "X_train_cat": np.asarray(X_train_cat),
        "X_test_num": np.asarray(X_test_num),
        "X_test_cat": np.asarray(X_test_cat),
    }
    return bundle


def _origcol_to_tabdiff_space(info: dict, orig_col_idx: int) -> tuple[str, int]:
    """
    Возвращает ("num", j) или ("cat", j), где j — индекс колонки в X_num или X_cat,
    как это ожидает TabDiff.

    Важно: TabDiff в preprocess делает concat target в:
      - classification: X_cat = [y] + X_cat
      - regression:     X_num = [y] + X_num
    """
    num_cols = list(info["num_col_idx"])
    cat_cols = list(info["cat_col_idx"])
    tgt_cols = list(info["target_col_idx"])
    task = info["task_type"]

    if task in ("binclass", "multiclass"):
        # X_num: num_cols
        # X_cat: [target(s)] + cat_cols
        if orig_col_idx in num_cols:
            return ("num", num_cols.index(orig_col_idx))
        if orig_col_idx in tgt_cols:
            return ("cat", tgt_cols.index(orig_col_idx))
        if orig_col_idx in cat_cols:
            return ("cat", len(tgt_cols) + cat_cols.index(orig_col_idx))
        raise ValueError(f"orig_col_idx={orig_col_idx} not found in info lists.")
    else:
        # regression
        # X_num: [target(s)] + num_cols
        # X_cat: cat_cols
        if orig_col_idx in tgt_cols:
            return ("num", tgt_cols.index(orig_col_idx))
        if orig_col_idx in num_cols:
            return ("num", len(tgt_cols) + num_cols.index(orig_col_idx))
        if orig_col_idx in cat_cols:
            return ("cat", cat_cols.index(orig_col_idx))
        raise ValueError(f"orig_col_idx={orig_col_idx} not found in info lists.")


@torch.no_grad()
def _split_num_cat_target(syn_data: np.ndarray, info: dict, num_inverse, int_inverse, cat_inverse):

    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    syn_num = num_inverse(syn_num).astype(np.float32)
    syn_num = int_inverse(syn_num).astype(np.float32)
    syn_cat = cat_inverse(syn_cat)

    if task_type == "regression":
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target


def _recover_data(syn_num: np.ndarray, syn_cat: np.ndarray, syn_target: np.ndarray, info: dict) -> pd.DataFrame:

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = {int(k): v for k, v in info["idx_mapping"].items()}

    syn_df = pd.DataFrame()
    D = len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)

    for i in range(D):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
        else:
            syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df


@torch.no_grad()
def tabdiff_impute(
    tabsyn_root: str,
    tabdiff_root: str,
    dataname: str,
    missing_cols_mask: np.ndarray,  # (N, D_original) True => missing
    gpu: int = 0,
    resample_rounds: int = 1,
    impute_condition: str = "x_t",
    seed: int = 1,
    ckpt_path: str | None = None,
    exp_name: str | None = None,
    non_learnable_schedule: bool = False,
) -> pd.DataFrame:
    """без CFG guidance -> y_only_model=None, w_num=0, w_cat=0.
    """
    device = _set_device(gpu)
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset_dir = os.path.join(tabsyn_root, "data", dataname)
    info = _load_info(tabsyn_root, dataname)

    test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))
    cols = list(test_df.columns)

    N, D_original = missing_cols_mask.shape

    # 1) ckpt + config
    if exp_name is None:
        exp_name = _default_exp_name(non_learnable_schedule)
    if ckpt_path is None:
        ckpt_path = _infer_ckpt_path(tabdiff_root, dataname, exp_name)

    config_pkl_path = os.path.join(os.path.dirname(ckpt_path), "config.pkl")

    # 2) build diffusion + load weights
    bundle = _build_tabdiff_model(
        tabdiff_root=tabdiff_root,
        info=info,
        dataset_dir=dataset_dir,
        device=device,
        non_learnable_schedule=non_learnable_schedule,
        config_pkl_path=config_pkl_path if os.path.exists(config_pkl_path) else None,
    )
    diffusion: UnifiedCtimeDiffusion = bundle["diffusion"]
    d_numerical = bundle["d_numerical"]
    categories = bundle["categories"]

    # load weights
    state = torch.load(ckpt_path, map_location=device)
    diffusion._denoise_fn.load_state_dict(state["denoise_fn"])
    diffusion.num_schedule.load_state_dict(state["num_schedule"])
    diffusion.cat_schedule.load_state_dict(state["cat_schedule"])
    diffusion.eval()

    # 3) which columns are missing (globally)
    missing_cols = [j for j in range(D_original) if missing_cols_mask[:, j].any()]
    num_mask_idx = []
    cat_mask_idx = []
    for j in missing_cols:
        kind, idx = _origcol_to_tabdiff_space(info, j)
        if kind == "num":
            num_mask_idx.append(idx)
        else:
            cat_mask_idx.append(idx)
    num_mask_idx = sorted(set(num_mask_idx))
    cat_mask_idx = sorted(set(cat_mask_idx))

    # 4) prepare x_0 in TabDiff space (transformed)
    X_train_num = torch.tensor(bundle["X_train_num"], dtype=torch.float32, device=device)
    X_train_cat = torch.tensor(bundle["X_train_cat"], dtype=torch.long, device=device)

    X_test_num = torch.tensor(bundle["X_test_num"], dtype=torch.float32, device=device)
    X_test_cat = torch.tensor(bundle["X_test_cat"], dtype=torch.long, device=device)

    x_num = deepcopy(X_test_num)
    x_cat = deepcopy(X_test_cat)

    # Apply “mask at x0” as TabDiff expects:
    # - numerical masked cols -> set to train mean
    # - categorical masked cols -> set to [MASK] token id = categories[col]
    if len(num_mask_idx) > 0:
        avg = X_train_num[:, num_mask_idx].mean(dim=0)
        x_num[:, num_mask_idx] = avg

    if len(cat_mask_idx) > 0:
        mask_tokens = torch.tensor(categories, device=device, dtype=x_cat.dtype)[cat_mask_idx]
        x_cat[:, cat_mask_idx] = mask_tokens

    # 5) sample imputed (NO guidance)
    syn = diffusion.sample_impute(
        x_num=x_num,
        x_cat=x_cat,
        num_mask_idx=num_mask_idx,
        cat_mask_idx=cat_mask_idx,
        resample_rounds=resample_rounds,
        impute_condition=impute_condition,
        w_num=0.0,
        w_cat=0.0,
    ).cpu().numpy()

    # small patch : если в test есть unseen categories и inverse ломается
    if bundle["X_train_cat"].shape[1] > 0:
        train_cat_max = torch.tensor(bundle["X_train_cat"]).max(dim=0).values.numpy()
        syn_cat_part = syn[:, d_numerical:]
        if syn_cat_part.shape[1] > 0 and (syn_cat_part.max(axis=0) > train_cat_max).any():
            # заменяем “опасные” cat на train-значения (кроме первого cat-столбца, если там target в classification)
            syn[:, d_numerical:] = bundle["X_train_cat"][: syn.shape[0], :]

    # 6) recover to original domain
    syn_num, syn_cat, syn_target = _split_num_cat_target(
        syn, info, bundle["num_inverse"], bundle["int_inverse"], bundle["cat_inverse"]
    )
    syn_df = _recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)

    
    for j, col in enumerate(cols):
        obs_rows = ~missing_cols_mask[:, j]
        if obs_rows.any():
            syn_df.loc[obs_rows, col] = test_df.loc[obs_rows, col].values

    
    syn_df = syn_df[cols]
    return syn_df
