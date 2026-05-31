import os
import json
import numpy as np
import pandas as pd
import torch

from tabsyn.model import MLPDiffusion, Model
from tabsyn.model import MLPDiffusion, Model
from tabsyn.vae.model import Encoder_model
from tabsyn.latent_utils import split_num_cat_target, recover_data
from utils_train import preprocess

from tabsyn.diffusion_utils import sample_step, SIGMA_MIN, SIGMA_MAX, rho


def _set_device(gpu: int) -> str:
    if gpu != -1 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def _load_info(tabsyn_root: str, dataname: str) -> dict:
    info_path = os.path.join(tabsyn_root, "data", dataname, "info.json")
    with open(info_path, "r") as f:
        return json.load(f)


def _token_index_from_original_col(info: dict, orig_col: int) -> int:
    """
    Индекс токена (без CLS) для колонки orig_col в latent.

    Порядок токенов:
    - classification: [num_cols] + [target_cols] + [cat_cols]
    - regression:     [target_cols] + [num_cols] + [cat_cols]
    """
    num_cols = list(info["num_col_idx"])
    cat_cols = list(info["cat_col_idx"])
    tgt_cols = list(info["target_col_idx"])

    n_num = len(num_cols)
    n_tgt = len(tgt_cols)

    task = info["task_type"]
    is_classif = task in ("binclass", "multiclass")

    if is_classif:
        if orig_col in num_cols:
            return num_cols.index(orig_col)
        if orig_col in tgt_cols:
            return n_num + tgt_cols.index(orig_col)
        if orig_col in cat_cols:
            return n_num + n_tgt + cat_cols.index(orig_col)
        raise ValueError(f"Column {orig_col} not found in info lists.")
    else:
        if orig_col in tgt_cols:
            return tgt_cols.index(orig_col)
        if orig_col in num_cols:
            return n_tgt + num_cols.index(orig_col)
        if orig_col in cat_cols:
            return n_tgt + n_num + cat_cols.index(orig_col)
        raise ValueError(f"Column {orig_col} not found in info lists.")


def _build_latent_mask(info: dict, missing_cols_mask: np.ndarray, token_dim: int) -> torch.Tensor:
    """
    missing_cols_mask: shape (N, D_original), True => missing
    Возвращает mask_latent: shape (N, M*token_dim), 1 => observed, 0 => missing
    """
    N, D = missing_cols_mask.shape
    num_cols = list(info["num_col_idx"])
    cat_cols = list(info["cat_col_idx"])
    tgt_cols = list(info["target_col_idx"])
    M = len(num_cols) + len(cat_cols) + len(tgt_cols)

    token_mask = np.ones((N, M), dtype=np.float32)

    for orig_col in range(D):
        miss_rows = missing_cols_mask[:, orig_col]
        if not miss_rows.any():
            continue
        tok = _token_index_from_original_col(info, orig_col)
        token_mask[miss_rows, tok] = 0.0

    return torch.tensor(token_mask).repeat_interleave(token_dim, dim=1)


def _load_tabsyn_models(tabsyn_root: str, dataname: str, device: str, cat_decoder_type: str):
    data_dir = os.path.join(tabsyn_root, "data", dataname)
    info = _load_info(tabsyn_root, dataname)

    X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        data_dir, task_type=info["task_type"], inverse=True
    )

    vae_ckpt_dir = os.path.join(tabsyn_root, "tabsyn", "vae", "ckpt", dataname)
    encoder_path = os.path.join(vae_ckpt_dir, "encoder.pt")

    encoder = Encoder_model(
        num_layers=2,
        d_numerical=d_numerical,
        categories=categories,
        d_token=4,
        n_head=1,
        factor=32,
        bias=True,
    ).to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()

    diff_ckpt_dir = os.path.join(tabsyn_root, "tabsyn", "ckpt", dataname)
    diff_path = os.path.join(diff_ckpt_dir, "model.pt")

    train_z_path = os.path.join(vae_ckpt_dir, "train_z.npy")
    train_z = torch.tensor(np.load(train_z_path)).float()
    train_z = train_z[:, 1:, :]
    B, M, token_dim = train_z.shape
    in_dim = M * token_dim

    train_z_flat = train_z.reshape(B, in_dim)
    mean = train_z_flat.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    #diff_model = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
    diff_model = Model(
        denoise_fn=denoise_fn,
        hid_dim=in_dim,
        token_dim=token_dim,
    ).to(device)
    diff_model.load_state_dict(torch.load(diff_path, map_location=device))
    diff_model.eval()

    from tabsyn.vae.model import Decoder_model
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32, cat_decoder_type=cat_decoder_type).to(device)
    decoder_path = os.path.join(vae_ckpt_dir, "decoder.pt")

    
    pre_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    pre_decoder.eval()

    info["pre_decoder"] = pre_decoder
    info["token_dim"] = token_dim

    (X_train_num, X_test_num), (X_train_cat, X_test_cat) = X_num, X_cat
    X_train_num = np.asarray(X_train_num)
    X_train_cat = np.asarray(X_train_cat)
    X_test_num = np.asarray(X_test_num)
    X_test_cat = np.asarray(X_test_cat)
    
    if hasattr(diff_model.denoise_fn_D, "get_noise_scale"):
        scales = diff_model.denoise_fn_D.get_noise_scale().detach().cpu()
        print("Loaded block noise scales:", scales)
    else:
        print("This checkpoint/model does not expose blockwise noise scales.")

    return {
        "info": info,
        "encoder": encoder,
        "diff_net": diff_model.denoise_fn_D,
        "mean": mean.to(device),
        "token_dim": token_dim,
        "num_inverse": num_inverse,
        "cat_inverse": cat_inverse,
        "X_train_num": X_train_num,
        "X_train_cat": X_train_cat,
        "X_test_num": X_test_num,
        "X_test_cat": X_test_cat,
    }


@torch.no_grad()
def tabsyn_impute(
    tabsyn_root: str,
    dataname: str,
    missing_cols_mask: np.ndarray,
    gpu: int = 0,
    num_steps: int = 50,
    resample_steps: int = 10,
    seed: int = 1,
    cat_decoder_type: str = 'mlp'
) -> pd.DataFrame:
    """
    TabSyn imputation по маске missing_cols_mask.
    Возвращает DataFrame в оригинальном формате.
    """
    device = _set_device(gpu)
    torch.manual_seed(seed)
    np.random.seed(seed)

    bundle = _load_tabsyn_models(tabsyn_root, dataname, device, cat_decoder_type)
    info = bundle["info"]
    encoder = bundle["encoder"]
    net = bundle["diff_net"]
    mean = bundle["mean"]
    token_dim = bundle["token_dim"]

    X_train_num = bundle["X_train_num"]
    X_train_cat = bundle["X_train_cat"]
    X_test_num = bundle["X_test_num"].copy()
    X_test_cat = bundle["X_test_cat"].copy()

    N = X_test_num.shape[0]
    D_original = len(info["column_names"])
    assert missing_cols_mask.shape == (N, D_original)

    mask_latent = _build_latent_mask(info, missing_cols_mask, token_dim=token_dim).to(device)

    task = info["task_type"]
    is_classif = task in ("binclass", "multiclass")
    num_cols = list(info["num_col_idx"])
    cat_cols = list(info["cat_col_idx"])
    tgt_cols = list(info["target_col_idx"])
    n_tgt = len(tgt_cols)


    # classification:
    #   X_num = [num_cols]
    #   X_cat = [target_cols] + [cat_cols]
    #
    # regression:
    #   X_num = [target_cols] + [num_cols]
    #   X_cat = [cat_cols]

    if is_classif:
        orig_to_xnum = {c: i for i, c in enumerate(num_cols)}

        orig_to_xcat = {}
        for i, c in enumerate(tgt_cols):
            orig_to_xcat[c] = i
        for i, c in enumerate(cat_cols):
            orig_to_xcat[c] = n_tgt + i
    else:

        orig_to_xnum = {}
        for i, c in enumerate(tgt_cols):
            orig_to_xnum[c] = i
        for i, c in enumerate(num_cols):
            orig_to_xnum[c] = n_tgt + i

        orig_to_xcat = {c: i for i, c in enumerate(cat_cols)}

    rng = np.random.default_rng(seed)


    if X_train_num.size:
        num_means = np.nanmean(X_train_num, axis=0)
        for orig_c, j in orig_to_xnum.items():
            miss_rows = missing_cols_mask[:, orig_c]
            if miss_rows.any():
                X_test_num[miss_rows, j] = num_means[j]


    if X_train_cat.size:
        cat_max = X_train_cat.max(axis=0)
        for orig_c, j in orig_to_xcat.items():
            miss_rows = missing_cols_mask[:, orig_c]
            if miss_rows.any():
                hi = int(cat_max[j]) + 1
                X_test_cat[miss_rows, j] = rng.integers(0, hi, size=miss_rows.sum())

    x_num_t = torch.tensor(X_test_num).float().to(device)
    x_cat_t = torch.tensor(X_test_cat).long().to(device) if X_test_cat.size else None

    z_tokens = encoder(x_num_t, x_cat_t)
    z_tokens = z_tokens[:, 1:, :]
    N, M, td = z_tokens.shape
    z = z_tokens.reshape(N, M * td)

    z_norm = (z - mean) / 2.0

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    sigma_min = max(SIGMA_MIN, net.sigma_min)
    sigma_max = min(SIGMA_MAX, net.sigma_max)

    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    # baseline
    #x = z_norm + t_steps[0] * torch.randn_like(z_norm)
    x = z_norm + net.scale_noise(torch.randn_like(z_norm), t_steps[0])

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        for u in range(resample_steps):
            x_prop = sample_step(net, num_steps, i, t_cur, t_next, x)

            # baseline
            # eps_k = torch.randn_like(z_norm)
            # known_noisy = z_norm + t_next * eps_k
            eps_k = torch.randn_like(z_norm)
            known_noisy = z_norm + net.scale_noise(eps_k, t_next)

            # observed оставляем из исходного z_norm, missing берем из x_prop
            x = mask_latent * known_noisy + (1.0 - mask_latent) * x_prop

            if u < resample_steps - 1:
                # baseline
                #sigma_up = torch.sqrt(torch.clamp(t_cur**2 - t_next**2, min=0.0))
                #x = x + sigma_up * torch.randn_like(x)
                
                sigma_up = torch.sqrt(torch.clamp(t_cur**2 - t_next**2, min=0.0))
                x = x + net.scale_noise(torch.randn_like(x), sigma_up)
                
                

    z_hat = x * 2.0 + mean
    syn_data = z_hat.detach().cpu().numpy()

    syn_num, syn_cat, syn_target = split_num_cat_target(
        syn_data, info, bundle["num_inverse"], bundle["cat_inverse"], device
    )
    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = {int(k): v for k, v in info["idx_name_mapping"].items()}
    syn_df.rename(columns=idx_name_mapping, inplace=True)
    return syn_df