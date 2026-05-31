# # experiments/tabcsdi_tabsyn_onehot/eval_test2_tabcsdi_tabsyn_onehot.py

# import argparse
# import os
# import json
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import yaml

# import os, sys

# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# TABCSDI_DIR = os.path.join(PROJECT_ROOT, "submodules", "TabCSDI")

# # чтобы работали импорты `from src...`
# sys.path.insert(0, TABCSDI_DIR)

# # чтобы относительные пути вроде "config/..." работали всегда
# os.chdir(TABCSDI_DIR)

# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score, f1_score, recall_score

# from src.main_model_table import TabCSDI
# from dataset_tabsyn_onehot_generic import get_dataloader_tabsyn_onehot


# # -----------------------------
# # Metrics + sanitization helpers
# # -----------------------------
# def sanitize_num(arr):
#     s = pd.Series(arr).replace(
#         ["", " ", "nan", "NaN", "None", "NULL", "null", "?", "-", "—"],
#         np.nan
#     )
#     return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


# def sanitize_cat(arr):
#     """
#     Универсально для категориальных:
#     - (n,1) -> (n,)
#     - (n,k) logits/one-hot -> argmax
#     - затем в строки
#     """
#     a = np.asarray(arr)

#     if a.ndim > 1 and a.shape[1] == 1:
#         a = a.reshape(-1)

#     if a.ndim == 2 and a.shape[1] > 1:
#         a = a.argmax(axis=1)

#     return a.astype(str)


# def _pair_str(y_true, y_pred):
#     yt = sanitize_cat(y_true)
#     yp = sanitize_cat(y_pred)

#     bad_tokens = {"nan", "NaN", "None", "NULL", "null", "?", "-", "—", ""}
#     m = np.ones(len(yt), dtype=bool)
#     for tok in bad_tokens:
#         m &= (yt != tok) & (yp != tok)

#     return yt[m], yp[m]


# def rmse(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     if len(y_true) == 0:
#         return float("nan")
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# def mae(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     if len(y_true) == 0:
#         return float("nan")
#     return float(np.mean(np.abs(y_true - y_pred)))


# def cat_acc(y_true, y_pred):
#     yt, yp = _pair_str(y_true, y_pred)
#     if yt.size == 0:
#         return float("nan")
#     return float(accuracy_score(yt, yp))


# def cat_f1(y_true, y_pred):
#     yt, yp = _pair_str(y_true, y_pred)
#     if yt.size == 0:
#         return float("nan")
#     return float(f1_score(yt, yp, average="macro", zero_division=0))


# def cat_recall(y_true, y_pred):
#     yt, yp = _pair_str(y_true, y_pred)
#     if yt.size == 0:
#         return float("nan")
#     return float(recall_score(yt, yp, average="macro", zero_division=0))


# def bootstrap_mean_sigma(metric_fn, y_true, y_pred, row_mask, B=300, seed=0, ddof=1):
#     """
#     Bootstrap mean and sigma (std) of metric over rows.
#     row_mask: boolean mask (N,) -- какие строки участвуют в оценке
#     Returns: (mean, sigma)
#     """
#     rng = np.random.default_rng(seed)
#     n = len(y_true)
#     vals = []

#     for _ in range(B):
#         idx = rng.integers(0, n, size=n)
#         m = row_mask[idx]
#         if m.sum() == 0:
#             continue

#         v = metric_fn(y_true[idx][m], y_pred[idx][m])

#         if v is None:
#             continue
#         try:
#             v = float(v)
#         except Exception:
#             continue
#         if not np.isfinite(v):
#             continue

#         vals.append(v)

#     if len(vals) == 0:
#         return float("nan"), float("nan")

#     vals = np.asarray(vals, dtype=float)
#     mu = float(vals.mean())
#     sigma = float(vals.std(ddof=ddof)) if vals.size > 1 else float("nan")
#     return mu, sigma


# # -----------------------------
# # Dataset for leave-one-column-out
# # -----------------------------
# class ColumnDropTestDataset(Dataset):
#     def __init__(self, values, base_observed_mask, drop_indices):
#         self.values = values.astype(np.float32)
#         self.observed_mask = base_observed_mask.astype(np.float32)
#         self.drop_indices = list(drop_indices)
#         self.L = self.values.shape[1]

#         # gt_mask = what is given to model as condition mask
#         # 1 -> observed, 0 -> hidden/imputed
#         self.gt_mask = self.observed_mask.copy()
#         self.gt_mask[:, self.drop_indices] = 0.0  # полностью скрываем выбранный блок колонки

#     def __len__(self):
#         return self.values.shape[0]

#     def __getitem__(self, idx):
#         return {
#             "observed_data": torch.from_numpy(self.values[idx]),
#             "observed_mask": torch.from_numpy(self.observed_mask[idx]),
#             "gt_mask": torch.from_numpy(self.gt_mask[idx]),
#             "timepoints": torch.arange(self.L),
#         }


# def save_progress(results, outdir, filename):
#     os.makedirs(outdir, exist_ok=True)
#     df = pd.DataFrame(results)
#     out_csv = os.path.join(outdir, filename)
#     df.to_csv(out_csv, index=False)
#     return out_csv


# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataname", type=str, required=True)
#     parser.add_argument("--tabsyn_root", type=str, required=True)
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
#     parser.add_argument("--device", default="cuda")
#     parser.add_argument("--seed", type=int, default=1)
#     parser.add_argument("--nsample", type=int, default=10)
#     parser.add_argument("--eval_batch_size", type=int, default=32)
#     parser.add_argument("--bootstrap", type=int, default=300)
#     parser.add_argument("--outdir", type=str, default="./results_test2")
#     parser.add_argument("--limit_cols", type=int, default=None)
#     args = parser.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # 1) config
#     with open(os.path.join("config", args.config), "r") as f:
#         config = yaml.safe_load(f)

#     # Test2: случайных масок нет, маскируем колонку вручную
#     config["model"]["test_missing_ratio"] = 0.0

#     # 2) построить/закешировать one-hot представление с missing_ratio=0.0
#     _tr, _va, _te = get_dataloader_tabsyn_onehot(
#         tabsyn_root=args.tabsyn_root,
#         dataname=args.dataname,
#         seed=args.seed,
#         batch_size=args.eval_batch_size,
#         missing_ratio=0.0,
#         val_ratio=0.1,
#     )

#     cache_dir = f"./data_{args.dataname}_tabsyn_onehot"
#     cache_file = os.path.join(cache_dir, f"{args.dataname}_tabsyn_onehot_miss0.0_seed{args.seed}.pk")
#     if not os.path.exists(cache_file):
#         raise FileNotFoundError(f"Cache file not found: {cache_file}")

#     with open(cache_file, "rb") as f:
#         payload = pickle.load(f)

#     test_vals = payload["test_vals"]
#     test_obs_mask = payload["test_obs_mask"]
#     ordered_names = payload.get("ordered_names", None)

#     with open(os.path.join(cache_dir, "transformed_columns.pk"), "rb") as f:
#         cont_cols, saved_cat_dict = pickle.load(f)

#     # 3) группы колонок в one-hot пространстве
#     groups = []

#     # numeric columns are 1-dim each in encoded space
#     for idx in cont_cols:
#         name = ordered_names[idx] if ordered_names is not None else f"cont_{idx}"
#         groups.append({
#             "name": str(name),
#             "type": "numeric",
#             "drop_indices": [int(idx)],
#             "orig_pos": int(idx),
#         })

#     # categorical columns are blocks in one-hot space
#     for key, block in saved_cat_dict.items():
#         orig_pos = int(key)
#         name = ordered_names[orig_pos] if ordered_names is not None else f"cat_{key}"
#         groups.append({
#             "name": str(name),
#             "type": "categorical",
#             "drop_indices": list(map(int, block)),
#             "orig_pos": orig_pos,
#         })

#     # восстановим исходный порядок колонок в reordered table
#     groups = sorted(groups, key=lambda g: g["orig_pos"])

#     if args.limit_cols is not None:
#         groups = groups[: args.limit_cols]

#     # 4) model
#     model = TabCSDI(config, args.device).to(args.device)
#     sd = torch.load(args.model_path, map_location=args.device)
#     model.load_state_dict(sd)
#     model.eval()

#     results = []

#     # 5) loop over columns
#     for gi, g in enumerate(groups):
#         col_name = g["name"]
#         col_type = g["type"]
#         drop_idx = g["drop_indices"]

#         print(
#             f"[{gi+1}/{len(groups)}] Drop column: {col_name} "
#             f"(type={col_type}, encoded_dims={len(drop_idx)})",
#             flush=True,
#         )

#         ds = ColumnDropTestDataset(test_vals, test_obs_mask, drop_idx)
#         loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False)

#         preds, trues, row_masks = [], [], []

#         with torch.no_grad():
#             for batch in loader:
#                 # Важно: используем evaluate как в ваших предыдущих скриптах
#                 samples, observed_data, target_mask, observed_mask, _tp = model.evaluate(batch, args.nsample)

#                 # samples: (B, nsample, K, L) или близкая форма -> приводим как в ваших скриптах
#                 samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, 1)
#                 med = samples.median(dim=1).values.squeeze(-1)  # (B, L)

#                 pred_batch = med.detach().cpu().numpy()
#                 true_batch = observed_data.squeeze(1).detach().cpu().numpy()

#                 # Строка участвует, если в исходных данных эта колонка (хотя бы первый индекс блока) была наблюдаемой
#                 # (в one-hot блоке маска дублируется, поэтому первого индекса достаточно)
#                 row_mask_batch = observed_mask.squeeze(1).detach().cpu().numpy().astype(bool)[:, drop_idx[0]]

#                 preds.append(pred_batch)
#                 trues.append(true_batch)
#                 row_masks.append(row_mask_batch)

#         pred_all = np.concatenate(preds, axis=0)       # (N, L)
#         true_all = np.concatenate(trues, axis=0)       # (N, L)
#         row_mask = np.concatenate(row_masks, axis=0)   # (N,)

#         # -------------------------
#         # Numeric column
#         # -------------------------
#         if col_type == "numeric":
#             block = drop_idx  # длина 1
#             y_true_raw = true_all[:, block]
#             y_pred_raw = pred_all[:, block]

#             if y_true_raw.ndim == 2 and y_true_raw.shape[1] == 1:
#                 y_true_raw = y_true_raw[:, 0]
#             if y_pred_raw.ndim == 2 and y_pred_raw.shape[1] == 1:
#                 y_pred_raw = y_pred_raw[:, 0]

#             yt = sanitize_num(y_true_raw)
#             yp = sanitize_num(y_pred_raw)

#             # Пропускаем только мусорные значения (nan/inf), а не нули:
#             # 0 может быть валидным значением для числовой колонки.
#             valid = row_mask & np.isfinite(yt) & np.isfinite(yp)

#             if valid.sum() == 0:
#                 rmse_val = mae_val = float("nan")
#                 rmse_bs_mean = rmse_bs_sigma = float("nan")
#                 mae_bs_mean = mae_bs_sigma = float("nan")
#             else:
#                 yt_v = yt[valid]
#                 yp_v = yp[valid]

#                 rmse_val = rmse(yt_v, yp_v)
#                 mae_val = mae(yt_v, yp_v)

#                 # bootstrap поверх уже отфильтрованных рядов
#                 row_all = np.ones(len(yt_v), dtype=bool)

#                 rmse_bs_mean, rmse_bs_sigma = bootstrap_mean_sigma(
#                     rmse, yt_v, yp_v, row_all,
#                     B=args.bootstrap, seed=args.seed + 1000 + gi
#                 )
#                 mae_bs_mean, mae_bs_sigma = bootstrap_mean_sigma(
#                     mae, yt_v, yp_v, row_all,
#                     B=args.bootstrap, seed=args.seed + 2000 + gi
#                 )

#             results.append({
#                 "column": col_name,
#                 "type": "numeric",
#                 "metric": "RMSE",
#                 "value": rmse_val,
#                 "bs_mean": rmse_bs_mean,
#                 "bs_sigma": rmse_bs_sigma,
#             })
#             results.append({
#                 "column": col_name,
#                 "type": "numeric",
#                 "metric": "MAE",
#                 "value": mae_val,
#                 "bs_mean": mae_bs_mean,
#                 "bs_sigma": mae_bs_sigma,
#             })

#         # -------------------------
#         # Categorical column (one-hot block)
#         # -------------------------
#         else:
#             block = drop_idx

#             # one-hot / logits -> class id
#             y_true = np.argmax(true_all[:, block], axis=1)
#             y_pred = np.argmax(pred_all[:, block], axis=1)

#             # Базовые метрики на всех валидных row_mask
#             if row_mask.sum() == 0:
#                 acc_val = f1_val = rec_val = float("nan")
#             else:
#                 acc_val = cat_acc(y_true[row_mask], y_pred[row_mask])
#                 f1_val = cat_f1(y_true[row_mask], y_pred[row_mask])
#                 rec_val = cat_recall(y_true[row_mask], y_pred[row_mask])

#             # Bootstrap mean + sigma
#             acc_bs_mean, acc_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: cat_acc(a, b),
#                 y_true, y_pred, row_mask,
#                 B=args.bootstrap, seed=args.seed + 3000 + gi
#             )
#             f1_bs_mean, f1_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: cat_f1(a, b),
#                 y_true, y_pred, row_mask,
#                 B=args.bootstrap, seed=args.seed + 4000 + gi
#             )
#             rec_bs_mean, rec_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: cat_recall(a, b),
#                 y_true, y_pred, row_mask,
#                 B=args.bootstrap, seed=args.seed + 5000 + gi
#             )

#             results.append({
#                 "column": col_name,
#                 "type": "categorical",
#                 "metric": "Accuracy",
#                 "value": acc_val,
#                 "bs_mean": acc_bs_mean,
#                 "bs_sigma": acc_bs_sigma,
#             })
#             results.append({
#                 "column": col_name,
#                 "type": "categorical",
#                 "metric": "F1_macro",
#                 "value": f1_val,
#                 "bs_mean": f1_bs_mean,
#                 "bs_sigma": f1_bs_sigma,
#             })
#             results.append({
#                 "column": col_name,
#                 "type": "categorical",
#                 "metric": "Recall_macro",
#                 "value": rec_val,
#                 "bs_mean": rec_bs_mean,
#                 "bs_sigma": rec_bs_sigma,
#             })

#         # сохраняем прогресс после каждой колонки
#         out_csv_progress = save_progress(
#             results, args.outdir, filename=f"{args.dataname}_test2_per_column_metrics_sigma.csv"
#         )
#         print(f"Progress saved: {out_csv_progress}", flush=True)

#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()

#     # 6) финальные файлы
#     df = pd.DataFrame(results)
#     out_csv = os.path.join(args.outdir, f"{args.dataname}_test2_per_column_metrics_sigma.csv")
#     df.to_csv(out_csv, index=False)

#     # агрегаты по типу колонки и метрике
#     avg_rows = []
#     for t in ["numeric", "categorical"]:
#         df_t = df[df["type"] == t]
#         if len(df_t) == 0:
#             continue

#         for metric in sorted(df_t["metric"].unique()):
#             part = df_t[df_t["metric"] == metric]
#             avg_rows.append({
#                 "dataset": args.dataname,
#                 "type": t,
#                 "metric": metric,
#                 "mean_over_columns": float(part["value"].dropna().mean()) if part["value"].notna().any() else float("nan"),
#                 "bootstrap_mean_over_columns": float(part["bs_mean"].dropna().mean()) if part["bs_mean"].notna().any() else float("nan"),
#                 "bootstrap_sigma_mean_over_columns": float(part["bs_sigma"].dropna().mean()) if part["bs_sigma"].notna().any() else float("nan"),
#             })

#     df_avg = pd.DataFrame(avg_rows)
#     out_avg = os.path.join(args.outdir, f"{args.dataname}_test2_avg_by_type_sigma.csv")
#     df_avg.to_csv(out_avg, index=False)

#     summary = {
#         "args": vars(args),
#         "dataset": args.dataname,
#         "n_columns_evaluated": len(groups),
#         "outputs": {
#             "per_column_csv": out_csv,
#             "avg_by_type_csv": out_avg,
#         },
#     }
#     out_json = os.path.join(args.outdir, f"{args.dataname}_test2_summary_sigma.json")
#     with open(out_json, "w") as f:
#         json.dump(summary, f, indent=2)

#     print("Saved:", out_csv)
#     print("Saved:", out_avg)
#     print("Saved:", out_json)


# if __name__ == "__main__":
#     main()

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import yaml
import sys
import datetime


import types
shim = types.ModuleType("pandas.core.indexes.numeric")
shim.Int64Index = pd.Index
shim.UInt64Index = pd.Index
shim.Float64Index = pd.Index
sys.modules["pandas.core.indexes.numeric"] = shim


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TABCSDI_DIR = os.path.join(PROJECT_ROOT, "submodules", "TabCSDI")

sys.path.insert(0, TABCSDI_DIR)
os.chdir(TABCSDI_DIR)

from torch.utils.data import Dataset, DataLoader
from src.main_model_table import TabCSDI
from dataset_tabsyn_onehot_generic import get_dataloader_tabsyn_onehot


class ColumnDropTestDataset(Dataset):
    def __init__(self, values, base_observed_mask, drop_indices):
        self.values = values.astype(np.float32)
        self.observed_mask = base_observed_mask.astype(np.float32)
        self.drop_indices = list(drop_indices)
        self.L = self.values.shape[1]

        self.gt_mask = self.observed_mask.copy()
        self.gt_mask[:, self.drop_indices] = 0.0

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return {
            "observed_data": torch.from_numpy(self.values[idx]),
            "observed_mask": torch.from_numpy(self.observed_mask[idx]),
            "gt_mask": torch.from_numpy(self.gt_mask[idx]),
            "timepoints": torch.arange(self.L),
        }


def build_groups(cont_cols, saved_cat_dict, ordered_names):
    groups = []

    for idx in cont_cols:
        name = ordered_names[idx] if ordered_names is not None else f"cont_{idx}"
        groups.append({
            "name": str(name),
            "type": "numeric",
            "drop_indices": [int(idx)],
            "orig_pos": int(idx),
        })

    for key, block in saved_cat_dict.items():
        orig_pos = int(key)
        name = ordered_names[orig_pos] if ordered_names is not None else f"cat_{key}"
        groups.append({
            "name": str(name),
            "type": "categorical",
            "drop_indices": list(map(int, block)),
            "orig_pos": orig_pos,
        })

    groups = sorted(groups, key=lambda g: g["orig_pos"])
    return groups


def load_raw_data_and_order(tabsyn_root, dataname):
    data_dir = os.path.join(tabsyn_root, "data", dataname)

    info_path = os.path.join(data_dir, "info.json")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Не найден info.json: {info_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Не найден train.csv: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Не найден test.csv: {test_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    train_df_orig = pd.read_csv(train_path)
    test_df_orig = pd.read_csv(test_path)

    num_col_idx = list(info["num_col_idx"])
    cat_col_idx = list(info["cat_col_idx"])
    target_col_idx = list(info["target_col_idx"])
    col_names = list(info["column_names"])
    task_type = info["task_type"]

    is_classif = task_type in ("binclass", "multiclass")

    if is_classif:
        cont_idx = num_col_idx
        cat_all_idx = cat_col_idx + target_col_idx
    else:
        cont_idx = target_col_idx + num_col_idx
        cat_all_idx = cat_col_idx

    cont_names = [col_names[i] for i in cont_idx]
    cat_names = [col_names[i] for i in cat_all_idx]
    ordered_names = cont_names + cat_names

    train_df_ord = train_df_orig[ordered_names].copy()
    test_df_ord = test_df_orig[ordered_names].copy()

    train_df_ord.columns = list(range(train_df_ord.shape[1]))
    test_df_ord.columns = list(range(test_df_ord.shape[1]))

    return {
        "data_dir": data_dir,
        "info": info,
        "train_df_orig": train_df_orig,
        "test_df_orig": test_df_orig,
        "train_df_ord": train_df_ord,
        "test_df_ord": test_df_ord,
        "ordered_names": ordered_names,
    }

def build_numeric_inverse_stats(train_df_ord, cont_cols):
    min_map = {}
    max_map = {}

    for k in cont_cols:
        x = pd.to_numeric(train_df_ord[k], errors="coerce").to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            min_map[int(k)] = 0.0
            max_map[int(k)] = 1.0
        else:
            min_map[int(k)] = float(x.min())
            max_map[int(k)] = float(x.max())

    return min_map, max_map


def inverse_numeric_transform(x_norm, col_idx, min_map, max_map):
    mn = min_map[int(col_idx)]
    mx = max_map[int(col_idx)]
    return x_norm * (mx - mn + 1.0) + (mn - 1.0)


def build_block_colnames_from_encoder_columns(encoded_columns, orig_pos):
    orig_pos = int(orig_pos)
    prefix_plain = f"{orig_pos}_"
    prefix_hash = f"{orig_pos}#"
    prefix_eq = f"{orig_pos}="
    prefix_dash = f"{orig_pos}-"

    block_cols = []
    for c in encoded_columns:
        if isinstance(c, str):
            if (
                c.startswith(prefix_plain)
                or c.startswith(prefix_hash)
                or c.startswith(prefix_eq)
                or c.startswith(prefix_dash)
            ):
                block_cols.append(c)
        elif c == orig_pos:
            block_cols.append(c)

    return block_cols


def decode_categorical_block(pred_block, orig_pos, encoded_columns, encoder, test_enc_base):
    block_cols = build_block_colnames_from_encoder_columns(encoded_columns, orig_pos)

    if len(block_cols) != pred_block.shape[1]:
        raise ValueError(
            f"Размер one-hot блока не совпадает для колонки {orig_pos}: "
            f"в encoder={len(block_cols)}, в prediction={pred_block.shape[1]}. "
            f"block_cols={block_cols}"
        )

    cls = np.argmax(pred_block, axis=1)
    onehot = np.zeros_like(pred_block, dtype=int)
    onehot[np.arange(len(cls)), cls] = 1

    # Берем реальный transformed test как базу, чтобы остальные one-hot блоки
    # оставались валидными для inverse_transform
    tmp = test_enc_base.copy()
    tmp[block_cols] = onehot

    inv = encoder.inverse_transform(tmp)

    if orig_pos not in inv.columns:
        raise ValueError(
            f"После inverse_transform не найдена колонка {orig_pos}. "
            f"Доступные колонки: {list(inv.columns)}"
        )

    return inv[orig_pos].to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--tabsyn_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nsample", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--limit_cols", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.outfile is None:
        args.outfile = f"prediction_{args.dataname}.csv"

    # 1) config
    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = 0.0

    # 2) построить/закешировать one-hot представление
    _tr, _va, _te = get_dataloader_tabsyn_onehot(
        tabsyn_root=args.tabsyn_root,
        dataname=args.dataname,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        missing_ratio=0.0,
        val_ratio=0.1,
    )

    cache_dir = f"./data_{args.dataname}_tabsyn_onehot"
    cache_file = os.path.join(
        cache_dir,
        f"{args.dataname}_tabsyn_onehot_miss0.0_seed{args.seed}.pk"
    )
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache file not found: {cache_file}")


    encoder_path = os.path.abspath(os.path.join(cache_dir, "encoder.pk"))
    print("encoder path =", encoder_path)
    print("cwd =", os.getcwd())
    with open(cache_file, "rb") as f:
        payload = pickle.load(f)
        


    test_vals = payload["test_vals"]
    test_obs_mask = payload["test_obs_mask"]
    ordered_names_from_payload = payload.get("ordered_names", None)
    cont_cols = payload["cont_cols"]
    saved_cat_dict = payload["saved_cat_dict"]

    with open(os.path.join(cache_dir, "encoder.pk"), "rb") as f:
        encoder = pickle.load(f)

    # 3) raw data / reordered raw tables
    raw = load_raw_data_and_order(args.tabsyn_root, args.dataname)
    test_df_orig = raw["test_df_orig"]
    train_df_ord = raw["train_df_ord"]
    test_df_ord = raw["test_df_ord"]
    ordered_names = raw["ordered_names"]

    if ordered_names_from_payload is not None and list(ordered_names_from_payload) != list(ordered_names):
        raise ValueError(
            "ordered_names из cache и из info.json не совпадают. "
            f"cache={ordered_names_from_payload}, rebuilt={ordered_names}"
        )

    min_map, max_map = build_numeric_inverse_stats(train_df_ord, cont_cols)

    # encoded train/test для согласованного inverse_transform
    train_enc = encoder.transform(train_df_ord.copy())
    test_enc = encoder.transform(test_df_ord.copy())
    encoded_columns = list(train_enc.columns)
    
    if not hasattr(encoder, "feature_names_out_"):
        encoder.feature_names_out_ = encoded_columns

    groups = build_groups(cont_cols, saved_cat_dict, ordered_names)

    if args.limit_cols is not None:
        groups = groups[:args.limit_cols]

    # prediction собираем в reordered raw-space, потом вернем в исходный порядок test.csv
    prediction_df_ord = test_df_ord.copy()

    # 4) model
    model = TabCSDI(config, args.device).to(args.device)
    sd = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(sd)
    model.eval()

    # 5) loop over columns
    for gi, g in enumerate(groups):
        col_name = g["name"]
        col_type = g["type"]

        drop_idx = g["drop_indices"]
        orig_pos = g["orig_pos"]

        print(
            f"[{gi+1}/{len(groups)}] Predict column: {col_name} "
            f"(type={col_type}, encoded_dims={len(drop_idx)})",
            flush=True,
        )

        ds = ColumnDropTestDataset(test_vals, test_obs_mask, drop_idx)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False)

        preds = []

        with torch.no_grad():
            for batch in loader:
                samples, observed_data, target_mask, observed_mask, _tp = model.evaluate(batch, args.nsample)

                samples = samples.permute(0, 1, 3, 2)   # (B, nsample, L, 1)
                med = samples.median(dim=1).values.squeeze(-1)  # (B, L)

                pred_batch = med.detach().cpu().numpy()  # (B, L)
                preds.append(pred_batch)

        pred_all = np.concatenate(preds, axis=0)  # (N, L)

        if col_type == "numeric":
            block = drop_idx[0]
            pred_norm = pred_all[:, block]
            pred_raw = inverse_numeric_transform(pred_norm, orig_pos, min_map, max_map)
            prediction_df_ord.iloc[:, orig_pos] = pred_raw

        else:
            block = drop_idx
            pred_block = pred_all[:, block]
            pred_labels = decode_categorical_block(
                pred_block=pred_block,
                orig_pos=orig_pos,
                encoded_columns=encoded_columns,
                encoder=encoder,
                test_enc_base=test_enc,
            )
            prediction_df_ord.iloc[:, orig_pos] = pred_labels

        print(f"[{args.dataname}] done col={col_name}", flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # вернуть в исходный порядок test.csv
    prediction_df_final = prediction_df_ord.copy()
    prediction_df_final.columns = ordered_names
    prediction_df_final = prediction_df_final[test_df_orig.columns]

    pred_path = os.path.join(args.outdir, args.outfile)
    prediction_df_final.to_csv(pred_path, index=False)

    meta = {
        "args": vars(args),
        "dataset": args.dataname,
        "data_dir": raw["data_dir"],
        "n_columns_predicted": len(groups),
        "prediction_path": pred_path,
        "ordered_names": ordered_names,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(args.outdir, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Saved:", pred_path)


if __name__ == "__main__":
    main()