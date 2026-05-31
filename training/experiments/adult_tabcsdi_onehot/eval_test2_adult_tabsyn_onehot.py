import argparse
import os
import json
import pickle
import numpy as np
import torch
import yaml

import os, sys

# project root = .../imputation-project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TABCSDI_DIR = os.path.join(PROJECT_ROOT, "submodules", "TabCSDI")

# чтобы работали импорты `from src...`
sys.path.insert(0, TABCSDI_DIR)

# чтобы относительные пути вроде "config/..." работали всегда
os.chdir(TABCSDI_DIR)


from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.main_model_table import TabCSDI
from dataset_adult_tabsyn_onehot import get_dataloader_adult_tabsyn_onehot




def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def bootstrap_ci(metric_fn, y_true, y_pred, mask_rows, B=500, seed=0):
    """
    Bootstrap by rows. metric computed on y_true[mask_rows], y_pred[mask_rows] for each bootstrap sample.
    mask_rows: boolean array (N,)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        mr = mask_rows[idx]
        if mr.sum() == 0:
            continue
        vals.append(metric_fn(y_true[idx][mr], y_pred[idx][mr]))
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    vals = np.asarray(vals)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))



class ColumnDropTestDataset(Dataset):
    def __init__(self, values, base_observed_mask, drop_indices):
        """
        values: (N, L) full encoded data
        base_observed_mask: (N, L) -> data exists
        drop_indices: list[int] indices to hide
        """
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

def save_progress(results, outdir, filename="test2_per_column_metrics_ci.csv"):
    import pandas as pd
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(results)
    out_csv = os.path.join(outdir, filename)
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tabsyn_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--outdir", type=str, default="./results_test2")
    parser.add_argument("--limit_cols", type=int, default=None, help="debug: evaluate only first K columns")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = 0.0


    _train_loader, _val_loader, _test_loader = get_dataloader_adult_tabsyn_onehot(
        tabsyn_root=args.tabsyn_root,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        missing_ratio=0.0,
        val_ratio=0.1,
    )

    cache_file = f"./data_adult_tabsyn_onehot/adult_tabsyn_onehot_miss0.0_seed{args.seed}.pk"

    with open(cache_file, "rb") as f:
        payload = pickle.load(f)

    test_vals = payload["test_vals"]
    test_obs_mask = payload["test_obs_mask"]
    ordered_names = payload.get("ordered_names", None)

    # cols -> onehot blocks
    with open("./data_adult_tabsyn_onehot/transformed_columns.pk", "rb") as f:
        cont_cols, saved_cat_dict = pickle.load(f)

    groups = []

    # numerical
    for idx in cont_cols:
        name = ordered_names[idx] if ordered_names is not None else f"cont_{idx}"
        groups.append({
            "name": str(name),
            "type": "numeric",
            "drop_indices": [int(idx)],
            "orig_pos": int(idx),
        })

    # сategorical
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

    if args.limit_cols is not None:
        groups = groups[: args.limit_cols]

    model = TabCSDI(config, args.device).to(args.device)
    sd = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(sd)
    model.eval()

    results = []

    for gi, g in enumerate(groups):
        col_name = g["name"]
        col_type = g["type"]
        drop_idx = g["drop_indices"]

        print(f"[{gi+1}/{len(groups)}] Drop column: {col_name} ({col_type}), dims={len(drop_idx)}")

        ds = ColumnDropTestDataset(test_vals, test_obs_mask, drop_idx)
        loader = DataLoader(ds, batch_size=args.eval_batch_size, shuffle=False)

        
        preds = []
        trues = []
        masks = []

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i % 10 == 0:
                    print(f'batch {i} of {len(loader)}')
                
                
                batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

                samples, observed_data, target_mask, observed_mask, _tp = model.evaluate(batch, args.nsample)
                # (B, nsample, 1, L)
                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, 1)
                med = samples.median(dim=1).values.squeeze(-1)  # (B, L)

                pred_batch = med.detach().cpu().numpy()
                true_batch = observed_data.squeeze(1).detach().cpu().numpy()  # (B, L)

                row_mask = observed_mask.squeeze(1).detach().cpu().numpy().astype(bool)[:, drop_idx[0]]

                preds.append(pred_batch)
                trues.append(true_batch)
                masks.append(row_mask)

        pred_all = np.concatenate(preds, axis=0) # (N, L)
        true_all = np.concatenate(trues, axis=0) # (N, L)
        row_mask = np.concatenate(masks, axis=0) # (N,)

        # numeric
        if col_type == "numeric":
            k = drop_idx[0]
            y_true = true_all[:, k]
            y_pred = pred_all[:, k]

            m_rmse, lo_rmse, hi_rmse = bootstrap_ci(rmse, y_true, y_pred, row_mask, B=args.bootstrap, seed=args.seed + 1000 + gi)
            m_mae, lo_mae, hi_mae = bootstrap_ci(mae, y_true, y_pred, row_mask, B=args.bootstrap, seed=args.seed + 2000 + gi)

            results.append({"column": col_name, "type": col_type, "metric": "RMSE", "mean": m_rmse, "ci_low": lo_rmse, "ci_high": hi_rmse})
            results.append({"column": col_name, "type": col_type, "metric": "MAE", "mean": m_mae, "ci_low": lo_mae, "ci_high": hi_mae})

        else:
            # categorical
            block = drop_idx
            y_true = np.argmax(true_all[:, block], axis=1)
            y_pred = np.argmax(pred_all[:, block], axis=1)

            def acc_fn(a, b): 
                return float(accuracy_score(a, b))
            def f1_fn(a, b): 
                return float(f1_score(a, b, average="macro", zero_division=0))
            def rec_fn(a, b): 
                return float(recall_score(a, b, average="macro", zero_division=0))

            m_acc, lo_acc, hi_acc = bootstrap_ci(acc_fn, y_true, y_pred, row_mask, B=args.bootstrap, seed=args.seed + 3000 + gi)
            m_f1, lo_f1, hi_f1 = bootstrap_ci(f1_fn, y_true, y_pred, row_mask, B=args.bootstrap, seed=args.seed + 4000 + gi)
            m_rec, lo_rec, hi_rec = bootstrap_ci(rec_fn, y_true, y_pred, row_mask, B=args.bootstrap, seed=args.seed + 5000 + gi)

            results.append({"column": col_name, "type": col_type, "metric": "Accuracy", "mean": m_acc, "ci_low": lo_acc, "ci_high": hi_acc})
            results.append({"column": col_name, "type": col_type, "metric": "F1_macro", "mean": m_f1, "ci_low": lo_f1, "ci_high": hi_f1})
            results.append({"column": col_name, "type": col_type, "metric": "Recall_macro", "mean": m_rec, "ci_low": lo_rec, "ci_high": hi_rec})

        
        out_csv_progress = save_progress(results, args.outdir)
        print(f"Progress saved: {out_csv_progress}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    
    import pandas as pd
    df = pd.DataFrame(results)
    out_csv = os.path.join(args.outdir, "test2_per_column_metrics_ci.csv")
    df.to_csv(out_csv, index=False)

    # avg over columns
    avg_rows = []
    for t in ["numeric", "categorical"]:
        df_t = df[df["type"] == t]
        for metric in sorted(df_t["metric"].unique()):
            avg_rows.append({
                "type": t,
                "metric": metric,
                "mean_over_columns": float(df_t[df_t["metric"] == metric]["mean"].mean())
            })

    df_avg = pd.DataFrame(avg_rows)
    out_avg = os.path.join(args.outdir, "test2_avg_by_type.csv")
    df_avg.to_csv(out_avg, index=False)

    
    summary = {
        "args": vars(args),
        "n_columns_evaluated": len(groups),
        "outputs": {"per_column_csv": out_csv, "avg_by_type_csv": out_avg},
    }
    out_json = os.path.join(args.outdir, "test2_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_avg)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
