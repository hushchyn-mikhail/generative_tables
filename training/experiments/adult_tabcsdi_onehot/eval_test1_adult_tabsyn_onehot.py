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


def bootstrap_ci(metric_fn, y_true, y_pred, mask_rows, B=1000, seed=0):
    """
    Bootstrap by rows. mask_rows: boolean array length N telling which rows contribute.
    metric computed on y_true[mask_rows], y_pred[mask_rows] for each bootstrap sample.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        mr = mask_rows[idx]
        if mr.sum() == 0:
            vals.append(np.nan)
            continue
        vals.append(metric_fn(y_true[idx][mr], y_pred[idx][mr]))
    vals = np.array([v for v in vals if not np.isnan(v)])
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--missingratio", type=float, default=0.2)
    parser.add_argument("--tabsyn_root", type=str, required=True)
    parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--nsample", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--outdir", type=str, default="./results_test1")
    args = parser.parse_args()
    print('начало')
    os.makedirs(args.outdir, exist_ok=True)

    with open(os.path.join("config", args.config), "r") as f:
        config = yaml.safe_load(f)
    config["model"]["test_missing_ratio"] = args.missingratio

    _, _, test_loader = get_dataloader_adult_tabsyn_onehot(
        tabsyn_root=args.tabsyn_root,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        missing_ratio=args.missingratio,
        val_ratio=0.1,
    )


    with open("./data_adult_tabsyn_onehot/transformed_columns.pk", "rb") as f:
        cont_cols, saved_cat_dict = pickle.load(f)

    # model
    model = TabCSDI(config, args.device).to(args.device)
    sd = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(sd)
    model.eval()

    print("Загрузили модель")

    all_pred = []
    all_true = []
    all_eval = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if (i + 1) % 1 == 0:
                 print("начали batch", i, "of", len(test_loader))
            if i == 150:
                break
            samples, c_target, eval_points, observed_points, observed_time = model.evaluate(batch, args.nsample)
            # samples: (B, nsample, K=1, L)
            # c_target: (B, K=1, L)
            # eval_points: (B, K=1, L)  -> 1 where artificially masked and should be evaluated
            samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
            c_target = c_target.permute(0, 2, 1)   # (B, L, K)
            eval_points = eval_points.permute(0, 2, 1)  # (B, L, K)

            med = samples.median(dim=1).values  # (B, L, K)
            pred = med.squeeze(-1).cpu().numpy()
            true = c_target.squeeze(-1).cpu().numpy()
            ev = eval_points.squeeze(-1).cpu().numpy().astype(bool)

            all_pred.append(pred)
            all_true.append(true)
            all_eval.append(ev)
            if i == 0:
                print("закончили batch", i, "of", len(test_loader))


    print('подсчитали предсказания')

    pred = np.concatenate(all_pred, axis=0)  # (N, L)
    true = np.concatenate(all_true, axis=0)  # (N, L)
    evm = np.concatenate(all_eval, axis=0)   # (N, L)

    N, L = true.shape
    results_rows = []

    # numeric metrics
    for k in cont_cols:
        y_true = true[:, k]
        y_pred = pred[:, k]
        mask_rows = evm[:, k]

        mean_rmse, lo_rmse, hi_rmse = bootstrap_ci(rmse, y_true, y_pred, mask_rows, B=args.bootstrap, seed=args.seed + 100 + k)
        mean_mae, lo_mae, hi_mae = bootstrap_ci(mae, y_true, y_pred, mask_rows, B=args.bootstrap, seed=args.seed + 200 + k)

        results_rows.append({
            "group": "numeric",
            "column": f"cont_{k}",
            "metric": "RMSE",
            "mean": mean_rmse,
            "ci_low": lo_rmse,
            "ci_high": hi_rmse,
        })
        results_rows.append({
            "group": "numeric",
            "column": f"cont_{k}",
            "metric": "MAE",
            "mean": mean_mae,
            "ci_low": lo_mae,
            "ci_high": hi_mae,
        })

    # categorical metrics
    for key, cate_cols in saved_cat_dict.items():
        cate_cols = list(cate_cols)
        
        true_lab = np.argmax(true[:, cate_cols], axis=1)
        pred_lab = np.argmax(pred[:, cate_cols], axis=1)

        mask_rows = evm[:, cate_cols[0]]

        def acc_fn(a, b): 
            return float(accuracy_score(a, b))
        def f1_fn(a, b): 
            return float(f1_score(a, b, average="macro", zero_division=0))
        def rec_fn(a, b): 
            return float(recall_score(a, b, average="macro", zero_division=0))

        mean_acc, lo_acc, hi_acc = bootstrap_ci(acc_fn, true_lab, pred_lab, mask_rows, B=args.bootstrap, seed=args.seed + 300 + int(key))
        mean_f1, lo_f1, hi_f1 = bootstrap_ci(f1_fn, true_lab, pred_lab, mask_rows, B=args.bootstrap, seed=args.seed + 400 + int(key))
        mean_rec, lo_rec, hi_rec = bootstrap_ci(rec_fn, true_lab, pred_lab, mask_rows, B=args.bootstrap, seed=args.seed + 500 + int(key))

        results_rows.append({
            "group": "categorical",
            "column": f"cat_{key}",
            "metric": "Accuracy",
            "mean": mean_acc,
            "ci_low": lo_acc,
            "ci_high": hi_acc,
        })
        results_rows.append({
            "group": "categorical",
            "column": f"cat_{key}",
            "metric": "F1_macro",
            "mean": mean_f1,
            "ci_low": lo_f1,
            "ci_high": hi_f1,
        })
        results_rows.append({
            "group": "categorical",
            "column": f"cat_{key}",
            "metric": "Recall_macro",
            "mean": mean_rec,
            "ci_low": lo_rec,
            "ci_high": hi_rec,
        })

    # avg over columns

    # numeric
    num_rmse = [r["mean"] for r in results_rows if r["group"] == "numeric" and r["metric"] == "RMSE"]
    num_mae = [r["mean"] for r in results_rows if r["group"] == "numeric" and r["metric"] == "MAE"]
    # categorical
    cat_acc = [r["mean"] for r in results_rows if r["group"] == "categorical" and r["metric"] == "Accuracy"]
    cat_f1 = [r["mean"] for r in results_rows if r["group"] == "categorical" and r["metric"] == "F1_macro"]
    cat_rec = [r["mean"] for r in results_rows if r["group"] == "categorical" and r["metric"] == "Recall_macro"]

    summary = {
        "numeric_RMSE_mean_over_columns": float(np.mean(num_rmse)) if len(num_rmse) else float("nan"),
        "numeric_MAE_mean_over_columns": float(np.mean(num_mae)) if len(num_mae) else float("nan"),
        "categorical_Accuracy_mean_over_columns": float(np.mean(cat_acc)) if len(cat_acc) else float("nan"),
        "categorical_F1_macro_mean_over_columns": float(np.mean(cat_f1)) if len(cat_f1) else float("nan"),
        "categorical_Recall_macro_mean_over_columns": float(np.mean(cat_rec)) if len(cat_rec) else float("nan"),
    }

    import pandas as pd

    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(args.outdir, "test1_per_column_metrics_ci.csv")
    df.to_csv(out_csv, index=False)

    out_json = os.path.join(args.outdir, "test1_summary.json")
    with open(out_json, "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)

    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("Summary:", summary)


if __name__ == "__main__":
    main()
