import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

from tabsyn_imputer import tabsyn_impute, _make_random_missing_by_column


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def bootstrap_mean_sigma(metric_fn, y_true, y_pred, row_mask, B=300, seed=0, ddof=1):
    """
    Bootstrap mean and sigma (std) of metric over rows.
    row_mask: boolean mask (True where we evaluate, e.g. artificially missing entries)
    Returns: (mean, sigma)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = row_mask[idx]
        if m.sum() == 0:
            continue
        vals.append(metric_fn(y_true[idx][m], y_pred[idx][m]))

    if not vals:
        return float("nan"), float("nan")

    vals = np.asarray(vals, dtype=float)
    mu = float(vals.mean())
    
    sigma = float(vals.std(ddof=ddof)) if vals.size > 1 else float("nan")
    return mu, sigma



def bootstrap_ci(metric_fn, y_true, y_pred, row_mask, B=300, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        m = row_mask[idx]
        if m.sum() == 0:
            continue
        vals.append(metric_fn(y_true[idx][m], y_pred[idx][m]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    vals = np.array(vals, dtype=float)
    return float(vals.mean()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def sanitize_labels(arr):
    """Make 1D string labels for categorical metrics."""
    a = np.asarray(arr)

    # squeeze (n,1) -> (n,)
    if a.ndim > 1 and a.shape[1] == 1:
        a = a.reshape(-1)

    # if accidentally got (n,k) (logits/one-hot), turn into class id string
    if a.ndim == 2 and a.shape[1] > 1:
        a = a.argmax(axis=1)

    # everything -> string
    return a.astype(str)

def _pair_str(a, b):
    aa = sanitize_labels(a)
    bb = sanitize_labels(b)
    mv = (aa != "nan") & (bb != "nan")
    return aa[mv], bb[mv]

def mean_sigma(s: pd.Series, ddof=1):
    s = pd.to_numeric(s, errors="coerce").dropna()
    mu = float(s.mean()) if len(s) else float("nan")
    sig = float(s.std(ddof=ddof)) if len(s) > 1 else float("nan")
    return mu, sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabsyn_root", required=True)
    p.add_argument("--dataname", required=True, choices=["adult", "default", "shoppers", "magic", "beijing"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--missingratio", type=float, default=0.2)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--resample_steps", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    data_dir = os.path.join(args.tabsyn_root, "data", args.dataname)
    info = json.load(open(os.path.join(data_dir, "info.json"), "r"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
    
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    cols = list(test_df.columns)
    N, D = test_df.shape
    
    num_cols = [cols[i] for i in info["num_col_idx"]]
    min_map = {c: float(train_df[c].min()) for c in num_cols}
    max_map = {c: float(train_df[c].max()) for c in num_cols}

    def tabsyn_like_norm(x, c):
        mn = min_map[c]
        mx = max_map[c]
        # как  в TabCSDI:
        return (x - (mn - 1.0)) / (mx - mn + 1.0)

    # missing mask: True => missing
    missing_mask = _make_random_missing_by_column(N, D, args.missingratio, args.seed)

    # impute
    syn_df = tabsyn_impute(
        tabsyn_root=args.tabsyn_root,
        dataname=args.dataname,
        missing_cols_mask=missing_mask,
        gpu=args.gpu,
        num_steps=args.num_steps,
        resample_steps=args.resample_steps,
        seed=args.seed,
    )

    # results dir
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, f"{args.dataname}_test1_{ts}")
    os.makedirs(outdir, exist_ok=True)
    syn_df.to_csv(os.path.join(outdir, "imputed.csv"), index=False)

    num_idx = set(info["num_col_idx"])
    cat_idx = set(info["cat_col_idx"]) | set(info["target_col_idx"])

    rows = []

    # per-column metrics (на тех местах, где реально были “пропуски”)
    for j, name in enumerate(cols):
        miss_rows = missing_mask[:, j]
        if miss_rows.sum() == 0:
            continue

        y_true = test_df.iloc[:, j].to_numpy()
        y_pred = syn_df.iloc[:, j].to_numpy()

        if j in num_idx:
            # numeric

            yt = y_true[miss_rows].astype(float)
            yp = y_pred[miss_rows].astype(float)

            yt_n = tabsyn_like_norm(yt, name)
            yp_n = tabsyn_like_norm(yp, name)
            rmse_bs_mean, rmse_bs_sigma = bootstrap_mean_sigma(
                rmse,
                tabsyn_like_norm(y_true.astype(float), name),
                tabsyn_like_norm(y_pred.astype(float), name),
                miss_rows, B=args.bootstrap, seed=args.seed
            )

            mae_bs_mean, mae_bs_sigma = bootstrap_mean_sigma(
                mae,
                tabsyn_like_norm(y_true.astype(float), name),
                tabsyn_like_norm(y_pred.astype(float), name),
                miss_rows, B=args.bootstrap, seed=args.seed + 7
            )

            rows.append({
                "col": name, "type": "num",
                "rmse": rmse(yt_n, yp_n),
                "rmse_bs_mean": rmse_bs_mean,
                "rmse_bs_sigma": rmse_bs_sigma,
                "mae": mae(yt_n, yp_n),
                "mae_bs_mean": mae_bs_mean,
                "mae_bs_sigma": mae_bs_sigma,
            })


        else:
            # categorical
            y_true_col = sanitize_labels(y_true)
            y_pred_col = sanitize_labels(y_pred)

            yt = sanitize_labels(y_true[miss_rows])
            yp = sanitize_labels(y_pred[miss_rows])
            

            mvalid = (yt != "nan") & (yp != "nan")
            yt = yt[mvalid]
            yp = yp[mvalid]
            
            if yt.size == 0:
                acc = f1 = rec = 0.0
            else:
                acc = accuracy_score(yt, yp)
                f1  = f1_score(yt, yp, average="macro", zero_division=0)
                rec = recall_score(yt, yp, average="macro", zero_division=0)

            acc_bs_mean, acc_bs_sigma = bootstrap_mean_sigma(
                lambda a, b: accuracy_score(*_pair_str(a, b)),
                sanitize_labels(y_true), sanitize_labels(y_pred), miss_rows,
                B=args.bootstrap, seed=args.seed
            )

            f1_bs_mean, f1_bs_sigma = bootstrap_mean_sigma(
                lambda a, b: f1_score(*_pair_str(a, b), average="macro", zero_division=0),
                sanitize_labels(y_true), sanitize_labels(y_pred), miss_rows,
                B=args.bootstrap, seed=args.seed + 11
            )

            rec_bs_mean, rec_bs_sigma = bootstrap_mean_sigma(
                lambda a, b: recall_score(*_pair_str(a, b), average="macro", zero_division=0),
                sanitize_labels(y_true), sanitize_labels(y_pred), miss_rows,
                B=args.bootstrap, seed=args.seed + 19
            )

            rows.append({
                "col": name, "type": "cat",
                "acc": acc,
                "acc_bs_mean": acc_bs_mean,
                "acc_bs_sigma": acc_bs_sigma,
                "f1": f1,
                "f1_bs_mean": f1_bs_mean,
                "f1_bs_sigma": f1_bs_sigma,
                "recall": rec,
                "recall_bs_mean": rec_bs_mean,
                "recall_bs_sigma": rec_bs_sigma,
            })


    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(os.path.join(outdir, "metrics_per_column.csv"), index=False)

    # summary: average by type
    summary = {}
    if (df_metrics["type"] == "num").any():
        num_df = df_metrics[df_metrics["type"] == "num"]

        summary["num_rmse_mean"] , _  = mean_sigma(num_df["rmse"])
        summary["num_mae_mean"]  , _   = mean_sigma(num_df["mae"])

        # summary["num_rmse_bs_mean_mean"], summary["num_rmse_bs_mean_sigma"] = mean_sigma(num_df["rmse_bs_mean"])
        # summary["num_mae_bs_mean_mean"],  summary["num_mae_bs_mean_sigma"]  = mean_sigma(num_df["mae_bs_mean"])

    if (df_metrics["type"] == "cat").any():
        cat_df = df_metrics[df_metrics["type"] == "cat"]

        summary["cat_acc_mean"]    , _     = mean_sigma(cat_df["acc"])
        summary["cat_f1_mean"]     , _      = mean_sigma(cat_df["f1"])
        summary["cat_recall_mean"] , _  = mean_sigma(cat_df["recall"])

    # if (df_metrics["type"] == "num").any():
    #     num_df = df_metrics[df_metrics["type"] == "num"]
    #     summary["num_rmse_mean"] = float(num_df["rmse"].mean())
    #     summary["num_mae_mean"] = float(num_df["mae"].mean())
    # if (df_metrics["type"] == "cat").any():
    #     cat_df = df_metrics[df_metrics["type"] == "cat"]
    #     summary["cat_acc_mean"] = float(cat_df["acc"].mean())
    #     summary["cat_f1_mean"] = float(cat_df["f1"].mean())

    #     summary["cat_recall_mean"] = float(cat_df["recall"].mean())

    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)

    print("Saved:", outdir)
    print(summary)


if __name__ == "__main__":
    main()
