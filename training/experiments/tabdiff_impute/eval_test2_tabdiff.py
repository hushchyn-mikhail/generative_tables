# import argparse
# import os
# import json
# import datetime
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score, recall_score
# from sklearn.preprocessing import StandardScaler

# from tabdiff_imputer import tabdiff_impute


# def sanitize_num(arr):
#     s = pd.Series(arr).replace(["", " ", "nan", "NaN", "None", "NULL", "null", "?", "-", "—"], np.nan)
#     return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


# def sanitize_cat(arr):
#     a = np.asarray(arr)
#     if a.ndim > 1 and a.shape[1] == 1:
#         a = a.reshape(-1)
#     if a.ndim == 2 and a.shape[1] > 1:
#         a = a.argmax(axis=1)
#     return a.astype(str)


# def _pair_str(a, b):
#     yt = sanitize_cat(a)
#     yp = sanitize_cat(b)
#     m = (yt != "nan") & (yp != "nan") & (yt != "?") & (yp != "?")
#     yt, yp = yt[m], yp[m]
#     return yt, yp


# def rmse(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float("nan")


# def mae(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


# def bootstrap_mean_sigma(metric_fn, y_true, y_pred, row_mask, B=300, seed=0, ddof=1):
#     rng = np.random.default_rng(seed)
#     n = len(y_true)
#     vals = []
#     for _ in range(B):
#         idx = rng.integers(0, n, size=n)
#         m = row_mask[idx]
#         if m.sum() == 0:
#             continue
#         vals.append(metric_fn(y_true[idx][m], y_pred[idx][m]))

#     if not vals:
#         return float("nan"), float("nan")

#     vals = np.asarray(vals, dtype=float)
#     mu = float(vals.mean())
#     sigma = float(vals.std(ddof=ddof)) if vals.size > 1 else float("nan")
#     return mu, sigma


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


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--tabsyn_root", required=True)
#     p.add_argument("--tabdiff_root", required=True)

#     p.add_argument("--dataname", required=True, choices=["adult", "default", "shoppers", "magic", "beijing"])
#     p.add_argument("--gpu", type=int, default=0)

#     p.add_argument("--resample_rounds", type=int, default=1)
#     p.add_argument("--impute_condition", type=str, default="x_t")

#     p.add_argument("--bootstrap", type=int, default=300)
#     p.add_argument("--seed", type=int, default=1)

#     p.add_argument("--ckpt_path", type=str, default=None)
#     p.add_argument("--exp_name", type=str, default=None)
#     p.add_argument("--non_learnable_schedule", action="store_true")

#     p.add_argument("--outdir", required=True)
#     args = p.parse_args()

#     data_dir = os.path.join(args.tabsyn_root, "data", args.dataname)
#     info = json.load(open(os.path.join(data_dir, "info.json"), "r"))
#     test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
#     train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))

#     cols = list(test_df.columns)
#     N, D = test_df.shape

#     num_cols = [cols[i] for i in info["num_col_idx"]]


#     min_map = {c: float(train_df[c].min()) for c in num_cols}
#     max_map = {c: float(train_df[c].max()) for c in num_cols}
#     num_idx = set(info["num_col_idx"])

#     def tabsyn_like_norm(x, c):
#         mn = min_map[c]
#         mx = max_map[c]
#         return (x - (mn - 1.0)) / (mx - mn + 1.0)

#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     outdir = os.path.join(args.outdir, f"{args.dataname}_tabdiff_test2_{ts}")
#     os.makedirs(outdir, exist_ok=True)

#     all_rows = []

#     for j, name in enumerate(cols):
#         missing_mask = np.zeros((N, D), dtype=bool)
#         missing_mask[:, j] = True  # убрали целиком колонку j
#         print(f"обработка {name}")

#         syn_df = tabdiff_impute(
#             tabsyn_root=args.tabsyn_root,
#             tabdiff_root=args.tabdiff_root,
#             dataname=args.dataname,
#             missing_cols_mask=missing_mask,
#             gpu=args.gpu,
#             resample_rounds=args.resample_rounds,
#             impute_condition=args.impute_condition,
#             seed=args.seed,
#             ckpt_path=args.ckpt_path,
#             exp_name=args.exp_name,
#             non_learnable_schedule=args.non_learnable_schedule,
#         )

#         y_true = test_df.iloc[:, j].to_numpy()
#         y_pred = syn_df.iloc[:, j].to_numpy()
#         miss_rows = missing_mask[:, j]

#         if j in num_idx:
#             yt = sanitize_num(y_true)
#             yp = sanitize_num(y_pred)
            

#             valid = miss_rows & np.isfinite(yt) & np.isfinite(yp)
#             if valid.sum() == 0:
#                 r = a = float("nan")
#                 rmse_bs_mean = rmse_bs_sigma = float("nan")
#                 mae_bs_mean = mae_bs_sigma = float("nan")
#             else:
#                 yt_n = tabsyn_like_norm(yt[valid], name)
#                 yp_n = tabsyn_like_norm(yp[valid], name)

#                 r = rmse(yt_n, yp_n)
#                 a = mae(yt_n, yp_n)

#                 row_all = np.ones(len(yt_n), dtype=bool)
#                 rmse_bs_mean, rmse_bs_sigma = bootstrap_mean_sigma(
#                     rmse, yt_n, yp_n, row_all, B=args.bootstrap, seed=args.seed
#                 )
#                 mae_bs_mean, mae_bs_sigma = bootstrap_mean_sigma(
#                     mae, yt_n, yp_n, row_all, B=args.bootstrap, seed=args.seed + 7
#                 )

#             all_rows.append({
#                 "col": name, "type": "num",
#                 "rmse": r,
#                 "rmse_bs_mean": rmse_bs_mean, "rmse_bs_sigma": rmse_bs_sigma,
#                 "mae": a,
#                 "mae_bs_mean": mae_bs_mean, "mae_bs_sigma": mae_bs_sigma,
#             })
#         else:
#             acc = cat_acc(y_true[miss_rows], y_pred[miss_rows])
#             f1 = cat_f1(y_true[miss_rows], y_pred[miss_rows])
#             rec = cat_recall(y_true[miss_rows], y_pred[miss_rows])

#             acc_bs_mean, acc_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: accuracy_score(*_pair_str(a, b)),
#                 y_true, y_pred, miss_rows, B=args.bootstrap, seed=args.seed
#             )
#             f1_bs_mean, f1_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: f1_score(*_pair_str(a, b), average="macro", zero_division=0),
#                 y_true, y_pred, miss_rows, B=args.bootstrap, seed=args.seed + 11
#             )
#             rec_bs_mean, rec_bs_sigma = bootstrap_mean_sigma(
#                 lambda a, b: recall_score(*_pair_str(a, b), average="macro", zero_division=0),
#                 y_true, y_pred, miss_rows, B=args.bootstrap, seed=args.seed + 19
#             )

#             all_rows.append({
#                 "col": name, "type": "cat",
#                 "acc": acc, "acc_bs_mean": acc_bs_mean, "acc_bs_sigma": acc_bs_sigma,
#                 "f1": f1, "f1_bs_mean": f1_bs_mean, "f1_bs_sigma": f1_bs_sigma,
#                 "recall": rec, "recall_bs_mean": rec_bs_mean, "recall_bs_sigma": rec_bs_sigma,
#             })

#         print(f"[{args.dataname}] done col={name}")

#     df = pd.DataFrame(all_rows)
#     df.to_csv(os.path.join(outdir, "metrics_per_column.csv"), index=False)

#     summary = {}
#     if (df["type"] == "num").any():
#         num_df = df[df["type"] == "num"]
#         summary["num_rmse_mean"] = float(num_df["rmse"].mean())
#         summary["num_mae_mean"] = float(num_df["mae"].mean())
#     if (df["type"] == "cat").any():
#         cat_df = df[df["type"] == "cat"]
#         summary["cat_acc_mean"] = float(cat_df["acc"].mean())
#         summary["cat_f1_mean"] = float(cat_df["f1"].mean())
#         summary["cat_recall_mean"] = float(cat_df["recall"].mean())

#     with open(os.path.join(outdir, "summary.json"), "w") as f:
#         json.dump({"args": vars(args), "summary": summary}, f, indent=2)

#     print("Saved:", outdir)
#     print(summary)


# if __name__ == "__main__":
#     main()

    

import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd

from tabdiff_imputer import tabdiff_impute


def main():
    print(1)
    p = argparse.ArgumentParser()



    p.add_argument("--bootstrap", type=int, default=300)

    
    p.add_argument("--tabsyn_root", required=True)
    p.add_argument("--tabdiff_root", required=True)

    p.add_argument(
        "--dataname",
        required=True,
        choices=["adult", "default", "shoppers", "magic", "beijing"]
    )
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--resample_rounds", type=int, default=1)
    p.add_argument("--impute_condition", type=str, default="x_t")

    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--exp_name", type=str, default=None)
    p.add_argument("--non_learnable_schedule", action="store_true")

    p.add_argument("--outdir", required=True)
    p.add_argument("--outfile", type=str, default="prediction.csv")

    args = p.parse_args()
    args.outfile = f'prediction_{args.dataname}.csv'

    data_dir = os.path.join(args.tabsyn_root, "data", args.dataname)

    info_path = os.path.join(data_dir, "info.json")
    test_path = os.path.join(data_dir, "test.csv")
    train_path = os.path.join(data_dir, "train.csv")

    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Не найден файл: {info_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Не найден файл: {test_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Не найден файл: {train_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    cols = list(test_df.columns)
    N, D = test_df.shape

    print(f"Dataset: {args.dataname}")
    print(f"Test shape: {test_df.shape}")
    print(f"Columns: {D}")

    # Стартуем с копии test_df, потом каждую колонку заменяем предсказанной версией
    prediction_df = test_df.copy()


    # prediction_df = pd.DataFrame(index=test_df.index, columns=test_df.columns)

    for j, name in enumerate(cols):
        missing_mask = np.zeros((N, D), dtype=bool)
        missing_mask[:, j] = True  # маскируем только одну колонку во всем test

        print(f"Обработка колонки {j + 1}/{D}: {name}")

        syn_df = tabdiff_impute(
            tabsyn_root=args.tabsyn_root,
            tabdiff_root=args.tabdiff_root,
            dataname=args.dataname,
            missing_cols_mask=missing_mask,
            gpu=args.gpu,
            resample_rounds=args.resample_rounds,
            impute_condition=args.impute_condition,
            seed=args.seed,
            ckpt_path=args.ckpt_path,
            exp_name=args.exp_name,
            non_learnable_schedule=args.non_learnable_schedule,
        )

        if not isinstance(syn_df, pd.DataFrame):
            syn_df = pd.DataFrame(syn_df, columns=test_df.columns)

        if syn_df.shape != test_df.shape:
            raise ValueError(
                f"tabdiff_impute вернул DataFrame неправильной формы "
                f"для колонки {name}: got {syn_df.shape}, expected {test_df.shape}"
            )

        if list(syn_df.columns) != list(test_df.columns):
            syn_df.columns = test_df.columns

        prediction_df.iloc[:, j] = syn_df.iloc[:, j].values

        print(f"[{args.dataname}] done col={name}")

    os.makedirs(args.outdir, exist_ok=True)

    pred_path = os.path.join(args.outdir, args.outfile)
    prediction_df.to_csv(pred_path, index=False)

    meta = {
        "args": vars(args),
        "dataname": args.dataname,
        "data_dir": data_dir,
        "test_shape": list(test_df.shape),
        "columns": cols,
        "prediction_path": pred_path,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(args.outdir, "run_info.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved prediction: {pred_path}")


if __name__ == "__main__":
    main()