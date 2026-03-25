import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler


MISSING_TOKENS = {"", " ", "nan", "NaN", "None", "NULL", "null", "?", "-", "—"}


def sanitize_num(arr: Iterable) -> np.ndarray:
    s = pd.Series(arr).replace(list(MISSING_TOKENS), np.nan)
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def sanitize_cat(arr: Iterable) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim > 1 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.ndim == 2 and a.shape[1] > 1:
        a = a.argmax(axis=1)
    return a.astype(str)


def pair_categorical(y_true: Iterable, y_pred: Iterable) -> Tuple[np.ndarray, np.ndarray]:
    yt = sanitize_cat(y_true)
    yp = sanitize_cat(y_pred)
    invalid = MISSING_TOKENS
    mask = (~np.isin(yt, list(invalid))) & (~np.isin(yp, list(invalid)))
    return yt[mask], yp[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))



def bootstrap_mean_std(metric_fn, y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 300, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        val = metric_fn(y_true[idx], y_pred[idx])
        if np.isfinite(val):
            vals.append(val)

    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1))


def format_mean_std(mean_value: float, std_value: float, digits: int = 6) -> str:
    if not np.isfinite(mean_value):
        return "nan"
    if not np.isfinite(std_value):
        return f"{mean_value:.{digits}f}±nan"
    return f"{mean_value:.{digits}f}±{std_value:.{digits}f}"


class DatasetSpec:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.info_path = dataset_dir / "info.json"
        self.train_path = dataset_dir / "train.csv"
        self.test_path = dataset_dir / "test.csv"

        self.info = json.loads(self.info_path.read_text(encoding="utf-8"))
        self.name = self.info.get("name", dataset_dir.name)
        self.train_df = pd.read_csv(self.train_path)
        self.test_df = pd.read_csv(self.test_path)
        self.columns = list(self.test_df.columns)

        self.num_idx = set(self.info.get("num_col_idx", []))
        self.cat_idx = set(self.info.get("cat_col_idx", []))
        self.target_idx = set(self.info.get("target_col_idx", []))

        metadata_cols = self.info.get("metadata", {}).get("columns", {})
        if not self.num_idx and not self.cat_idx:
            for key, value in metadata_cols.items():
                idx = int(key)
                if value.get("sdtype") == "numerical":
                    self.num_idx.add(idx)
                elif value.get("sdtype") == "categorical":
                    self.cat_idx.add(idx)

        # target --  добавим его в нужный тип, если он был пропущен
        for idx in self.target_idx:
            if idx in self.num_idx or idx in self.cat_idx:
                continue
            sdtype = metadata_cols.get(str(idx), {}).get("sdtype")
            if sdtype == "numerical":
                self.num_idx.add(idx)
            else:
                self.cat_idx.add(idx)

        self.num_cols = [self.columns[i] for i in range(len(self.columns)) if i in self.num_idx]
        self.cat_cols = [self.columns[i] for i in range(len(self.columns)) if i in self.cat_idx]
        self.scalers = self._fit_numeric_scalers()

    def _fit_numeric_scalers(self) -> Dict[str, Optional[StandardScaler]]:
        scalers: Dict[str, Optional[StandardScaler]] = {}
        for col in self.num_cols:
            x = sanitize_num(self.train_df[col])
            x = x[np.isfinite(x)]
            scaler = StandardScaler()
            scaler.fit(x.reshape(-1, 1))
            scalers[col] = scaler
        return scalers




def validate_prediction_structure(true_df: pd.DataFrame, pred_df: pd.DataFrame, dataset_name: str) -> None:
    if list(true_df.columns) != list(pred_df.columns):
        raise ValueError(
            f"[{dataset_name}] Колонки датасетов не совпадают.\n"
            f"Expected: {list(true_df.columns)}\nGot: {list(pred_df.columns)}"
        )
    if len(true_df) != len(pred_df):
        raise ValueError(f"{dataset_name} Количество строк не совпадает: в test.csv {len(true_df)}, в prediction: {len(pred_df)}")


def evaluate_dataset(spec: DatasetSpec, prediction_path: Path, output_dir: Path, n_bootstrap: int = 300, seed: int = 42) -> Dict:
    pred_df = pd.read_csv(prediction_path)

    # на всякий случай проверим, что все ок с тестом и предсказаниями (один размер и колонки)
    validate_prediction_structure(spec.test_df, pred_df, spec.name)

    rows = []
    for col_idx, col_name in enumerate(spec.columns):
        y_true = spec.test_df[col_name].to_numpy()
        y_pred = pred_df[col_name].to_numpy()

        if col_idx in spec.num_idx:
            yt = sanitize_num(y_true)
            yp = sanitize_num(y_pred)
            valid = np.isfinite(yt) & np.isfinite(yp)

            if valid.sum() == 0:
                pass
            else:
                scaler = spec.scalers.get(col_name)
                yt_valid = yt[valid]
                yp_valid = yp[valid]

                yt_eval = scaler.transform(yt_valid.reshape(-1, 1)).reshape(-1)
                yp_eval = scaler.transform(yp_valid.reshape(-1, 1)).reshape(-1)

                rmse_value = rmse(yt_eval, yp_eval)
                rmse_mu, rmse_sigma = bootstrap_mean_std(rmse, yt_eval, yp_eval, n_bootstrap=n_bootstrap, seed=seed)
                
                mae_value = mae(yt_eval, yp_eval)
                mae_mu, mae_sigma = bootstrap_mean_std(mae, yt_eval, yp_eval, n_bootstrap=n_bootstrap, seed=seed+1)
                row = {
                    "dataset": spec.name,
                    "column": col_name,
                    "type": "num",
                    "rmse": rmse_value,
                    "rmse_mean_std": format_mean_std(rmse_mu, rmse_sigma),
                    "mae": mae_value,
                    "mae_mean_std": format_mean_std(mae_mu, mae_sigma),
                }
        else:
            yt_pair, yp_pair = pair_categorical(y_true, y_pred)
            if yt_pair.size == 0:
                pass
            else:
                acc = float(accuracy_score(yt_pair, yp_pair))
                acc_mu, acc_sigma = bootstrap_mean_std(
                    lambda a, b: accuracy_score(*pair_categorical(a, b)), yt_pair, yp_pair, n_bootstrap=n_bootstrap, seed=seed
                )
                f1 = float(f1_score(yt_pair, yp_pair, average="macro", zero_division=0))
                f1_mu, f1_sigma = bootstrap_mean_std(
                    lambda a, b: f1_score(*pair_categorical(a, b), average="macro", zero_division=0),
                    yt_pair,
                    yp_pair,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 1,
                )
                precision = float(precision_score(yt_pair, yp_pair, average="macro", zero_division=0))
                precision_mu, precision_sigma = bootstrap_mean_std(
                    lambda a, b: precision_score(*pair_categorical(a, b), average="macro", zero_division=0),
                    yt_pair,
                    yp_pair,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 2,
                )
                recall = float(recall_score(yt_pair, yp_pair, average="macro", zero_division=0))
                recall_mu, recall_sigma = bootstrap_mean_std(
                    lambda a, b: recall_score(*pair_categorical(a, b), average="macro", zero_division=0),
                    yt_pair,
                    yp_pair,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 3,
                )
                row = {
                    "dataset": spec.name,
                    "column": col_name,
                    "type": "cat",
                    "accuracy": acc,
                    "accuracy_mean_std": format_mean_std(acc_mu, acc_sigma),
                    "f1": f1,
                    "f1_mean_std": format_mean_std(f1_mu, f1_sigma),
                    "precision": precision,
                    "precision_mean_std": format_mean_std(precision_mu, precision_sigma),
                    "recall": recall,
                    "recall_mean_std": format_mean_std(recall_mu, recall_sigma),
                }
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics_per_column.csv", index=False)

    summary = {
        "dataset": spec.name,
        "prediction_file": str(prediction_path),
        "numeric_columns": spec.num_cols,
        "categorical_columns": spec.cat_cols,
    }

    num_df = metrics_df[metrics_df["type"] == "num"]
    if not num_df.empty:
        summary["numeric_mean"] = {
            "rmse": float(num_df["rmse"].mean()),
            "mae": float(num_df["mae"].mean()),
        }

    cat_df = metrics_df[metrics_df["type"] == "cat"]
    if not cat_df.empty:
        summary["categorical_mean"] = {
            "accuracy": float(cat_df["accuracy"].mean()),
            "f1": float(cat_df["f1"].mean()),
            "precision": float(cat_df["precision"].mean()),
            "recall": float(cat_df["recall"].mean()),
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardized test")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser("evaluate", help="Evaluate one prediction file for one dataset")
    
    p_eval.add_argument("--dataset", required=True)
    p_eval.add_argument("--prediction", required=True, help="Path to model prediction csv")
    p_eval.add_argument("--output-dir", required=True)

    p_eval.add_argument("--data-root", type=str, default='./data')
    p_eval.add_argument("--bootstrap", type=int, default=300)
    p_eval.add_argument("--seed", type=int, default=42)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "evaluate":
        spec = DatasetSpec(Path(args.data_root) / args.dataset)
        summary = evaluate_dataset(
            spec=spec,
            prediction_path=Path(args.prediction),
            output_dir=Path(args.output_dir),
            n_bootstrap=args.bootstrap,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
