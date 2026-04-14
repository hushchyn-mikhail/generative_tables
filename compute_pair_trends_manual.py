"""
python compute_pair_trends_manual.py \
  --dataset-dir data/adult \
  --dataset-name adult \
  --pred-path outputs/prediction_adult.csv \
  --output-dir metrics \
  --model-name tabsyn
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Directory with train.csv and info.json",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pred-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory where metrics will be saved",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used in output path",
    )
    parser.add_argument(
        "--train-filename",
        type=str,
        default="train.csv",
    )
    parser.add_argument(
        "--info-filename",
        type=str,
        default="info.json",
    )
    parser.add_argument(
        "--drop-target",
        action="store_true",
        help="Drop target columns if target_col_idx exists in info.json",
    )
    parser.add_argument(
        "--correlation-coefficient",
        choices=["Pearson", "Spearman"],
        default="Pearson",
        help="Coeff for CorrelationSimilarity",
    )
    parser.add_argument(
        "--real-correlation-threshold",
        type=float,
        default=None,
        help="threshold for CorrelationSimilarity",
    )
    parser.add_argument(
        "--real-association-threshold",
        type=float,
        default=None,
        help="threshold for ContingencySimilarity",
    )
    parser.add_argument(
        "--num-discrete-bins",
        type=int,
        default=10,
        help="Bins for numerical columns when mixed pairs use ContingencySimilarity",
    )
    parser.add_argument(
        "--num-rows-subsample",
        type=int,
        default=None,
        help="for ContingencySimilarity speedup.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_flexible(path: Path, column_names: Optional[Iterable[str]]) -> pd.DataFrame:
    if column_names is None:
        return pd.read_csv(path)

    column_names = list(column_names)
    df = pd.read_csv(path, header=None)
    df.columns = column_names
    return df


def parse_metadata(info: dict) -> Tuple[list[str], set[str], set[str], list[str]]:
    column_names = info.get("column_names")
    task_type = info.get("task_type")

    num_idx = set(info.get("num_col_idx", []))
    cat_idx = set(info.get("cat_col_idx", []))
    target_idx = set(info.get("target_col_idx", []))

    numeric = {column_names[i] for i in num_idx}
    categorical = {column_names[i] for i in cat_idx}
    targets = [column_names[i] for i in target_idx]

    if task_type == "regression":
        numeric |= set(targets)
    else:
        categorical |= set(targets)

    #inferred = set(column_names) - numeric - categorical
    # categorical |= inferred

    return column_names, numeric, categorical, targets


def coerce_types(df: pd.DataFrame, numeric_cols: set[str]) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string").fillna("<NA>")
    return df


def build_matrix(long_df: pd.DataFrame, columns: list[str], value_col: str, diagonal: float) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
    for _, row in long_df.iterrows():
        c1, c2, value = row["column_1"], row["column_2"], row[value_col]
        matrix.loc[c1, c2] = value
        matrix.loc[c2, c1] = value
    for col in columns:
        matrix.loc[col, col] = diagonal
    return matrix


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()

    real_path = args.dataset_dir / args.train_filename
    meta_path = args.dataset_dir / args.info_filename

    info = load_json(meta_path)
    ordered_columns, numeric_cols, categorical_cols, target_cols = parse_metadata(info)

    real_df = read_csv_flexible(real_path, ordered_columns)
    syn_df = read_csv_flexible(args.pred_path, ordered_columns)

    if list(real_df.columns) != list(syn_df.columns):
        syn_df = syn_df.copy()
        syn_df.columns = real_df.columns

    # if args.drop_target and target_cols:
    #     keep = [c for c in ordered_columns if c not in set(target_cols)]
    #     ordered_columns = keep
    #     real_df = real_df[keep]
    #     syn_df = syn_df[keep]
    #     numeric_cols = numeric_cols & set(keep)
    #     categorical_cols = categorical_cols & set(keep)

    real_df = coerce_types(real_df, numeric_cols)
    syn_df = coerce_types(syn_df, numeric_cols)
    syn_df = syn_df[real_df.columns]

    from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity

    rows = []
    for c1, c2 in itertools.combinations(ordered_columns, 2):
        pair_real = real_df[[c1, c2]].copy()
        pair_syn = syn_df[[c1, c2]].copy()

        both_numeric = c1 in numeric_cols and c2 in numeric_cols

        metric_name = None
        score = np.nan
        error = None

        try:
            if both_numeric:
                metric_name = "CorrelationSimilarity"
                kwargs = {
                    "real_data": pair_real,
                    "synthetic_data": pair_syn,
                    "coefficient": args.correlation_coefficient,
                }
                if args.real_correlation_threshold is not None:
                    kwargs["real_correlation_threshold"] = args.real_correlation_threshold

                score = CorrelationSimilarity.compute(**kwargs)
            else:
                metric_name = "ContingencySimilarity"
                continuous_cols = [c for c in [c1, c2] if c in numeric_cols]
                kwargs = {
                    "real_data": pair_real,
                    "synthetic_data": pair_syn,
                    "num_discrete_bins": args.num_discrete_bins,
                }
                if continuous_cols:
                    kwargs["continuous_column_names"] = continuous_cols
                if args.real_association_threshold is not None:
                    kwargs["real_association_threshold"] = args.real_association_threshold
                if args.num_rows_subsample is not None:
                    kwargs["num_rows_subsample"] = args.num_rows_subsample

                score = ContingencySimilarity.compute(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        score = safe_float(score)
        rows.append(
            {
                "column_1": c1,
                "column_2": c2,
                "metric": metric_name,
                "score": score,
                "divergence": (1.0 - score) if pd.notna(score) else np.nan,
                "is_numeric_1": c1 in numeric_cols,
                "is_numeric_2": c2 in numeric_cols,
                "error": error,
            }
        )

    long_df = pd.DataFrame(rows)
    score_matrix = build_matrix(long_df, ordered_columns, "score", diagonal=1.0)
    divergence_matrix = build_matrix(long_df, ordered_columns, "divergence", diagonal=0.0)

    out_dir = args.output_dir / args.dataset_name / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(out_dir / "pair_trends_manual_long.csv", index=False)
    score_matrix.to_csv(out_dir / "pair_trends_manual_score_matrix.csv")
    divergence_matrix.to_csv(out_dir / "pair_trends_manual_divergence_matrix.csv")

    summary = {
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "real_path": str(real_path.resolve()),
        "metadata_path": str(meta_path.resolve()),
        "pred_path": str(args.pred_path.resolve()),
        "n_rows_real": int(len(real_df)),
        "n_rows_synthetic": int(len(syn_df)),
        "n_columns": int(len(ordered_columns)),
        "train_filename": args.train_filename,
        "info_filename": args.info_filename,
        "correlation_coefficient": args.correlation_coefficient,
        "real_correlation_threshold": args.real_correlation_threshold,
        "real_association_threshold": args.real_association_threshold,
        "num_discrete_bins": args.num_discrete_bins,
        "num_rows_subsample": args.num_rows_subsample,
        "mean_score_ignoring_nan": safe_float(long_df["score"].mean(skipna=True)),
        "num_nan_scores": int(long_df["score"].isna().sum()),
        "output_dir": str(out_dir.resolve()),
    }
    with open(out_dir / "summary_manual.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()