"""
python plot_pair_trends_heatmap.py \
  --metrics-root ./metrics/correlation/baseline \
  --output ./metrics/correlation/baseline/pair_trends_heatmap.png \
  --datasets adult default magic beijing \
  --models tabsyn tabdiff tabcsdi catboost
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm


DEFAULT_DATASETS = ["adult", "default", "magic", "beijing"]
DEFAULT_MODELS = ["tabsyn", "tabdiff", "tabcsdi", "catboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset folder names in row order",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
    )
    parser.add_argument(
        "--matrix-filename",
        type=str,
        default="pair_trends_manual_divergence_matrix.csv",
    )
    parser.add_argument(
        "--long-filename",
        type=str,
        default="pair_trends_manual_long.csv",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Greens",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pair-wise column divergence",
    )
    parser.add_argument(
        "--show-block-separator",
        action="store_true",
        help="Draw separator lines between numeric and categorical blocks.",
    )
    parser.add_argument(
        "--show-missing-cross",
        action="store_true",
        help="Draw a red X for missing dataset/model cells.",
    )
    parser.add_argument(
        "--show-ticks",
        action="store_true",
        help="Show reordered column names on every subplot.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def load_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def infer_column_types(long_df: pd.DataFrame) -> Dict[str, bool]:
    type_map: Dict[str, bool] = {}
    for _, row in long_df.iterrows():
        c1 = str(row["column_1"])
        c2 = str(row["column_2"])
        type_map.setdefault(c1, bool(row["is_numeric_1"]))
        type_map.setdefault(c2, bool(row["is_numeric_2"]))
    return type_map


def derive_dataset_order(dataset_dir: Path, models: List[str], matrix_filename: str, long_filename: str) -> Tuple[List[str], int]:
    for model in models:
        long_path = dataset_dir / model / long_filename
        matrix_path = dataset_dir / model / matrix_filename
        if long_path.exists() and matrix_path.exists():
            long_df = load_long(long_path)
            matrix = load_matrix(matrix_path)
            type_map = infer_column_types(long_df)

            columns = [str(c) for c in matrix.index]
            numeric_cols = [c for c in columns if type_map.get(c, False)]
            categorical_cols = [c for c in columns if not type_map.get(c, False)]
            ordered_columns = numeric_cols + categorical_cols
            return ordered_columns, len(numeric_cols)


def reorder_matrix(matrix: pd.DataFrame, ordered_columns: List[str]) -> pd.DataFrame:
    return matrix.loc[ordered_columns, ordered_columns]


def prettify_model_name(name: str) -> str:
    mapping = {
        "tabsyn": "TabSyn",
        "tabdiff": "TabDiff",
        "tabcsdi": "TabCSDI",
        "catboost": "CatBoost",
    }
    return mapping.get(name.lower(), name)


def prettify_dataset_name(name: str) -> str:
    mapping = {
        "adult": "Adult",
        "default": "Default",
        "magic": "Magic",
        "beijing": "Beijing",
        "shoppers": "Shoppers",
    }
    return mapping.get(name.lower(), name)


def collect_global_vmax(metrics_root: Path, datasets: List[str], models: List[str], matrix_filename: str, dataset_orders: Dict[str, List[str]]) -> float:
    vals = []
    for dataset in datasets:
        ordered_columns = dataset_orders[dataset]
        for model in models:
            path = metrics_root / dataset / model / matrix_filename
            if not path.exists():
                continue
            matrix = load_matrix(path)
            matrix = reorder_matrix(matrix, ordered_columns)
            arr = matrix.to_numpy(dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                vals.append(float(np.nanmax(finite)))
    return max(vals) if vals else 1.0


def main() -> None:
    args = parse_args()
    dataset_orders: Dict[str, List[str]] = {}
    dataset_splits: Dict[str, int] = {}
    available_datasets = []

    for dataset in args.datasets:
        dataset_dir = args.metrics_root / dataset
        if not dataset_dir.exists():
            continue
        ordered_columns, split_idx = derive_dataset_order(
            dataset_dir, args.models, args.matrix_filename, args.long_filename
        )
        dataset_orders[dataset] = ordered_columns
        dataset_splits[dataset] = split_idx
        available_datasets.append(dataset)


    vmax = args.vmax if args.vmax is not None else collect_global_vmax(
        args.metrics_root,
        available_datasets,
        args.models,
        args.matrix_filename,
        dataset_orders,
    )
    norm = PowerNorm(gamma=args.gamma, vmin=args.vmin, vmax=vmax)

    n_rows = len(available_datasets)
    n_cols = len(args.models)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.3 * n_cols, 2.3 * n_rows),
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.18, hspace=0.25, right=0.92, top=0.90, bottom=0.08)

    im = None

    for i, dataset in enumerate(available_datasets):
        ordered_columns = dataset_orders[dataset]
        split_idx = dataset_splits[dataset]

        for j, model in enumerate(args.models):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            matrix_path = args.metrics_root / dataset / model / args.matrix_filename

            if not matrix_path.exists():
                ax.set_facecolor("white")
                if args.show_missing_cross:
                    ax.plot([0.25, 0.75], [0.25, 0.75], transform=ax.transAxes, color="red", lw=0.8)
                    ax.plot([0.25, 0.75], [0.75, 0.25], transform=ax.transAxes, color="red", lw=0.8)
                if i == 0:
                    ax.set_title(prettify_model_name(model), fontsize=11)
                if j == 0:
                    ax.set_ylabel(prettify_dataset_name(dataset), fontsize=11)
                continue

            matrix = load_matrix(matrix_path)
            matrix = reorder_matrix(matrix, ordered_columns)
            arr = matrix.to_numpy(dtype=float)

            im = ax.imshow(
                arr,
                cmap=args.cmap,
                norm=norm,
                interpolation="nearest",
            )

            if i == 0:
                ax.set_title(prettify_model_name(model), fontsize=11)

            if j == 0:
                ax.set_ylabel(prettify_dataset_name(dataset), fontsize=11)

            if args.show_ticks:
                ticks = np.arange(len(ordered_columns))
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_xticklabels(ordered_columns, rotation=90, fontsize=6)
                ax.set_yticklabels(ordered_columns, fontsize=6)

            if args.show_block_separator and 0 < split_idx < len(ordered_columns):
                ax.axhline(split_idx - 0.5, linewidth=0.8)
                ax.axvline(split_idx - 0.5, linewidth=0.8)

            ax.set_aspect("equal")

    if im is None:
        raise RuntimeError("No heatmaps were rendered. Check your metrics folder structure.")

    cax = fig.add_axes([0.94, 0.15, 0.02, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Divergence (1 - score)", rotation=90)

    fig.suptitle(args.title, fontsize=13)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {args.output.resolve()}")
    print("Datasets:")
    for dataset in available_datasets:
        print(f" - {dataset}: {len(dataset_orders[dataset])} columns "
              f"({dataset_splits[dataset]} numeric, {len(dataset_orders[dataset]) - dataset_splits[dataset]} categorical)")
    print("Models:")
    for model in args.models:
        print(f" - {model}")


if __name__ == "__main__":
    main()
