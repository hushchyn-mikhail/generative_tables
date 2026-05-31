import argparse
import json
import shutil
from pathlib import Path

TEMPLATES = {
    "adult": {
        "task_type": "binclass",
        "header": "infer",
        "column_names": [
            "age", "workclass", "fnlwgt", "education", "education.num",
            "marital.status", "occupation", "relationship", "race", "sex",
            "capital.gain", "capital.loss", "hours.per.week", "native.country", "income"
        ],
        "num_col_idx": [0, 2, 4, 10, 11, 12],
        "cat_col_idx": [1, 3, 5, 6, 7, 8, 9, 13],
        "target_col_idx": [14],
        "file_type": "csv",
    },
    "beijing": {
        "task_type": "regression",
        "header": "infer",
        "column_names": None,
        "num_col_idx": [5, 6, 7, 9, 10, 11],
        "cat_col_idx": [0, 1, 2, 3, 8],
        "target_col_idx": [4],
        "file_type": "csv",
    },
    "shoppers": {
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": list(range(10)),
        "cat_col_idx": [10, 11, 12, 13, 14, 15, 16],
        "target_col_idx": [17],
        "file_type": "csv",
    },
    "default": {
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": [0, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "cat_col_idx": [1, 2, 3, 5, 6, 7, 8, 9, 10],
        "target_col_idx": [23],
        "file_type": "csv",
    },
    "magic": {
        "task_type": "binclass",
        "header": "infer",
        "column_names": [
            "Length", "Width", "Size", "Conc", "Conc1", "Asym",
            "M3Long", "M3Trans", "Alpha", "Dist", "class"
        ],
        "num_col_idx": list(range(10)),
        "cat_col_idx": [],
        "target_col_idx": [10],
        "file_type": "csv",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TabDiff data directory from existing train.csv/test.csv splits")
    parser.add_argument("--dataset", required=True, choices=sorted(TEMPLATES.keys()))
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of symlink")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    train_src = Path(args.train_csv).resolve()
    test_src = Path(args.test_csv).resolve()

    if not train_src.exists():
        raise FileNotFoundError(f"train_csv not found: {train_src}")
    if not test_src.exists():
        raise FileNotFoundError(f"test_csv not found: {test_src}")

    dataset = args.dataset
    data_dir = repo_root / "data" / dataset
    info_dir = repo_root / "data" / "Info"
    data_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    train_dst = data_dir / "train.csv"
    test_dst = data_dir / "test.csv"

    for dst in [train_dst, test_dst]:
        if dst.exists() or dst.is_symlink():
            dst.unlink()

    if args.copy:
        shutil.copy2(train_src, train_dst)
        shutil.copy2(test_src, test_dst)
    else:
        train_dst.symlink_to(train_src)
        test_dst.symlink_to(test_src)

    info = dict(TEMPLATES[dataset])
    info.update({
        "name": dataset,
        "data_path": f"data/{dataset}/train.csv",
        "test_path": f"data/{dataset}/test.csv",
        "val_path": None,
    })

    info_path = info_dir / f"{dataset}.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(f"Prepared dataset '{dataset}'")
    print(f"  train -> {train_dst}")
    print(f"  test  -> {test_dst}")
    print(f"  info  -> {info_path}")
    print("Next step: python process_dataset.py --dataname " + dataset)


if __name__ == "__main__":
    main()
