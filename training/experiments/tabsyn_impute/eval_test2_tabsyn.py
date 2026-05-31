
import argparse
import os
import json
import datetime
import numpy as np
import pandas as pd

from tabsyn_imputer import tabsyn_impute


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--tabsyn_root", required=True)
    p.add_argument(
        "--dataname",
        required=True,
        choices=["adult", "default", "shoppers", "magic", "beijing"]
    )
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--resample_steps", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cat_decoder_type", type=str, default='mlp')

    p.add_argument("--outdir", required=True)
    p.add_argument("--outfile", type=str, default=None)

    args = p.parse_args()

    if args.outfile is None:
        args.outfile = f"prediction_{args.dataname}.csv"

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

    cols = list(test_df.columns)
    N, D = test_df.shape

    print(f"Dataset: {args.dataname}")
    print(f"Test shape: {test_df.shape}")
    print(f"Columns: {D}")

    prediction_df = test_df.copy()

    for j, name in enumerate(cols):
        missing_mask = np.zeros((N, D), dtype=bool)
        missing_mask[:, j] = True  # маскируем только одну колонку в test

        print(f"Обработка колонки {j + 1}/{D}: {name}")

        syn_df = tabsyn_impute(
            tabsyn_root=args.tabsyn_root,
            dataname=args.dataname,
            missing_cols_mask=missing_mask,
            gpu=args.gpu,
            num_steps=args.num_steps,
            resample_steps=args.resample_steps,
            seed=args.seed,
            cat_decoder_type=args.cat_decoder_type
        )

        if not isinstance(syn_df, pd.DataFrame):
            syn_df = pd.DataFrame(syn_df, columns=test_df.columns)

        if syn_df.shape != test_df.shape:
            raise ValueError(
                f"tabsyn_impute вернул DataFrame неправильной формы "
                f"для колонки {name}: got {syn_df.shape}, expected {test_df.shape}"
            )

        if list(syn_df.columns) != list(test_df.columns):
            syn_df.columns = test_df.columns

        # в итоговый prediction записываем только текущую предсказанную колонку
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