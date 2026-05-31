import os
import json
import pickle
import numpy as np
import pandas as pd
import category_encoders as ce

from torch.utils.data import Dataset, DataLoader


def _make_gt_mask(observed_masks: np.ndarray, missing_ratio: float, rng: np.random.Generator) -> np.ndarray:
    """
    Masking missing values in observed data by column.
    """
    masks = observed_masks.copy()
    n, d = masks.shape
    for col in range(d):
        obs_idx = np.where(masks[:, col])[0]
        if len(obs_idx) == 0:
            continue
        m = int(len(obs_idx) * missing_ratio)
        if m <= 0:
            continue
        miss_idx = rng.choice(obs_idx, size=m, replace=False)
        masks[miss_idx, col] = False
    return masks


def _prepare_onehot_from_tabsyn(
    tabsyn_root: str,
    missing_ratio: float,
    seed: int,
    cache_dir: str,
):
    """
    Load Adult train/test from TabSyn, reorder columns to [cont...cat...], create observed_mask/gt_mask,
    one-hot encode categorical columns (including target income), expand masks, normalize cont cols using train.
    """
    data_dir = os.path.join(tabsyn_root, "data", "adult")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    info_json = os.path.join(data_dir, "info.json")

    with open(info_json, "r") as f:
        info = json.load(f)

    num_col_idx = info["num_col_idx"]               # [0,2,4,10,11,12]
    cat_col_idx = info["cat_col_idx"]               # [1,3,5,6,7,8,9,13]
    target_col_idx = info["target_col_idx"]         # [14]

    cat_all_idx = list(cat_col_idx) + list(target_col_idx)
    col_names = info["column_names"]

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # reorder: numerical(cont) then categorical (+ target)
    cont_names = [col_names[i] for i in num_col_idx]
    cat_names = [col_names[i] for i in cat_all_idx]
    ordered_names = cont_names + cat_names

    train_df = train_df[ordered_names].copy()
    test_df = test_df[ordered_names].copy()

    # rename cols ;ike (0...d-1)
    train_df.columns = list(range(train_df.shape[1]))
    test_df.columns = list(range(test_df.shape[1]))

    cont_len = len(cont_names)
    # positions of cat features after reorder
    cat_list = list(range(cont_len, train_df.shape[1])) 

    def _is_nan_string(x):
        return isinstance(x, str) and x.strip().lower() == "nan"

    # # to make cat features with "nan" values - missing
    # I need to check it
    # train_df[cat_list] = train_df[cat_list].applymap(lambda x: np.nan if _is_nan_string(x) else x)
    # test_df[cat_list]  = test_df[cat_list].applymap(lambda x: np.nan if _is_nan_string(x) else x)

    # cat features with "nan" values are not missing
    train_obs_mask = ~pd.isnull(train_df)
    test_obs_mask = ~pd.isnull(test_df)
    train_obs_mask = train_obs_mask.values
    test_obs_mask = test_obs_mask.values

    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 10000)

    # we have dynamic masking on train  (!!!!) ? 

    train_gt_mask = _make_gt_mask(train_obs_mask, missing_ratio, rng_train)
    test_gt_mask = _make_gt_mask(test_obs_mask, missing_ratio, rng_test)

    # One-hot encode categorical columns (fit on train only)
    encoder = ce.one_hot.OneHotEncoder(cols=cat_list, use_cat_names=True)
    encoder.fit(train_df)

    train_enc = encoder.transform(train_df)
    test_enc = encoder.transform(test_df)

    # Expand masks to match encoded columns
    def expand_masks(df_enc, obs_mask, gt_mask):
        cum_num_bits = 0
        new_obs = obs_mask.copy()
        new_gt = gt_mask.copy()

        for col in cat_list:
            corresponding_cols = len([c for c in df_enc.columns if isinstance(c, str) and c.startswith(str(col) + "_")])
            add_col_num = corresponding_cols
            insert_obs = obs_mask[:, col]
            insert_gt = gt_mask[:, col]

            # original col already accounts for 1 column; insert extra (add_col_num-1)
            for _ in range(add_col_num - 1):
                new_obs = np.insert(new_obs, cum_num_bits + col, insert_obs, axis=1)
                new_gt = np.insert(new_gt, cum_num_bits + col, insert_gt, axis=1)

            cum_num_bits += add_col_num - 1

        return new_obs, new_gt

    train_obs_mask_exp, train_gt_mask_exp = expand_masks(train_enc, train_obs_mask, train_gt_mask)
    test_obs_mask_exp, test_gt_mask_exp = expand_masks(test_enc, test_obs_mask, test_gt_mask)

    # turn 0 -> -1 for cat features
    def finalize_values(df_enc, cont_len_):
        values = df_enc.values
        cat_part = values[:, cont_len_:]
        cat_part[cat_part == 0] = -1
        values[:, cont_len_:] = cat_part
        values = np.nan_to_num(values).astype(float)
        return values

    train_vals = finalize_values(train_enc, cont_len)
    test_vals = finalize_values(test_enc, cont_len)

    # col_index (reorded) ->  indices in one-hot matrix
    saved_cat_dict = {}
    for col in cat_list:
        idxs = [i for i, c in enumerate(train_enc.columns) if isinstance(c, str) and c.startswith(str(col))]
        saved_cat_dict[str(col)] = idxs

    cont_cols = list(range(cont_len))

    max_arr = np.zeros(len(cont_cols))
    min_arr = np.zeros(len(cont_cols))

    for j, k in enumerate(cont_cols):
        obs = train_obs_mask_exp[:, k].astype(bool)
        if obs.sum() == 0:
            max_arr[j] = 1.0
            min_arr[j] = 0.0
        else:
            tmp = train_vals[:, k]
            max_arr[j] = float(tmp[obs].max())
            min_arr[j] = float(tmp[obs].min())

    # normalization: 0 is missing
    def apply_norm(vals, obs_mask_exp):
        vals = vals.copy()
        for j, k in enumerate(cont_cols):
            vals[:, k] = ((vals[:, k] - (min_arr[j] - 1.0)) / (max_arr[j] - min_arr[j] + 1.0)) * obs_mask_exp[:, k]
        return vals

    train_vals = apply_norm(train_vals, train_obs_mask_exp)
    test_vals = apply_norm(test_vals, test_obs_mask_exp)

    os.makedirs(cache_dir, exist_ok=True)


    with open(os.path.join(cache_dir, "transformed_columns.pk"), "wb") as f:
        pickle.dump([cont_cols, saved_cat_dict], f)

    with open(os.path.join(cache_dir, "encoder.pk"), "wb") as f:
        pickle.dump(encoder, f)

    payload = {
        "train_vals": train_vals,
        "train_obs_mask": train_obs_mask_exp.astype(bool),
        "train_gt_mask": train_gt_mask_exp.astype(bool),
        "test_vals": test_vals,
        "test_obs_mask": test_obs_mask_exp.astype(bool),
        "test_gt_mask": test_gt_mask_exp.astype(bool),
        "cont_cols": cont_cols,
        "saved_cat_dict": saved_cat_dict,
        "ordered_names": ordered_names, 
    }
    return payload


class AdultTabSynOneHot(Dataset):
    def __init__(
        self,
        split: str,
        tabsyn_root: str,
        missing_ratio: float = 0.2,
        seed: int = 1,
        val_ratio: float = 0.1,
    ):
        self.split = split

        self.cache_dir = "./data_adult_tabsyn_onehot"
        os.makedirs(self.cache_dir, exist_ok=True)

        cache_file = os.path.join(self.cache_dir, f"adult_tabsyn_onehot_miss{missing_ratio}_seed{seed}.pk")

        if not os.path.isfile(cache_file):
            payload = _prepare_onehot_from_tabsyn(
                tabsyn_root=tabsyn_root,
                missing_ratio=missing_ratio,
                seed=seed,
                cache_dir=self.cache_dir,
            )
            with open(cache_file, "wb") as f:
                pickle.dump(payload, f)
        else:
            with open(cache_file, "rb") as f:
                payload = pickle.load(f)

        train_vals = payload["train_vals"]
        train_obs = payload["train_obs_mask"]
        train_gt = payload["train_gt_mask"]

        test_vals = payload["test_vals"]
        test_obs = payload["test_obs_mask"]
        test_gt = payload["test_gt_mask"]

        
        rng = np.random.default_rng(seed + 2026)
        n_train = train_vals.shape[0]
        idx = np.arange(n_train)
        rng.shuffle(idx)
        n_val = int(n_train * val_ratio)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        if split == "train":
            self.values = train_vals[tr_idx]
            self.observed_mask = train_obs[tr_idx]
            self.gt_mask = train_gt[tr_idx]
        elif split == "val":
            self.values = train_vals[val_idx]
            self.observed_mask = train_obs[val_idx]
            self.gt_mask = train_gt[val_idx]
        else:
            self.values = test_vals
            self.observed_mask = test_obs
            self.gt_mask = test_gt

        self.eval_length = self.values.shape[1]

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, idx):
        return {
            "observed_data": self.values[idx],
            "observed_mask": self.observed_mask[idx],
            "gt_mask": self.gt_mask[idx],
            "timepoints": np.arange(self.eval_length),
        }


def get_dataloader_adult_tabsyn_onehot(
    tabsyn_root: str,
    seed: int = 1,
    batch_size: int = 64,
    missing_ratio: float = 0.2,
    val_ratio: float = 0.1,
):
    train_ds = AdultTabSynOneHot("train", tabsyn_root, missing_ratio, seed, val_ratio)
    val_ds = AdultTabSynOneHot("val", tabsyn_root, missing_ratio, seed, val_ratio)
    test_ds = AdultTabSynOneHot("test", tabsyn_root, missing_ratio, seed, val_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print("Adult(TabSyn) onehot:")
    print("  train:", len(train_ds), "val:", len(val_ds), "test:", len(test_ds))
    
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        obs = ds.observed_mask.astype(np.float32)
        gt = ds.gt_mask.astype(np.float32)
        masked = (obs - gt).sum()
        denom = obs.sum() + 1e-9
        print(f"  {name} gt-masked rate:", float(masked / denom))

    return train_loader, val_loader, test_loader
