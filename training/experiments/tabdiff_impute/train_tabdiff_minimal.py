import argparse
import glob
import os
import time
import json
import pickle
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

import src
from utils_train import TabDiffDataset, update_ema
from tabdiff.modules.main_modules import UniModMLP, Model
from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion


def _set_device(gpu: int) -> str:
    if gpu != -1 and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cpu"


def _default_exp_name(non_learnable_schedule: bool) -> str:
    return "non_learnable_schedule" if non_learnable_schedule else "learnable_schedule"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabsyn_root", required=True)
    p.add_argument("--tabdiff_root", required=True)
    p.add_argument("--dataname", required=True)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--non_learnable_schedule", action="store_true")
    p.add_argument("--exp_name", type=str, default=None)

    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--ema_decay", type=float, default=0.997)

    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = _set_device(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = os.path.join(args.tabsyn_root, "data", args.dataname)
    info_path = os.path.join(dataset_dir, "info.json")
    info = json.load(open(info_path, "r"))

    # load default config
    config_path = os.path.join(args.tabdiff_root, "tabdiff", "configs", "tabdiff_configs.toml")
    raw_config = src.load_config(config_path)

    if args.exp_name is None:
        args.exp_name = _default_exp_name(args.non_learnable_schedule)
        print(args.exp_name)

    # overrides
    if args.batch_size is not None:
        raw_config["train"]["main"]["batch_size"] = args.batch_size
    # как в статье
    args.batch_size = 4096
    
    if args.lr is not None:
        raw_config["train"]["main"]["lr"] = args.lr
    if args.weight_decay is not None:
        raw_config["train"]["main"]["weight_decay"] = args.weight_decay

    # dataset
    train_data = TabDiffDataset(
        dataname=args.dataname,
        data_dir=dataset_dir,
        info=info,
        isTrain=True,
        y_only=False,
        dequant_dist=raw_config["data"]["dequant_dist"],
        int_dequant_factor=raw_config["data"]["int_dequant_factor"],
    )
    train_loader = DataLoader(
        train_data,
        batch_size=raw_config["train"]["main"]["batch_size"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    d_numerical, categories = train_data.d_numerical, train_data.categories

    # model
    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = (categories + 1).tolist()

    backbone = UniModMLP(**raw_config["unimodmlp_params"])
    denoise_fn = Model(backbone, **raw_config["diffusion_params"]["edm_params"]).to(device)

    if not args.non_learnable_schedule:
        raw_config["diffusion_params"]["scheduler"] = "power_mean_per_column"
        raw_config["diffusion_params"]["cat_scheduler"] = "log_linear_per_column"

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=denoise_fn,
        y_only_model=None,
        device=device,
        **raw_config["diffusion_params"],
    ).to(device)

    opt = torch.optim.AdamW(
        diffusion.parameters(),
        lr=raw_config["train"]["main"]["lr"],
        weight_decay=raw_config["train"]["main"]["weight_decay"],
    )

    # EMA copies
    ema_model = deepcopy(diffusion._denoise_fn).to(device)
    for p_ in ema_model.parameters():
        p_.detach_()
    ema_num_schedule = deepcopy(diffusion.num_schedule).to(device)
    for p_ in ema_num_schedule.parameters():
        p_.detach_()
    ema_cat_schedule = deepcopy(diffusion.cat_schedule).to(device)
    for p_ in ema_cat_schedule.parameters():
        p_.detach_()

    # save dir
    save_dir = os.path.join(args.tabdiff_root, "tabdiff", "ckpt", args.dataname, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
        pickle.dump(raw_config, f)

    best_ema = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        diffusion.train()
        total = 0.0
        count = 0

        for batch in train_loader:
            x = batch.float().to(device)

            opt.zero_grad()
            dloss, closs = diffusion.mixed_loss(x)
            loss = dloss + closs
            loss.backward()
            opt.step()

            # EMA update (denoise + schedules)
            update_ema(list(ema_model.parameters()), list(diffusion._denoise_fn.parameters()), rate=args.ema_decay)
            update_ema(list(ema_num_schedule.parameters()), list(diffusion.num_schedule.parameters()), rate=args.ema_decay)
            update_ema(list(ema_cat_schedule.parameters()), list(diffusion.cat_schedule.parameters()), rate=args.ema_decay)

            total += float(loss.item()) * x.size(0)
            count += x.size(0)

        mean_loss = total / max(count, 1)

        if mean_loss < best_ema:
            best_ema = mean_loss
            ckpt = {
                "denoise_fn": ema_model.state_dict(),
                "num_schedule": ema_num_schedule.state_dict(),
                "cat_schedule": ema_cat_schedule.state_dict(),
            }
            # чистим старые best_ema_model
            for fp in glob.glob(os.path.join(save_dir, "best_ema_model*.pt")):
                try:
                    os.remove(fp)
                except OSError:
                    pass
            torch.save(ckpt, os.path.join(save_dir, f"best_ema_model_{best_ema:.6f}_{epoch}.pt"))

        if epoch % 50 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"[{args.dataname}] epoch={epoch}/{args.epochs} loss={mean_loss:.6f}" )

    print("DONE. Saved to:", save_dir)


if __name__ == "__main__":
    main()
