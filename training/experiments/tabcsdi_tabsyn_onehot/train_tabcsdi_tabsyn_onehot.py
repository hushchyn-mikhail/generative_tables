import argparse
import os
import json
import yaml
import datetime
import torch
import types

import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TABCSDI_DIR = os.path.join(PROJECT_ROOT, "submodules", "TabCSDI")

sys.path.insert(0, TABCSDI_DIR)
os.chdir(TABCSDI_DIR)

from src.main_model_table import TabCSDI
from src.utils_table import train

from dataset_tabsyn_onehot_generic import get_dataloader_tabsyn_onehot


def patch_train_mask_ratio(model: TabCSDI, train_missing_ratio: float):
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)

        for i in range(len(observed_mask)):
            sample_ratio = train_missing_ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            if num_masked > 0:
                rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1

        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    model.get_randmask = types.MethodType(get_randmask, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, required=True)
    parser.add_argument("--config", type=str, default="census_onehot_analog.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--missingratio", type=float, default=0.2)
    parser.add_argument("--trainmissingratio", type=float, default=0.2)
    parser.add_argument("--tabsyn_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--modelfolder", type=str, default="")
    args = parser.parse_args()
    print(args)

    path = os.path.join("config", args.config)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["test_missing_ratio"] = args.missingratio
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = f"./save/{args.dataname}_tabsyn_onehot_{current_time}/"
    os.makedirs(foldername, exist_ok=True)
    with open(os.path.join(foldername, "config.json"), "w") as f:
        json.dump({"args": vars(args), "config": config}, f, indent=2)

    train_loader, valid_loader, _test_loader = get_dataloader_tabsyn_onehot(
        tabsyn_root=args.tabsyn_root,
        dataname=args.dataname,
        seed=args.seed,
        batch_size=config["train"]["batch_size"],
        missing_ratio=args.missingratio,
        val_ratio=0.1,
    )

    model = TabCSDI(config, args.device).to(args.device)
    patch_train_mask_ratio(model, args.trainmissingratio)

    if args.modelfolder == "":
        train(model, config["train"], train_loader, valid_loader=valid_loader, valid_epoch_interval=100, foldername=foldername)
    else:
        model.load_state_dict(torch.load(os.path.join("./save", args.modelfolder, "model.pth"), map_location=args.device))

    torch.save(model.state_dict(), os.path.join(foldername, "model.pth"))
    print("Saved:", os.path.join(foldername, "model.pth"))


if __name__ == "__main__":
    main()