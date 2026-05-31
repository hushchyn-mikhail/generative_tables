import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model, build_block_noise_scale_init
from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')


def main(args): 
    device = args.device

    train_z, _, _, ckpt_path, info = get_input_train(args)

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1]
    token_dim = info['token_dim']

    mean = train_z.mean(0)

    train_z = (train_z - mean) / 2
    train_data = train_z

    noise_scale_init = build_block_noise_scale_init(
        train_data,
        token_dim=token_dim,
        mode=args.block_noise_mode,
        min_scale=args.block_noise_min,
        max_scale=args.block_noise_max,
    )


    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = 5000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)
# старый где была ошибка у learned
#     model = Model(
#         denoise_fn=denoise_fn,
#         hid_dim=train_z.shape[1],
#         noise_scale_init=noise_scale_init.to(device),
#         learn_noise_scale=args.block_noise_mode == 'learned',
#     ).to(device)
    
    model = Model(
        denoise_fn=denoise_fn,
        hid_dim=in_dim,
        token_dim=token_dim,
        noise_scale_init=noise_scale_init,
        learn_noise_scale=(args.block_noise_mode == "learned"),
    ).to(device)
    # старый где была ошибка у learned
    # if args.block_noise_mode != 'none':
    #     expanded_scale = model.denoise_fn_D.get_noise_scale().detach().cpu().view(-1)
    #     block_scale = expanded_scale.view(-1, token_dim).mean(dim=1)
    #     print(f'Using blockwise latent noise scales ({args.block_noise_mode}): {block_scale.tolist()}')
    
    if args.block_noise_mode != 'none':
        block_scale = model.denoise_fn_D.get_block_noise_scale().detach().cpu().view(-1)
        print(f'Using blockwise latent noise scales ({args.block_noise_mode}): {block_scale.tolist()}')

    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    if args.block_noise_mode == "learned":
        scale_params = [model.denoise_fn_D.log_noise_scale]
        main_params = [p for n, p in model.named_parameters() if n != "denoise_fn_D.log_noise_scale"]
        optimizer = torch.optim.Adam(
            [
                {"params": main_params, "lr": 1e-3, "weight_decay": 0.0},
                {"params": scale_params, "lr": 1e-4, "weight_decay": 0.0},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    best_loss = float('inf')
    patience = 0
    start_time = time.time()
    for epoch in range(num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)

        if curr_loss < best_loss:
            best_loss = curr_loss
            patience = 0
            torch.save(model.state_dict(), f'{ckpt_path}/model.pt')
        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--block_noise_mode', type=str, default='data', choices=['none', 'data', 'learned'], help='Blockwise latent noise scaling: none, data-driven fixed, or data-initialized learned.')
    parser.add_argument('--block_noise_min', type=float, default=0.5, help='Lower clamp for blockwise latent noise amplitudes.')
    parser.add_argument('--block_noise_max', type=float, default=1.5, help='Upper clamp for blockwise latent noise amplitudes.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    main(args)
