from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tabsyn.diffusion_utils import EDMLoss

ModuleType = Union[str, Callable[..., nn.Module]]


# def build_block_noise_scale_init(
#     train_z: Tensor,
#     token_dim: int,
#     mode: str = 'data',
#     min_scale: float = 0.5,
#     max_scale: float = 1.5,
#     eps: float = 1e-6,
# ) -> Tensor:
#     """Build per-column noise amplitudes a_i and expand them to flattened dims.

#     We keep the original scalar time/noise schedule sigma(t), and only modulate it
#     by a blockwise amplitude a_i for each latent token (column block):
#         sigma_i(t) = a_i * sigma(t)

#     For the default data-driven mode, a_i is proportional to the empirical latent
#     dispersion of the corresponding token block after the TabSyn normalization.
#     This preserves the original architecture while making the perturbation scale
#     aware of the column-wise latent geometry.
#     """
#     if train_z.ndim != 2:
#         raise ValueError(f'train_z must be a 2D tensor, got shape={tuple(train_z.shape)}')
#     if train_z.shape[1] % token_dim != 0:
#         raise ValueError(
#             f'Flattened latent dim ({train_z.shape[1]}) is not divisible by token_dim ({token_dim}).'
#         )

#     if mode == 'none':
#         return torch.ones(train_z.shape[1], dtype=train_z.dtype, device=train_z.device)

#     z_blocks = train_z.view(train_z.shape[0], -1, token_dim)
#     block_std = z_blocks.std(dim=0, unbiased=False).mean(dim=-1)
#     block_std = block_std.clamp_min(eps)
#     block_scale = block_std / block_std.mean().clamp_min(eps)
#     block_scale = block_scale.clamp(min=min_scale, max=max_scale)
#     return block_scale.repeat_interleave(token_dim)

from typing import Optional, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tabsyn.diffusion_utils import EDMLoss

# новая функция для learnable sch (fixed)
def build_block_noise_scale_init(
    train_z: Tensor,
    token_dim: int,
    mode: str = "data",
    min_scale: float = 0.5,
    max_scale: float = 1.5,
    eps: float = 1e-6,
) -> Tensor:
    """
    Возвращает scale НА УРОВНЕ BLOCK/TOKEN, shape = [num_blocks].

    Если latent имеет shape [B, M*token_dim], то:
      num_blocks = M
      один scale соответствует одному токену колонки
    """
    if train_z.ndim != 2:
        raise ValueError(f"train_z must be 2D, got shape={tuple(train_z.shape)}")
    if token_dim <= 0:
        raise ValueError(f"token_dim must be positive, got {token_dim}")
    if train_z.shape[1] % token_dim != 0:
        raise ValueError(
            f"Flattened latent dim ({train_z.shape[1]}) is not divisible by token_dim ({token_dim})."
        )

    num_blocks = train_z.shape[1] // token_dim

    if mode == "none":
        return torch.ones(num_blocks, dtype=train_z.dtype, device=train_z.device)

    if mode not in {"data", "learned"}:
        raise ValueError(f"Unknown mode={mode!r}. Expected 'none', 'data', or 'learned'.")

    # [B, M, token_dim]
    z_blocks = train_z.view(train_z.shape[0], num_blocks, token_dim)

    # Один scale на block: средний std по координатам блока
    block_std = z_blocks.std(dim=0, unbiased=False).mean(dim=-1)   # [M]
    block_std = block_std.clamp_min(eps)

    # Нормировка относительно среднего
    block_scale = block_std / block_std.mean().clamp_min(eps)

    # Ограничиваем диапазон
    block_scale = block_scale.clamp(min=min_scale, max=max_scale)

    return block_scale


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

def reglu(x: Tensor) -> Tensor:
    """The ReGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """The GEGLU activation function from [1].
    References:
        [1] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = ReGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu].

    Examples:
        .. testcode::

            module = GEGLU()
            x = torch.randn(3, 4)
            assert module(x).shape == (3, 2)

    References:
        * [shazeer2020glu] Noam Shazeer, "GLU Variants Improve Transformer", 2020
    """

    def forward(self, x: Tensor) -> Tensor:
        return geglu(x)


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        )

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
    
    def forward(self, x, noise_labels, class_labels=None):
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)
    
        x = self.proj(x) + emb
        return self.mlp(x)

# старая версия где learned был с ошибкой
# class Precond(nn.Module):
#     def __init__(self,
#         denoise_fn,
#         hid_dim,
#         sigma_min = 0,                # Minimum supported noise level.
#         sigma_max = float('inf'),     # Maximum supported noise level.
#         sigma_data = 0.5,              # Expected standard deviation of the training data.
#         noise_scale_init: Optional[Tensor] = None,
#         learn_noise_scale: bool = False,
#     ):
#         super().__init__()

#         self.hid_dim = hid_dim
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.sigma_data = sigma_data
#         ###########
#         self.denoise_fn_F = denoise_fn

#         if noise_scale_init is None:
#             noise_scale_init = torch.ones(hid_dim, dtype=torch.float32)
#         noise_scale_init = torch.as_tensor(noise_scale_init, dtype=torch.float32)
#         if noise_scale_init.ndim == 1:
#             if noise_scale_init.numel() != hid_dim:
#                 raise ValueError(
#                     f'noise_scale_init must have {hid_dim} entries, got {noise_scale_init.numel()}.'
#                 )
#             noise_scale_init = noise_scale_init.unsqueeze(0)
#         elif noise_scale_init.shape != (1, hid_dim):
#             raise ValueError(
#                 f'noise_scale_init must have shape ({hid_dim},) or (1, {hid_dim}), '
#                 f'got {tuple(noise_scale_init.shape)}.'
#             )
#         self.log_noise_scale = nn.Parameter(noise_scale_init.clamp_min(1e-6).log())
#         self.log_noise_scale.requires_grad_(learn_noise_scale)

#     def get_noise_scale(self) -> Tensor:
#         return self.log_noise_scale.exp()

#     def _reshape_sigma(self, sigma: Tensor) -> Tensor:
#         sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.log_noise_scale.device)
#         if sigma.ndim == 0:
#             sigma = sigma.unsqueeze(0)
#         return sigma.reshape(-1, 1)

#     def get_effective_sigma(self, sigma: Tensor) -> Tensor:
#         return self._reshape_sigma(sigma) * self.get_noise_scale()

#     def scale_noise(self, noise: Tensor, sigma: Tensor) -> Tensor:
#         return noise.to(torch.float32) * self.get_effective_sigma(sigma)

#     def forward(self, x, sigma):

#         x = x.to(torch.float32)

#         sigma = self._reshape_sigma(sigma)
#         sigma_eff = self.get_effective_sigma(sigma)
#         dtype = torch.float32

#         c_skip = self.sigma_data ** 2 / (sigma_eff ** 2 + self.sigma_data ** 2)
#         c_out = sigma_eff * self.sigma_data / (sigma_eff ** 2 + self.sigma_data ** 2).sqrt()
#         c_in = 1 / (self.sigma_data ** 2 + sigma_eff ** 2).sqrt()
#         c_noise = sigma.log() / 4

#         x_in = c_in * x
#         F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

#         assert F_x.dtype == dtype
#         D_x = c_skip * x + c_out * F_x.to(torch.float32)
#         return D_x

#     def round_sigma(self, sigma):
#         return torch.as_tensor(sigma)
    

class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        token_dim,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        noise_scale_init: Optional[Tensor] = None,
        learn_noise_scale: bool = False,
    ):
        super().__init__()

        if hid_dim % token_dim != 0:
            raise ValueError(
                f"hid_dim ({hid_dim}) must be divisible by token_dim ({token_dim})."
            )

        self.hid_dim = hid_dim
        self.token_dim = token_dim
        self.num_blocks = hid_dim // token_dim

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.denoise_fn_F = denoise_fn

        # Храним scale ИМЕННО ПО BLOCK'АМ: shape [1, M, 1]
        if noise_scale_init is None:
            noise_scale_init = torch.ones(self.num_blocks, dtype=torch.float32)

        noise_scale_init = torch.as_tensor(noise_scale_init, dtype=torch.float32)

        if noise_scale_init.ndim == 1:
            if noise_scale_init.numel() != self.num_blocks:
                raise ValueError(
                    f"noise_scale_init must have {self.num_blocks} entries, "
                    f"got {noise_scale_init.numel()}."
                )
            noise_scale_init = noise_scale_init.view(1, self.num_blocks, 1)

        elif noise_scale_init.ndim == 3:
            if noise_scale_init.shape != (1, self.num_blocks, 1):
                raise ValueError(
                    f"noise_scale_init must have shape (1, {self.num_blocks}, 1), "
                    f"got {tuple(noise_scale_init.shape)}."
                )
        else:
            raise ValueError(
                f"noise_scale_init must have shape ({self.num_blocks},) or "
                f"(1, {self.num_blocks}, 1), got {tuple(noise_scale_init.shape)}."
            )

        self.log_noise_scale = nn.Parameter(noise_scale_init.clamp_min(1e-6).log())
        self.log_noise_scale.requires_grad_(learn_noise_scale)

    def get_block_noise_scale(self) -> Tensor:
        """
        Возвращает scale по токенам/блокам, shape [1, M, 1]
        """
        return self.log_noise_scale.exp()

    def get_noise_scale(self) -> Tensor:
        """
        Возвращает expanded scale по flattened latent dims, shape [1, hid_dim]
        """
        block_scale = self.get_block_noise_scale()                 # [1, M, 1]
        full_scale = block_scale.repeat(1, 1, self.token_dim)     # [1, M, token_dim]
        return full_scale.view(1, self.hid_dim)                   # [1, M*token_dim]

    def _reshape_sigma(self, sigma: Tensor) -> Tensor:
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=self.log_noise_scale.device)
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        return sigma.reshape(-1, 1)   # [B, 1]

    def get_effective_sigma(self, sigma: Tensor) -> Tensor:
        """
        sigma: [B] or scalar
        return: [B, hid_dim]
        """
        sigma = self._reshape_sigma(sigma)         # [B, 1]
        return sigma * self.get_noise_scale()      # [B, hid_dim]

    def scale_noise(self, noise: Tensor, sigma: Tensor) -> Tensor:
        """
        noise: [B, hid_dim]
        sigma: scalar or [B]
        """
        return noise.to(torch.float32) * self.get_effective_sigma(sigma)

    def forward(self, x, sigma):
        x = x.to(torch.float32)

        sigma = self._reshape_sigma(sigma)              # [B,1]
        sigma_eff = self.get_effective_sigma(sigma)     # [B,hid_dim]
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma_eff ** 2 + self.sigma_data ** 2)
        c_out  = sigma_eff * self.sigma_data / (sigma_eff ** 2 + self.sigma_data ** 2).sqrt()
        c_in   = 1 / (self.sigma_data ** 2 + sigma_eff ** 2).sqrt()

        # Denoiser по-прежнему кондиционируем глобальным sigma(t), а не per-dim sigma_eff
        c_noise = sigma.log() / 4                      # [B,1]

        x_in = c_in * x
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
# старая версия где learned был с ошибкой
# class Model(nn.Module):
#     def __init__(
#         self,
#         denoise_fn,
#         hid_dim,
#         P_mean=-1.2,
#         P_std=1.2,
#         sigma_data=0.5,
#         gamma=5,
#         opts=None,
#         pfgmpp = False,
#         noise_scale_init: Optional[Tensor] = None,
#         learn_noise_scale: bool = False,
#     ):
#         super().__init__()

#         self.denoise_fn_D = Precond(
#             denoise_fn,
#             hid_dim,
#             sigma_data=sigma_data,
#             noise_scale_init=noise_scale_init,
#             learn_noise_scale=learn_noise_scale,
#         )
#         self.loss_fn = EDMLoss(P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None)

#     def forward(self, x):

#         loss = self.loss_fn(self.denoise_fn_D, x)
#         return loss.mean(-1).mean()
class Model(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        token_dim,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        gamma=5,
        opts=None,
        pfgmpp=False,
        noise_scale_init: Optional[Tensor] = None,
        learn_noise_scale: bool = False,
    ):
        super().__init__()

        self.denoise_fn_D = Precond(
            denoise_fn=denoise_fn,
            hid_dim=hid_dim,
            token_dim=token_dim,
            sigma_data=sigma_data,
            noise_scale_init=noise_scale_init,
            learn_noise_scale=learn_noise_scale,
        )

        self.loss_fn = EDMLoss(
            P_mean=P_mean,
            P_std=P_std,
            sigma_data=sigma_data,
            hid_dim=hid_dim,
            gamma=gamma,
            opts=opts,
        )

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()