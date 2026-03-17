import torch
from torch import nn

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


# https://arxiv.org/abs/2203.05556
class SinusoidalEmbedding(nn.Module):
    def __init__(
        self, features: int, embed_dim: int, moments: int = 10, std: float = 1.0
    ):
        super().__init__()

        self.c = nn.Parameter(torch.empty(features, moments, dtype=torch.float32))
        nn.init.normal_(self.c, mean=0, std=std)
        self.fc = nn.Linear(2 * moments, embed_dim)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2

        c = self.c.unsqueeze(0)
        x = x.unsqueeze(2)
        angles = x * c

        out = torch.cat([torch.sin(angles), torch.cos(angles)], dim=2)
        out = self.fc(out)
        return out


class AvailabilityEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()

        offset_ = 2 * torch.arange(0, input_dim, dtype=torch.long)
        self.embed = nn.Embedding(2 * input_dim, embed_dim)

        self.register_buffer("offset_", offset_)

    def forward(self, mask: torch.Tensor):
        assert mask.dim() == 2

        mask = mask.to(torch.long)
        out = mask + self.offset_.unsqueeze(0)
        out = self.embed(out)
        return out


class NumericEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()

        self.activation = SinusoidalEmbedding(input_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        out = self.activation(x)
        return out


class CategoricalEmbedding(nn.Module):
    def __init__(self, classes_count: torch.Tensor, embed_dim: int):
        super().__init__()

        assert torch.all(classes_count > 0)

        offset_ = torch.zeros(classes_count.size(0), dtype=torch.long)
        offset_[1:] = torch.cumsum(classes_count, dim=0, dtype=torch.long)[:-1]

        self.embed = nn.Embedding(torch.sum(classes_count).item(), embed_dim)

        self.register_buffer("offset_", offset_)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2
        # 0 <= x[:, i] <= classes_count[i] - 1
        out = x.to(torch.long) + self.offset_.unsqueeze(0)
        out = self.embed(out)
        return out


class TabularEmbedder(nn.Module):
    def __init__(
        self,
        n_columns: int = None,
        num_features: torch.Tensor = None,
        cat_features: torch.Tensor = None,
        classes_count: torch.Tensor = None,
        hidden_embed_dim: int = 512,
        output_embed_dim: int = 256,
    ):
        super().__init__()

        if cat_features is None and num_features is None:
            num_features = torch.arange(n_columns, dtype=torch.long)
            cat_features = torch.tensor([], dtype=torch.long)
        elif cat_features is not None and num_features is None:
            all = torch.arange(n_columns, dtype=torch.long)
            mask = ~torch.isin(all, cat_features)
            num_features = all[mask]
        elif num_features is not None and cat_features is None:
            all = torch.arange(n_columns, dtype=torch.long)
            mask = ~torch.isin(all, num_features)
            cat_features = all[mask]

        s1 = set(num_features.tolist())
        s2 = set(cat_features.tolist())
        assert len(set.intersection(s1, s2)) == 0
        if n_columns is not None:
            assert set.union(s1, s2) == set(range(n_columns))

        if classes_count is None:
            classes_count = torch.tensor([], dtype=torch.long)
        else:
            assert classes_count.shape[0] == cat_features.shape[0]

        self.num_features = num_features
        self.cat_features = cat_features
        self.classes_count = classes_count

        self.availability_embed = AvailabilityEmbedding(
            self.n_columns, hidden_embed_dim
        )
        self.numeric_embed = NumericEmbedding(num_features.shape[0], hidden_embed_dim)
        self.categorical_embed = CategoricalEmbedding(classes_count, hidden_embed_dim)

        self.fc = nn.Linear(2 * hidden_embed_dim, output_embed_dim)

    def forward(self, table: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert table.dim() == 2
        assert mask.dim() == 2
        assert table.shape == mask.shape
        assert table.shape[1] == self.n_columns

        num_columns = table[:, self.num_features]
        cat_columns = table[:, self.cat_features]

        num_embeds = self.numeric_embed(num_columns)
        cat_embeds = self.categorical_embed(cat_columns)

        availability_embeds = self.availability_embed(mask)
        num_availability = availability_embeds[:, self.num_features]
        cat_availability = availability_embeds[:, self.cat_features]

        num_availability_embeds = torch.cat([num_embeds, num_availability], dim=2)
        cat_availability_embeds = torch.cat([cat_embeds, cat_availability], dim=2)
        out = torch.cat([num_availability_embeds, cat_availability_embeds], dim=1)

        out = self.fc(out)
        return out

    @property
    def n_columns(self):
        return self.num_features.size(0) + self.cat_features.size(0)

    def get_classes_count(self, col: int = None):
        if col is None:
            return self.classes_count

        ind = torch.searchsorted(self.cat_features, col).item()
        assert self.cat_features[ind].item() == col

        return self.classes_count[ind].item()


class Bottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 32,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        bottleneck_dims = (
            torch.linspace(input_dim, output_dim, layers)
            .round()
            .to(torch.long)
            .tolist()
        )

        layers = [nn.Linear(bottleneck_dims[0], bottleneck_dims[1])]
        for i in range(1, len(bottleneck_dims) - 1):
            layers += [
                activation(),
                nn.Linear(bottleneck_dims[i], bottleneck_dims[i + 1]),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


class MLPImputer(nn.Module):
    def __init__(
        self,
        n_columns: int = None,
        num_features: torch.Tensor = None,
        cat_features: torch.Tensor = None,
        classes_count: torch.Tensor = None,
        hidden_embed_dim: int = 512,
        output_embed_dim: int = 256,
        bottleneck_output_dim: int = 32,
        bottleneck_layers: int = 32,
        head_layers: int = 16,
    ):
        super().__init__()

        self.tab_embed = TabularEmbedder(
            n_columns,
            num_features,
            cat_features,
            classes_count,
            hidden_embed_dim,
            output_embed_dim,
        )

        self.bottleneck = Bottleneck(
            self.tab_embed.n_columns * output_embed_dim,
            bottleneck_output_dim,
            bottleneck_layers,
        )

        self.heads = nn.ModuleList()
        for col in range(self.tab_embed.n_columns):
            if col in self.tab_embed.num_features:
                model = Bottleneck(
                    bottleneck_output_dim, 2, head_layers
                )  # mu, log_sigma
            else:
                model = Bottleneck(
                    bottleneck_output_dim,
                    self.tab_embed.get_classes_count(col),
                    head_layers,
                )  # logit1, ..., logitK
            self.heads.append(model)

        exits_count_ = torch.tensor(
            [
                self.get_classes_count(col) if col in self.cat_features else 2
                for col in range(self.n_columns)
            ],
            dtype=torch.long,
        )
        summary_exit_count_ = torch.sum(exits_count_)
        cumsum = torch.cumsum(exits_count_, dim=0)
        starts = torch.cat([torch.zeros(1, dtype=torch.long), cumsum[:-1]])
        ranges_ = torch.stack([starts, cumsum], dim=1)

        self.register_buffer("exits_count_", exits_count_)
        self.register_buffer("summary_exit_count_", summary_exit_count_)
        self.register_buffer("ranges_", ranges_)

    @property
    def cat_features(self):
        return self.tab_embed.cat_features

    @property
    def num_features(self):
        return self.tab_embed.num_features

    def get_classes_count(self, col: int = None):
        return self.tab_embed.get_classes_count(col)

    @property
    def n_columns(self):
        return self.tab_embed.n_columns

    @classmethod
    def _basic_check(cls, x: torch.Tensor, mask: torch.Tensor):
        assert x.dim() == 2
        assert mask.dim() == 2
        assert x.shape == mask.shape
        assert torch.all(~x.isnan())
        assert torch.all(~mask.isnan())
        assert mask.dtype == torch.float or mask.dtype == torch.bool
        assert torch.all(torch.logical_or(mask == 0, mask == 1))

    @classmethod
    def get_dropped_mask(
        cls, x: torch.Tensor, mask: torch.Tensor, drop_rate: float = 0.5
    ):
        cls._basic_check(x, mask)

        random_mask = (torch.rand_like(mask, dtype=torch.float) > drop_rate).to(
            torch.float32
        )
        new_mask = mask * random_mask
        new_x = x * new_mask
        dropped_mask = mask * (1 - new_mask)

        return new_x, new_mask, dropped_mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        output = self.tab_embed(x, mask)
        output = output.reshape(output.size(0), -1)
        output = self.bottleneck(output)

        out_list = []
        for col, head in enumerate(self.heads):
            # column is already known
            if not self.training and torch.all(mask[:, col].to(torch.bool)):
                theta = torch.zeros(
                    output.size(0),
                    self.exits_count_[col],
                    device=output.device,
                    dtype=output.dtype,
                )
            else:
                theta = head(output)

            out_list.append(theta)

        output = torch.cat(out_list, dim=1)
        return output

    def loss(
        self,
        output: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
        label_smoothing: float = 0,
    ):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns
        assert output.size(1) == self.summary_exit_count_

        device = next(self.parameters()).device

        num_loss = torch.tensor(0, dtype=torch.float32, device=device)
        if len(self.num_features) > 0:
            num_mask = mask[:, self.num_features].to(torch.bool)
            if torch.sum(num_mask) > 0:
                mu_ind = self.ranges_[self.num_features, 0]
                log_sigma_ind = self.ranges_[self.num_features, 0] + 1

                mu = output[:, mu_ind]
                var = torch.exp(2 * output[:, log_sigma_ind])
                target = x[:, self.num_features]

                nll = nn.functional.gaussian_nll_loss(
                    mu, target, var, eps=eps, reduction="none"
                )
                num_loss += torch.sum(nll * num_mask) / torch.sum(num_mask)

        cat_loss = torch.tensor(0, dtype=torch.float32, device=device)
        cat_elems = torch.tensor(0, dtype=torch.float32, device=device)
        for col in self.cat_features:
            col_mask = mask[:, col].to(torch.bool)
            if torch.sum(col_mask) == 0:
                continue

            start, end = self.ranges_[col]
            logits = output[:, start:end][col_mask].to(torch.float)
            target = x[:, col][col_mask].to(torch.long)

            cat_loss += nn.functional.cross_entropy(
                logits, target, label_smoothing=label_smoothing, reduction="sum"
            )
            cat_elems += target.size(0)

        if cat_elems > 0:
            cat_loss /= cat_elems

        return num_loss + cat_loss

    @torch.inference_mode()
    def gibbs_sample(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        iterations: int = 10,
        eps: float = 1e-6,
        temp: float = 1.0,
    ):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        self.eval()
        device = next(self.parameters()).device

        x_t = x * mask
        if torch.all(mask):
            return x

        for step in range(iterations):

            for col in range(self.n_columns):

                col_miss = ~mask[:, col].to(torch.bool)

                if not torch.any(col_miss):
                    continue

                if step == 0:
                    mask_t = mask.clone()
                else:
                    mask_t = torch.ones_like(x_t, dtype=torch.float32)

                model_mask = mask_t[col_miss, :].clone()
                model_mask[:, col] = 0

                model_x = x_t[col_miss, :].clone()
                model_x[:, col] = 0

                model_x = model_x.to(device)
                model_mask = model_mask.to(device)

                output = self.forward(model_x, model_mask)

                start, end = self.ranges_[col]

                if col in self.num_features:
                    mu = output[:, start].to(torch.float32)
                    log_sigma = output[:, start + 1].to(torch.float32)
                    sigma = torch.exp(log_sigma) + eps
                    generated = Normal(mu, sigma * temp).sample()
                else:
                    logits = output[:, start:end].to(torch.float32)
                    generated = Categorical(logits=logits / temp).sample()

                x_t[col_miss, col] = generated

        return x_t

    @torch.inference_mode()
    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        iterations: int = 10,
        eps: float = 1e-6,
        temp: float = 1.0,
    ):
        return self.gibbs_sample(x, mask, iterations, eps, temp)
