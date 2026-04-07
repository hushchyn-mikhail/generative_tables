import torch
from torch import nn


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
        d_model: int = 256,
        embed_dim: int = 256,
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

        self.availability_embed = AvailabilityEmbedding(self.n_columns, embed_dim)
        self.numeric_embed = NumericEmbedding(num_features.shape[0], embed_dim)
        self.categorical_embed = CategoricalEmbedding(classes_count, embed_dim)

        self.fc = nn.Linear(2 * embed_dim, d_model)

    def forward(self, table: torch.Tensor, mask: torch.Tensor):
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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 32,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert layers >= 1

        dims = (
            torch.linspace(input_dim, output_dim, layers + 1)
            .round()
            .to(torch.long)
            .tolist()
        )

        layers = [nn.Linear(dims[0], dims[1])]
        for i in range(1, len(dims) - 1):
            layers += [
                activation(),
                nn.Dropout(dropout),
                nn.Linear(dims[i], dims[i + 1]),
            ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.activation = activation()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)

        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        out = self.norm(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layers: int = 32,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
    ):
        super().__init__()
        assert layers >= 1

        dims = (
            torch.linspace(input_dim, output_dim, layers + 1)
            .round()
            .to(torch.long)
            .tolist()
        )

        layers = []
        for i in range(len(dims) - 1):
            layers.append(
                ResidualBlock(
                    dims[i],
                    dims[i + 1],
                    dropout=dropout,
                    activation=activation,
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = self.layers(x)
        return out


class TabImputer(nn.Module):
    def __init__(
        self,
        *,
        n_columns: int = None,
        num_features: torch.Tensor = None,
        cat_features: torch.Tensor = None,
        classes_count: torch.Tensor = None,
        d_model: int = 32,
        embed_dim: int = 16,
        encoder_output_dim: int = None,
        encoder_layers: int = 16,
        head_layers: int = 16,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
        backbone: nn.Module = MLP
    ):
        super().__init__()

        self.tab_embed = TabularEmbedder(
            n_columns,
            num_features,
            cat_features,
            classes_count,
            d_model,
            embed_dim,
        )

        if encoder_output_dim is None:
            encoder_output_dim = self.tab_embed.n_columns * d_model

        self.encoder = backbone(
            self.tab_embed.n_columns * d_model,
            encoder_output_dim,
            encoder_layers,
            dropout,
            activation,
        )

        self.heads = nn.ModuleList()
        for col in range(self.tab_embed.n_columns):
            if torch.any(self.tab_embed.num_features == col):
                model = MLP(
                    encoder_output_dim, 1, head_layers, dropout, activation
                )  # mu
            else:
                model = MLP(
                    encoder_output_dim,
                    self.tab_embed.get_classes_count(col),
                    head_layers,
                    dropout,
                    activation,
                )  # logit1, ..., logitK
            self.heads.append(model)

        exits_count_ = torch.tensor(
            [
                (
                    self.get_classes_count(col)
                    if torch.any(self.cat_features == col)
                    else 1
                )
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

        self.repr = {
            "d_model": d_model,
            "embed_dim": embed_dim,
            "encoder_output_dim": encoder_output_dim,
            "encoder_layers": encoder_layers,
            "head_layers": head_layers,
            "dropout": dropout,
            "activation": activation.__name__,
            "backbone": backbone.__name__,
        }

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
        assert x.dtype.is_floating_point
        assert torch.all(~x.isnan())

        assert mask.dtype in (torch.bool, torch.float32, torch.float64)
        if mask.dtype.is_floating_point:
            assert torch.all(~mask.isnan())

        assert torch.all(torch.logical_or(mask == 0, mask == 1))

    @classmethod
    def transform(cls, x: torch.Tensor, mask: torch.Tensor, drop_rate: float = 0.5):
        cls._basic_check(x, mask)

        random_mask = (torch.rand_like(mask, dtype=torch.float32) > drop_rate).to(
            torch.float32
        )
        mask_out = mask * random_mask
        x_out = x * mask_out
        dropped = mask * (1 - mask_out)

        return x_out, mask_out, dropped

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        mask = mask.to(torch.bool)
        x = x.clone()
        x = x * mask

        output = self.tab_embed(x, mask)
        output = output.reshape(output.size(0), -1)
        output = self.encoder(output)

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
        target_input: torch.Tensor,
        target_mask: torch.Tensor,
        eps: float = 1e-6,
        label_smoothing: float = 0.0,
    ):
        self._basic_check(target_input, target_mask)
        assert target_input.size(1) == self.n_columns
        assert output.dim() == 2
        assert output.size(1) == self.summary_exit_count_

        device = next(self.parameters()).device

        num_loss = torch.tensor(0, dtype=torch.float32, device=device)
        num_elems = torch.tensor(0, dtype=torch.float32, device=device)
        for col in self.num_features:
            col_mask = target_mask[:, col].to(torch.bool)
            if torch.sum(col_mask) == 0:
                continue

            mu_ind = self.ranges_[col, 0]

            mu = output[:, mu_ind][col_mask]
            target = target_input[:, col][col_mask]

            mse = nn.functional.mse_loss(
                mu,
                target,
                reduction="sum",
            )
            num_loss += mse
            num_elems += target.size(0)

        if num_elems > 0:
            num_loss /= num_elems

        cat_loss = torch.tensor(0, dtype=torch.float32, device=device)
        cat_elems = torch.tensor(0, dtype=torch.float32, device=device)
        for col in self.cat_features:
            col_mask = target_mask[:, col].to(torch.bool)
            if torch.sum(col_mask) == 0:
                continue

            start, end = self.ranges_[col]
            logits = output[:, start:end][col_mask].to(torch.float)
            target = target_input[:, col][col_mask].to(torch.long)

            cat_loss += nn.functional.cross_entropy(
                logits, target, label_smoothing=label_smoothing, reduction="sum"
            )
            cat_elems += target.size(0)

        if cat_elems > 0:
            cat_loss /= cat_elems

        return num_loss + cat_loss

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, mask: torch.Tensor):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        self.eval()

        mask = mask.to(torch.bool)
        x = x.clone()
        x = x * mask

        output = self.forward(x, mask)

        best_inserts = torch.zeros_like(x)

        for col in self.num_features:
            mu_exit = self.ranges_[col, 0]
            best_inserts[:, col] = output[:, mu_exit]

        for col in self.cat_features:
            start, end = self.ranges_[col]
            logits = output[:, start:end]
            best_inserts[:, col] = torch.argmax(logits, dim=1).to(best_inserts.dtype)

        x[~mask] = best_inserts[~mask]
        return x
