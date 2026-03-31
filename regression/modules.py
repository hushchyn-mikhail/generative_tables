import torch
from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features

        self.fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor):
        return self.fc(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        self.eval()
        return self.forward(x).squeeze(1)


class LogisticRegression(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.fc(x)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor):
        self.eval()
        return torch.softmax(self.forward(x), dim=1)


class RegressionImputer(nn.Module):
    def __init__(
        self,
        *,
        n_columns: int = None,
        num_features: torch.Tensor = None,
        cat_features: torch.Tensor = None,
        classes_count: torch.Tensor = None,
    ):
        super().__init__()

        self.n_columns = n_columns
        self.num_features = num_features
        self.cat_features = cat_features
        self.classes_count = classes_count

        heads = [None] * n_columns

        for col in num_features:
            heads[col.item()] = LinearRegression(n_columns - 1)

        for col in cat_features:
            heads[col.item()] = LogisticRegression(
                n_columns - 1, self.get_classes_count(col.item())
            )

        self.heads = nn.ModuleList(heads)

        exits_count_ = torch.tensor(
            [
                self.get_classes_count(col) if col in self.cat_features else 1
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

    def get_classes_count(self, col: int = None):
        if col is None:
            return self.classes_count

        ind = torch.searchsorted(self.cat_features, col).item()
        assert self.cat_features[ind].item() == col

        return self.classes_count[ind].item()

    def prepare_input(self, x: torch.Tensor, mask: torch.Tensor):
        out = x.clone()

        # if cat. feature is 0...K, then None must be encoded as K + 1
        for col in self.cat_features:
            col_mask = mask[:, col] == 0
            if torch.sum(col_mask) == 0:
                continue

            none_token = self.get_classes_count(col)
            out[col_mask, col] = none_token

        return out

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        mask = mask.to(torch.bool)
        x_prepared = self.prepare_input(x, mask)

        out_list = []
        for col, head in enumerate(self.heads):
            take = torch.ones(size=(self.n_columns,), dtype=torch.bool, device=x.device)
            take[col] = False
            x_input = x_prepared[:, take]
            theta = head(x_input)

            out_list.append(theta)

        output = torch.cat(out_list, dim=1)
        return output

    def loss(
        self,
        output: torch.Tensor,
        target_input: torch.Tensor,
        target_mask: torch.Tensor,
        label_smoothing: float = 0.0,
    ):
        self._basic_check(target_input, target_mask)
        assert target_input.size(1) == self.n_columns
        assert output.dim() == 2
        assert output.size(1) == self.summary_exit_count_

        device = next(self.parameters()).device

        num_loss = torch.tensor(0, dtype=torch.float32, device=device)
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
                reduction="mean",
            )
            num_loss += mse

        cat_loss = torch.tensor(0, dtype=torch.float32, device=device)
        for col in self.cat_features:
            col_mask = target_mask[:, col].to(torch.bool)
            if torch.sum(col_mask) == 0:
                continue

            start, end = self.ranges_[col]
            logits = output[:, start:end][col_mask].to(torch.float)
            target = target_input[:, col][col_mask].to(torch.long)

            cat_loss += nn.functional.cross_entropy(
                logits, target, label_smoothing=label_smoothing, reduction="mean"
            )

        return num_loss + cat_loss

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, mask: torch.Tensor):
        self._basic_check(x, mask)
        assert x.size(1) == self.n_columns

        self.eval()

        mask = mask.to(torch.bool)
        x_prepared = self.prepare_input(x, mask)

        best_inserts = torch.zeros_like(x)

        for col, head in enumerate(self.heads):
            col_mask = ~mask[:, col]
            if torch.sum(col_mask) == 0:
                continue

            take = torch.ones(self.n_columns, dtype=torch.bool, device=x.device)
            take[col] = False
            x_input = x_prepared[col_mask][:, take]
            pred = head.predict(x_input).to(best_inserts.dtype)

            best_inserts[col_mask, col] = pred

        output = x.clone()
        output[~mask] = best_inserts[~mask]
        return output
