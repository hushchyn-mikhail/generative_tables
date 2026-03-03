from typing import List

import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from tqdm.notebook import tqdm


class LinearRegression(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()

        self.num_features = num_features

        self.fc = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor):
        return self.fc(x).squeeze(1)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        return self.forward(x)


class LogisticRegression(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.fc(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor):
        self.eval()
        return torch.softmax(self.forward(x), dim=1)


class TabularDataset(Dataset):
    def __init__(
        self,
        table: np.ndarray,
        columns_names: List[str | int],
        num_features: List[int],
        cat_features: List[int],
    ):

        super().__init__()

        assert table.shape[1] == len(columns_names)
        assert len(set(num_features)) == len(num_features)
        assert len(set(cat_features)) == len(cat_features)
        assert len(np.union1d(num_features, cat_features)) == len(columns_names)
        assert len(np.intersect1d(num_features, cat_features)) == 0

        self.table = table
        self.num_features = num_features
        self.cat_features = cat_features
        self.columns_names = columns_names

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, index):
        return self.table[index, :]

    @property
    def n_columns(self):
        return self.table.shape[1]

    def is_num_feature(self, target_column: int):
        return target_column in self.num_features

    def is_cat_feature(self, target_column: int):
        return target_column in self.cat_features

    def get_shifted_cat_features(self, target_column: int):
        shifted = []
        for column in self.cat_features:
            if column == target_column:
                continue
            elif column > target_column:
                shifted.append(column - 1)
            else:
                shifted.append(column)
        return shifted

    def get_mask(self, target_column: int):
        mask = np.ones(self.n_columns, dtype=bool)
        mask[target_column] = False
        return mask

    def get_train_test_for_column(
        self, target_column: int, test_size: float, random_state: float
    ):
        y = self.table[:, target_column]

        non_na = ~pd.isna(y)
        X_y, y = self.table[non_na], y[non_na]

        if self.is_num_feature(target_column):
            stratify = None
        else:
            vc = pd.Series(y).value_counts()
            stratify = y if (vc >= 2).all() else None

        X_y_train, X_y_test = train_test_split(
            X_y, test_size=test_size, stratify=stratify, random_state=random_state
        )

        return X_y_train, X_y_test


class TabularModel:
    def __init__(self):
        self.n_columns = None
        self.num_features = None
        self.cat_features = None

        self.models = dict()
        self.metrics = dict()
        self.imputers = dict()
        self.transformers = dict()

    def fit_transformers(
        self,
        dataset: TabularDataset,
        test_size: float,
        random_state: float,
        fill_strategy_num: str = "mean",
        fill_strategy_cat: str = "most_frequent",
    ):

        self.n_columns = dataset.n_columns
        self.num_features = dataset.num_features
        self.cat_features = dataset.cat_features

        for target_column in tqdm(
            range(self.n_columns), desc="Fitting column", leave=False
        ):
            X_y_train, _ = dataset.get_train_test_for_column(
                target_column, test_size, random_state
            )

            imputers = dict()
            transformers = dict()

            for column in tqdm(
                range(self.n_columns), desc="Fitting column within", leave=False
            ):
                if dataset.is_num_feature(column):
                    imputer = SimpleImputer(strategy=fill_strategy_num)
                    transformer = StandardScaler()

                else:
                    imputer = SimpleImputer(strategy=fill_strategy_cat)
                    transformer = OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=np.nan,
                    )

                values = X_y_train[:, [column]]
                mask = ~pd.isna(values[:, 0])
                values = values[mask]

                imputer.fit(values)
                transformer.fit(values)

                imputers[column] = imputer
                transformers[column] = transformer

            self.imputers[target_column] = imputers
            self.transformers[target_column] = transformers

    def transform(
        self,
        table: np.ndarray,
        target_column: int,
        impute_num_features: bool = True,
        impute_cat_features: bool = True,
        transform_num_features: bool = True,
        transform_cat_features: bool = True,
        with_target_column: bool = True,
    ):
        assert table.shape[1] == self.n_columns

        imputers = self.imputers[target_column]
        transformers = self.transformers[target_column]

        X = table.copy()
        for column in range(self.n_columns):
            if (
                column != target_column
                and (
                    self.is_num_feature(column)
                    and impute_num_features
                    or self.is_cat_feature(column)
                    and impute_cat_features
                )
                or column == target_column
                and with_target_column
            ):
                X[:, [column]] = imputers[column].transform(X[:, [column]])

        for column in range(self.n_columns):
            if (
                column != target_column
                and (
                    self.is_num_feature(column)
                    and transform_num_features
                    or self.is_cat_feature(column)
                    and transform_cat_features
                )
                or column == target_column
                and with_target_column
            ):
                X[:, [column]] = transformers[column].transform(X[:, [column]])
                if column in self.cat_features:
                    X[:, [column]] = X[:, [column]].astype(int)

        return X

    def get_categories(self, target_column: int) -> List:
        assert target_column in self.cat_features
        return self.transformers[target_column][target_column].categories_[0].tolist()

    def get_mask(self, target_column: int):
        mask = np.ones(self.n_columns, dtype=bool)
        mask[target_column] = False
        return mask

    def is_num_feature(self, target_column: int):
        return target_column in self.num_features

    def is_cat_feature(self, target_column: int):
        return target_column in self.cat_features

    def get_shifted_cat_features(self, target_column: int):
        shifted = []
        for column in self.cat_features:
            if column == target_column:
                continue
            elif column > target_column:
                shifted.append(column - 1)
            else:
                shifted.append(column)
        return shifted
