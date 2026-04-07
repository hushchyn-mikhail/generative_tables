from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class NumericEncoder(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, y=None):
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

        values = X[~pd.isna(X)]
        values = values.astype(np.float32)

        if len(values) == 0:
            self.mean_ = 0.0
            self.std_ = 1.0
            self.is_fitted_ = True
            return self

        self.mean_ = np.float32(np.mean(values))

        if len(values) <= 1:
            self.std_ = 1.0
        else:
            self.std_ = np.std(values, ddof=1)
            if self.std_ == 0.0 or np.isnan(self.std_):
                self.std_ = 1.0

        self.std_ = np.float32(self.std_)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray):
        assert self.is_fitted_

        out = np.full(X.shape, np.nan, dtype=np.float32)
        mask = ~pd.isna(X)

        values = X[mask].astype(np.float32)
        out[mask] = (values - self.mean_) / self.std_
        return out

    def inverse_transform(self, X: np.ndarray):
        assert self.is_fitted_

        out = np.full(X.shape, np.nan, dtype=np.float32)
        mask = ~pd.isna(X)

        values = X[mask].astype(np.float32)
        out[mask] = values * self.std_ + self.mean_
        return out


class CategoricalEncoder(TransformerMixin, BaseEstimator):
    UNK_ID = 0
    UNK_LABEL = "__UNSEEN__"

    def __init__(self):
        super().__init__()

    def fit(self, X: np.ndarray, y=None):
        self.word_to_token_ = dict()
        self.token_to_word_ = dict()
        self.is_fitted_ = False

        values = X[~pd.isna(X)]

        unique = list(dict.fromkeys(values))
        tokens = np.arange(1, len(unique) + 1)

        self.word_to_token_ = dict(zip(unique, tokens))
        self.token_to_word_ = dict(zip(tokens, unique))

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray):
        assert self.is_fitted_

        out = np.full(X.shape, np.nan, dtype=np.float32)
        mask = ~pd.isna(X)
        out[mask] = np.array(
            [self.word_to_token_.get(key, self.UNK_ID) for key in X[mask]],
            dtype=np.float32,
        )
        return out

    def inverse_transform(self, X: np.ndarray):
        assert self.is_fitted_

        X = X.astype(object)
        out = np.full(X.shape, np.nan, dtype=object)

        mask = ~pd.isna(X)
        values = np.asarray(X[mask], dtype=np.float32)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "CategoricalEncoder.inverse_transform received non-finite values"
            )

        if not np.all(values == np.floor(values)):
            raise ValueError(
                "CategoricalEncoder.inverse_transform expects integer-valued tokens"
            )

        if not np.all(values >= 0):
            raise ValueError(
                "CategoricalEncoder.inverse_transform expects non-negative tokens"
            )

        tokens = values.astype(np.int64)

        for token in tokens:
            if token != self.UNK_ID and token not in self.token_to_word_:
                raise ValueError(
                    f"Unknown categorical token (not seen in fit): {token}"
                )

        put = []
        for token in tokens:
            word = self.token_to_word_.get(token, self.UNK_LABEL)
            put.append(word)

        out[mask] = np.array(put, dtype=object)
        return out

    def get_classes_count(self, with_unseen=False):
        out = len(self.word_to_token_)
        if with_unseen:
            out += 1
        return out


class TabularEncoder(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        n_columns: int,
        num_features: np.ndarray = None,
        cat_features: np.ndarray = None,
        columns_names: list = None,
    ):
        super().__init__()

        if cat_features is None and num_features is None:
            num_features = np.arange(n_columns, dtype=int)
            cat_features = np.array([], dtype=int)

        elif cat_features is not None and num_features is None:
            cat_features = np.asarray(cat_features, dtype=int)
            all_features = np.arange(n_columns, dtype=int)
            mask = ~np.isin(all_features, cat_features)
            num_features = all_features[mask]

        elif num_features is not None and cat_features is None:
            num_features = np.asarray(num_features, dtype=int)
            all_features = np.arange(n_columns, dtype=int)
            mask = ~np.isin(all_features, num_features)
            cat_features = all_features[mask]

        else:
            num_features = np.asarray(num_features, dtype=int)
            cat_features = np.asarray(cat_features, dtype=int)

        s1 = set(num_features.tolist())
        s2 = set(cat_features.tolist())

        assert len(s1.intersection(s2)) == 0
        assert s1.union(s2) == set(range(n_columns))

        if columns_names is not None:
            assert len(columns_names) == n_columns

        self.n_columns = n_columns
        self.num_features = np.sort(num_features)
        self.cat_features = np.sort(cat_features)
        self.columns_names = columns_names

    def get_classes_count(self, col: int = None, with_unseen=True):
        assert self.is_fitted_

        if col is None:
            return np.array(
                [
                    self.cat_encoders_[i].get_classes_count(with_unseen)
                    for i in self.cat_features
                ],
                dtype=np.int64,
            )

        assert col in self.cat_features
        return self.cat_encoders_[col].get_classes_count(with_unseen)

    def fit(self, X: np.ndarray, y=None):
        assert X.ndim == 2
        assert X.shape[1] == self.n_columns

        self.num_encoders_ = {}
        self.cat_encoders_ = {}
        self.is_fitted_ = False

        for col in self.num_features:
            encoder = NumericEncoder()
            encoder.fit(X[:, col])
            self.num_encoders_[col] = encoder

        for col in self.cat_features:
            encoder = CategoricalEncoder()
            encoder.fit(X[:, col])
            self.cat_encoders_[col] = encoder

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray, col: int = None):
        assert self.is_fitted_

        if col is not None:
            assert X.ndim == 1
            assert 0 <= col < self.n_columns

            if col in self.num_features:
                return self.num_encoders_[col].transform(X)

            if col in self.cat_features:
                return self.cat_encoders_[col].transform(X)

        assert X.ndim == 2
        assert X.shape[1] == self.n_columns

        out = np.empty(X.shape, dtype=np.float32)

        for col in self.num_features:
            out[:, col] = self.num_encoders_[col].transform(X[:, col])

        for col in self.cat_features:
            out[:, col] = self.cat_encoders_[col].transform(X[:, col])

        return out

    def inverse_transform(self, X: np.ndarray, col: int = None):
        assert self.is_fitted_

        if col is not None:
            assert X.ndim == 1
            assert 0 <= col < self.n_columns

            if col in self.num_features:
                return self.num_encoders_[col].inverse_transform(X)

            if col in self.cat_features:
                return self.cat_encoders_[col].inverse_transform(X)

        assert X.ndim == 2
        assert X.shape[1] == self.n_columns

        out = np.empty((X.shape[0], self.n_columns), dtype=object)

        for col in self.num_features:
            out[:, col] = self.num_encoders_[col].inverse_transform(X[:, col])

        for col in self.cat_features:
            out[:, col] = self.cat_encoders_[col].inverse_transform(X[:, col])

        return out


class TabularDataset(Dataset):
    def __init__(self, table: np.ndarray, encoder: TabularEncoder):
        super().__init__()

        assert table.ndim == 2
        assert encoder.is_fitted_ == True

        self.table = table
        self.encoder = encoder

    def __len__(self):
        return self.table.shape[0]

    def __getitem__(self, idx):
        row_table = self.table[idx : idx + 1, :].copy()
        row_mask = ~pd.isna(row_table)

        row_table = self.encoder.transform(row_table)
        row_table[~row_mask] = 0.0

        row_table = row_table.astype(np.float32)
        row_mask = row_mask.astype(bool)

        row_table = torch.from_numpy(row_table[0])
        row_mask = torch.from_numpy(row_mask[0])

        return row_table, row_mask
