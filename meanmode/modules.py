from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd


class MeanModeImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        *,
        n_columns: int = None,
        num_features: np.ndarray = None,
        cat_features: np.ndarray = None,
        columns_names: np.ndarray = None,
    ):

        self.n_columns = n_columns
        self.num_features = num_features
        self.cat_features = cat_features
        self.columns_names = columns_names

    def fit(self, X: np.ndarray, y=None):
        assert X.ndim == 2
        assert X.shape[1] == self.n_columns

        self.num_values_ = {}
        self.cat_values_ = {}
        self.is_fitted_ = False

        for col in self.num_features:
            mean_value = 0.0

            values = X[:, col]
            values = values[~pd.isna(values)]
            values = values.astype(np.float32)

            if len(values) > 0:
                mean_value = np.mean(values)

            self.num_values_[col] = mean_value

        for col in self.cat_features:
            values = X[:, col]
            values = values[~pd.isna(values)]
            values = values.astype(object)

            if len(values) == 0:
                mode_value = "__UNKNOWN__"
            else:
                c = Counter(values)
                mode_value = c.most_common(1)[0][0]

            self.cat_values_[col] = mode_value

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray):
        assert self.is_fitted_
        assert X.ndim == 2
        assert X.shape[1] == self.n_columns

        out = X.copy().astype(object)

        for col in self.num_features:
            mask = pd.isna(out[:, col])
            if np.any(mask):
                out[mask, col] = self.num_values_[col]

        for col in self.cat_features:
            mask = pd.isna(out[:, col])
            if np.any(mask):
                out[mask, col] = self.cat_values_[col]

        return out

    def predict(self, X: np.ndarray):
        return self.transform(X)
