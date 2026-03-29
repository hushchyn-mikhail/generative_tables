from catboost import CatBoostClassifier, CatBoostRegressor

from sklearn.base import BaseEstimator

import numpy as np
import pandas as pd

import mlflow

from utils import TabularEncoder


class CatBoostImputer(BaseEstimator):
    def __init__(
        self,
        *,
        n_columns: int = None,
        num_features: np.ndarray = None,
        cat_features: np.ndarray = None,
        columns_names: np.ndarray = None,
        random_state: int = 0,
        iterations: int = 300,
        learning_rate: float = 0.01,
        depth: int = 1,
        task_type: str = None,
        verbose: bool = False,
    ):

        self.n_columns = n_columns
        self.num_features = num_features
        self.cat_features = cat_features
        self.columns_names = columns_names

        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.task_type = task_type
        self.verbose = verbose

    def get_shifted_categories(self, target_column: int):
        assert self.encoder.is_fitted_
        assert 0 <= target_column < self.encoder.n_columns

        shifted = []
        for col in self.encoder.cat_features:
            if col == target_column:
                continue
            if col > target_column:
                shifted.append(col - 1)
            else:
                shifted.append(col)

        return np.array(shifted, dtype=np.int64)

    def build_model(self, col: int):
        assert self.encoder.is_fitted_

        if col in self.encoder.num_features:
            return CatBoostRegressor(
                cat_features=self.get_shifted_categories(col),
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                loss_function="RMSE",
                random_seed=self.random_state,
                task_type=self.task_type,
                verbose=self.verbose,
                allow_writing_files=False,
            )

        return CatBoostClassifier(
            cat_features=self.get_shifted_categories(col),
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            loss_function="MultiClass",
            classes_count=int(self.encoder.get_classes_count(col, with_unseen=False)),
            random_seed=self.random_state,
            task_type=self.task_type,
            verbose=self.verbose,
            allow_writing_files=False,
        )

    def prepare_input(self, X: np.ndarray, target_column: int, is_train: bool = True):
        assert self.encoder.is_fitted_
        assert X.ndim == 2
        assert X.shape[1] == self.encoder.n_columns
        assert 0 <= target_column < self.encoder.n_columns

        X_out = np.empty(shape=(X.shape[0], self.encoder.n_columns - 1), dtype=object)
        y_out = np.empty(shape=(X.shape[0],), dtype=object)

        new_col = 0
        for col in range(self.encoder.n_columns):
            values = X[:, col].copy()

            if col in self.encoder.num_features:
                values = self.encoder.transform(values, col)
                values = values.astype(np.float32)

            else:
                mask = pd.isna(values)
                values = self.encoder.transform(values, col)

                if col != target_column:
                    missing_token_id = self.encoder.get_classes_count(
                        col, with_unseen=True
                    )  # equals K + 1. useless for us, but useful for catboost, bcs it can't transform categorical none
                    values[mask] = missing_token_id
                    values = values.astype(np.int64)

            if is_train:
                if col == target_column:
                    y_out = values
                else:
                    X_out[:, new_col] = values
                    new_col += 1

            else:
                if col != target_column:
                    X_out[:, new_col] = values
                    new_col += 1

                # else: skip column

        if not is_train:
            return X_out

        mask = ~pd.isna(y_out)
        X_out = X_out[mask, :]
        y_out = y_out[mask]

        if target_column in self.encoder.cat_features:
            mask = y_out != 0
            X_out = X_out[mask, :]
            y_out = (
                y_out[mask].astype(np.int64) - 1
            )  # 1...K -> 0...K-1 without UNKNOWN token

        return X_out, y_out

    def fit(self, X_train: np.ndarray, X_valid: np.ndarray = None, verbose=None):
        self.models_ = {}
        self.is_fitted_ = False

        self.encoder = TabularEncoder(
            self.n_columns, self.num_features, self.cat_features, self.columns_names
        )

        self.encoder.fit(X_train)

        for target_column in range(self.encoder.n_columns):
            model = self.build_model(target_column)

            X_tr, y_tr = self.prepare_input(X_train, target_column, is_train=True)

            if len(y_tr) == 0:
                continue

            eval_set = None
            use_best_model = False

            if X_valid is not None:
                X_val, y_val = self.prepare_input(X_valid, target_column, is_train=True)
                if len(y_val) > 0:
                    eval_set = (X_val, y_val)
                    use_best_model = True

            model.fit(
                X_tr,
                y_tr,
                use_best_model=use_best_model,
                eval_set=eval_set,
                verbose=verbose,
            )

            if verbose is not None:
                column_name = (
                    self.encoder.columns_names[target_column]
                    if self.encoder.columns_names is not None
                    else f"{target_column}"
                )
                log_catboost_column_losses(model, column_name)

            self.models_[target_column] = model

        self.is_fitted_ = True

        if verbose is not None:
            print("Training is done!")

        return self

    def predict(self, X: np.ndarray, to_pandas: bool = False, task_type: str = "CPU"):
        assert self.is_fitted_
        assert X.ndim == 2
        assert X.shape[1] == self.encoder.n_columns

        out = np.array(X, dtype=object)

        for target_column in range(self.encoder.n_columns):
            if target_column not in self.models_:
                continue

            model = self.models_[target_column]

            mask = pd.isna(X[:, target_column])
            if mask.sum() == 0:
                continue

            X_input = X[mask, :]
            X_input = self.prepare_input(X_input, target_column, is_train=False)

            y = model.predict(X_input, task_type=task_type)

            if target_column in self.encoder.num_features:
                y = np.array(y, dtype=np.float32).reshape(-1)
                y = self.encoder.inverse_transform(y, target_column)
            else:
                y = np.array(y, dtype=np.int64).reshape(-1) + 1  # 0...K-1 -> 1...K
                y = self.encoder.inverse_transform(y, target_column)

            out[mask, target_column] = y

        if to_pandas:
            columns = self.encoder.columns_names
            if columns is None:
                columns = [f"column_{i}" for i in range(self.encoder.n_columns)]
            return pd.DataFrame(out, columns=columns)

        return out


def log_catboost_column_losses(
    model: CatBoostRegressor | CatBoostClassifier, column_name: str
):
    evals = model.get_evals_result()

    if "learn" in evals:
        learn_metrics = evals["learn"]
        for metric_name, values in learn_metrics.items():
            for step, value in enumerate(values):
                mlflow.log_metric(
                    f"{column_name}/train_{metric_name}",
                    float(value),
                    step=step,
                )

    for valid_key in ("validation", "validation_0", "test"):
        if valid_key in evals:
            valid_metrics = evals[valid_key]
            for metric_name, values in valid_metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(
                        f"{column_name}/valid_{metric_name}",
                        float(value),
                        step=step,
                    )
            break
