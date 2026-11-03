#!/usr/bin/env python3
import copy
import random
from typing import Literal, Sequence
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ActivationName = Literal["relu", "tanh", "gelu"]
OptimizerName = Literal["sgd", "adam", "adamw"]
TaskName = Literal["classification", "regression"]


def _validate_hidden_dims(values: Sequence[int], name: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    dims = tuple(values)
    if not dims and not allow_empty:
        raise ValueError(f"{name} must contain at least one hidden layer.")
    if any(not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d <= 0 for d in dims):
        raise ValueError(f"Every dimension in {name} must be a positive integer.")
    return tuple(int(d) for d in dims)


class SplitNN:
    """Sklearn-like SplitNN estimator for classification or regression.

    Feature columns are assigned to clients as contiguous, deterministic blocks.
    When D is not divisible by P, the final client receives the remainder, matching
    FedTree's vertical homogeneous partitioning convention.
    """

    def __init__(
        self,
        *,
        task: TaskName = "classification",
        n_parties: int = 2,
        hidden_dims_clients: Sequence[int] = (32, 64),
        hidden_dims_server: Sequence[int] = (64,),
        activation: ActivationName = "relu",
        learning_rate: float = 0.05,
        epochs: int = 25,
        batch_size: int = 64,
        dropout: float = 0.0,
        optimizer: OptimizerName = "sgd",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        shuffle: bool = True,
        random_state: int = 42,
        device: str = "cpu",
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 0.0,
        validation_fraction: float = 0.1,
        restore_best_weights: bool = True,
        verbose: bool = True,
        snapshot_frequency: int = 0,
    ) -> None:
        self.task = task
        self.n_parties = n_parties
        self.hidden_dims_clients = hidden_dims_clients
        self.hidden_dims_server = hidden_dims_server
        self.activation = activation
        self.dropout = dropout
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.device = device
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        self.snapshot_frequency = snapshot_frequency

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2-D array; got shape {X.shape}.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; clean/impute them first.")
        return np.ascontiguousarray(X)

    def _activation_layer(self) -> nn.Module:
        choices = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        return choices[self.activation]()

    def _make_mlp(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int | None = None) -> nn.Sequential:
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(current, width), self._activation_layer()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            current = width
        if output_dim is not None:
            layers.append(nn.Linear(current, output_dim))
        return nn.Sequential(*layers)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        params = list(self.client_models_.parameters()) + list(self.server_model_.parameters())
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, momentum=self.momentum, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        representations = [model(X[:, sl]) for model, sl in zip(self.client_models_, self.feature_slices_)]
        return self.server_model_(torch.cat(representations, dim=1))

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task == "classification":
            return nn.functional.cross_entropy(prediction, target.long())
        return nn.functional.mse_loss(prediction.squeeze(-1), target.float().reshape(-1))

    def _evaluate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        self.client_models_.eval()
        self.server_model_.eval()
        with torch.no_grad():
            loss = self._loss(self._forward(X), y)
        return float(loss.item())

    def fit(self, X: np.ndarray, y: np.ndarray, *, validation_data: tuple[np.ndarray, np.ndarray] | None = None) -> "SplitNN":
        X = self._validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values.")
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        if not isinstance(self.n_parties, int) or isinstance(self.n_parties, bool) or not 1 <= self.n_parties <= X.shape[1]:
            raise ValueError("n_parties must be an integer between 1 and X.shape[1].")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive integers.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.momentum < 0:
            raise ValueError("learning_rate must be > 0; weight_decay and momentum must be >= 0.")
        if self.optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'.")
        if self.activation not in ("relu", "tanh", "gelu"):
            raise ValueError("activation must be 'relu', 'tanh', or 'gelu'.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.early_stopping and self.patience < 1:
            raise ValueError("patience must be >= 1 when early_stopping is enabled.")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1).")
        client_dims = _validate_hidden_dims(self.hidden_dims_clients, "hidden_dims_clients")
        server_dims = _validate_hidden_dims(self.hidden_dims_server, "hidden_dims_server", allow_empty=True)
        if not np.isfinite(y.astype(np.float64)).all():
            raise ValueError("y contains NaN or infinite values.")

        self.n_features_in_ = X.shape[1]
        stride = X.shape[1] // self.n_parties
        starts = [i * stride for i in range(self.n_parties - 1)] + [(self.n_parties - 1) * stride]
        stops = [(i + 1) * stride for i in range(self.n_parties - 1)] + [X.shape[1]]
        self.feature_slices_ = [slice(a, b) for a, b in zip(starts, stops)]
        self.classes_ = None
        if self.task == "classification":
            self.classes_, y_numeric = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("Classification requires at least two distinct classes.")
            y_numeric = y_numeric.astype(np.int64)
        else:
            y_numeric = y.astype(np.float32)

        # Deterministic initialisation and shuffled minibatch order.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        self.device_ = torch.device(self.device)
        self.client_models_ = nn.ModuleList([
            self._make_mlp(sl.stop - sl.start, client_dims) for sl in self.feature_slices_
        ]).to(self.device_)
        out_dim = len(self.classes_) if self.task == "classification" else 1  # ty: ignore[invalid-argument-type]
        self.server_model_ = self._make_mlp(self.n_parties * client_dims[-1], server_dims, out_dim).to(self.device_)
        optimizer = self._make_optimizer()

        X_tensor = torch.from_numpy(X)
        y_tensor = torch.as_tensor(y_numeric, dtype=torch.long if self.task == "classification" else torch.float32)
        if validation_data is not None:
            X_val_np = self._validate_X(validation_data[0])
            y_val_np = np.asarray(validation_data[1]).reshape(-1)
            if len(X_val_np) != len(y_val_np) or X_val_np.shape[1] != X.shape[1]:
                raise ValueError("validation_data must have matching X/y lengths and the same feature count as X.")
            if self.task == "classification":
                # Validation labels must be represented in the training classes.
                class_to_index = {label: i for i, label in enumerate(self.classes_.tolist())}  # ty: ignore[unresolved-attribute]
                try:
                    y_val_np = np.asarray([class_to_index[v] for v in y_val_np], dtype=np.int64)
                except KeyError as exc:
                    raise ValueError(f"Validation label {exc.args[0]!r} was not present in training labels.") from exc
                y_val_tensor = torch.as_tensor(y_val_np, dtype=torch.long, device=self.device_)
            else:
                y_val_tensor = torch.as_tensor(y_val_np.astype(np.float32), dtype=torch.float32, device=self.device_)
            X_val_tensor = torch.from_numpy(X_val_np).to(self.device_)
            val_X, val_y = X_val_tensor, y_val_tensor
            train_indices = np.arange(len(X))
        elif self.early_stopping and self.validation_fraction > 0:
            # Internal holdout is only for early stopping; it is not used for gradient updates.
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(len(X))
            n_val = max(1, int(round(len(X) * self.validation_fraction)))
            if n_val >= len(X):
                raise ValueError("Need at least two training samples for an internal validation split.")
            val_idx, train_indices = indices[:n_val], indices[n_val:]
            val_X = X_tensor[val_idx].to(self.device_)
            val_y = y_tensor[val_idx].to(self.device_)
        else:
            val_X = val_y = None
            train_indices = np.arange(len(X))

        dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False, generator=generator)
        self.loss_history_: list[float] = []
        self.validation_loss_history_: list[float] = []
        batch_records: list[dict] = []
        epoch_records: list[dict] = []
        self.weight_snapshots_: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        fit_start = time.perf_counter()
        cumulative_samples_seen = 0
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.n_iter_ = 0
        snapshot_batch_iters = {round((len(loader) / self.snapshot_frequency) * i) for i in range(1, self.snapshot_frequency + 1)}

        for epoch in range(self.epochs):
            self.client_models_.train()
            self.server_model_.train()
            loss_sum = 0.0
            seen = 0
            epoch_start = time.perf_counter()
            for batch_index, (batch_X, batch_y) in enumerate(loader, start=1):
                step_start = time.perf_counter()
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                prediction = self._forward(batch_X)
                loss = self._loss(prediction, batch_y)
                loss.backward()
                optimizer.step()
                batch_n = len(batch_X)
                loss_sum += float(loss.item()) * batch_n
                seen += batch_n
                self.n_iter_ += 1
                cumulative_samples_seen += batch_n
                step_elapsed = time.perf_counter() - step_start
                # Logical payload estimate for plain SplitNN: client activations
                # sent to the server and their gradients returned, float32 values.
                # This is not measured network traffic; protocol overhead is omitted.
                rep_values = batch_n * self.n_parties * client_dims[-1]
                forward_bytes = rep_values * np.dtype(np.float32).itemsize
                backward_bytes = forward_bytes
                batch_records.append({
                    "round": self.n_iter_,
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "batch_size": batch_n,
                    "loss": float(loss.item()),
                    "samples_seen_cumulative": cumulative_samples_seen,
                    "step_time_s": step_elapsed,
                    "elapsed_time_s": time.perf_counter() - fit_start,
                    "estimated_forward_bytes": int(forward_bytes),
                    "estimated_backward_bytes": int(backward_bytes),
                    "estimated_communication_bytes": int(forward_bytes + backward_bytes),
                })
                if batch_index in snapshot_batch_iters: self.save_snapshot(self.n_iter_)
            epoch_loss = loss_sum / max(seen, 1)
            self.loss_history_.append(epoch_loss)
            val_loss = self._evaluate_loss(val_X, val_y) if val_X is not None else None  # ty: ignore[invalid-argument-type]
            if val_loss is not None:
                self.validation_loss_history_.append(val_loss)
            if self.verbose:
                suffix = f" - val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch + 1:03d}/{self.epochs} - loss: {epoch_loss:.4f}{suffix}")
            epoch_elapsed = time.perf_counter() - epoch_start
            client_norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.client_models_.parameters())).item())  # ty: ignore[invalid-argument-type]
            server_norm = float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.server_model_.parameters())).item())  # ty: ignore[invalid-argument-type]
            epoch_records.append({
                "epoch": epoch + 1,
                "train_loss": epoch_loss,
                "validation_loss": val_loss,
                "epoch_time_s": epoch_elapsed,
                "elapsed_time_s": time.perf_counter() - fit_start,
                "optimizer_steps": self.n_iter_,
                "samples_seen_cumulative": cumulative_samples_seen,
                "client_weights_l2_norm": client_norm,
                "server_weights_l2_norm": server_norm,
                "total_weights_l2_norm": float(np.hypot(client_norm, server_norm)),
            })
            if self.early_stopping and val_loss is not None:
                if val_loss < best_loss - self.min_delta:
                    best_loss = val_loss
                    best_state = {
                        "clients": copy.deepcopy(self.client_models_.state_dict()),
                        "server": copy.deepcopy(self.server_model_.state_dict()),
                    }
                    self.best_epoch_ = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}; best epoch was {self.best_epoch_:d}.")
                        break

        self.n_epochs_ = len(self.loss_history_)
        self.batch_history_ = pd.DataFrame.from_records(batch_records)
        self.epoch_history_ = pd.DataFrame.from_records(epoch_records)

        self.fit_time_s_ = time.perf_counter() - fit_start
        if self.early_stopping and self.restore_best_weights and best_state is not None:
            self.client_models_.load_state_dict(best_state["clients"])
            self.server_model_.load_state_dict(best_state["server"])
        return self

    def save_snapshot(self, round):
        self.weight_snapshots_[round] = {
            "clients": [{k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
                        for model in self.client_models_],
            "server":   {k: v.detach().cpu().clone()
                            for k, v in self.server_model_.state_dict().items()}
            }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "server_model_"):
            raise RuntimeError("Call fit before predict.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        self.client_models_.eval()
        self.server_model_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = []
            for start in range(0, len(tensor), self.batch_size):
                outputs.append(self._forward(tensor[start:start + self.batch_size]).cpu())
            predictions = torch.cat(outputs, dim=0).numpy()
        if self.task == "classification":
            return self.classes_[predictions.argmax(axis=1)]  # ty: ignore[not-subscriptable]
        return predictions.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification.")
        if not hasattr(self, "server_model_"):
            raise RuntimeError("Call fit before predict_proba.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")
        self.client_models_.eval()
        self.server_model_.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = [self._forward(tensor[i:i + self.batch_size]).cpu() for i in range(0, len(tensor), self.batch_size)]
            logits = torch.cat(outputs, dim=0)
            return torch.softmax(logits, dim=1).numpy()

    def _forward_with_snapshot(self, X: torch.Tensor, snapshot: dict,) -> torch.Tensor:
 
        representations = []
        for model, state, sl in zip(self.client_models_, snapshot["clients"], self.feature_slices_,):
            state = {k: v.to(X.device) for k, v in state.items()}
            representations.append(torch.func.functional_call(model, state, (X[:, sl],),))

        server_state = {k: v.to(X.device) for k, v in snapshot["server"].items()}
        return torch.func.functional_call(self.server_model_, server_state, (torch.cat(representations, dim=1),),)

    def predict_proba_at(self, X: np.ndarray, round: int,) -> np.ndarray:

        if self.task != "classification":
            raise AttributeError("predict_proba_at is available only for classification.")

        if round not in self.weight_snapshots_:
            raise ValueError(f"No weight snapshot available for round {round}.")

        X = self._validate_X(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        snapshot = self.weight_snapshots_[round]
        self.client_models_.eval()
        self.server_model_.eval()

        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = [self._forward_with_snapshot(tensor[i:i + self.batch_size], snapshot).cpu()
                       for i in range(0, len(tensor), self.batch_size)]
            logits = torch.cat(outputs, dim=0)
            return torch.softmax(logits, dim=1).numpy()


class SplitNNClassifier(SplitNN):
    """Convenience wrapper for classification."""
    def __init__(self, **kwargs):
        super().__init__(task="classification", **kwargs)


class SplitNNRegressor(SplitNN):
    """Convenience wrapper for single-target regression."""
    def __init__(self, **kwargs):
        super().__init__(task="regression", **kwargs)
