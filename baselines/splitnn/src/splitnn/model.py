#!/usr/bin/env python3
import copy
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from typing import Literal, Sequence, Self


ActivationName = Literal["relu", "tanh", "gelu"]
OptimizerName = Literal["sgd", "adam", "adamw"]
TaskName = Literal["classification", "regression"]


def _validate_hidden_dims(values: Sequence[int], name: str, allow_empty: bool=False) -> tuple[int,...]:
    dims = tuple(values)
    if not dims and not allow_empty:
        raise ValueError(f"{name} must contain at least one hidden layer.")
    if any(not isinstance(d, (int, np.integer)) or isinstance(d, bool) or d <= 0 for d in dims):
        raise ValueError(f"Every dimension in {name} must be a positive integer.")
    return tuple(int(d) for d in dims)


class SplitNN(BaseEstimator):
    """Sklearn-like SplitNN estimator for classification or regression.
    Feature columns are assigned to clients as contiguous blocks.
    When 'm' is not divisible by 'p', the final client receives the remainder (as FedTree does).
    """

    def __init__(
        self,
        # NN
        task:                   TaskName = "classification",
        n_parties:              int = 2,
        hidden_dims_clients:    Sequence[int]  = (32, 64),
        hidden_dims_server:     Sequence[int]  = (64,),
        activation:             ActivationName = "relu",
        # FIT
        epochs:                 int = 25,
        batch_size:             int = 64,
        dropout:                float = 0.0,
        # OPTIM
        optimizer:              OptimizerName = "sgd",
        learning_rate:          float = 0.05,
        momentum:               float = 0.0,
        weight_decay:           float = 0.0,
        # TRAIN STOP
        early_stopping:         bool  = False,
        patience:               int   = 10,
        min_delta:              float = 0.0,
        validation_fraction:    float = 0.1,
        # MISC
        shuffle:                bool = True,
        random_state:           int  = 42,
        device:                 str  = "cpu",
        verbose:                bool = True,
        snapshot_frequency:     int  = 0
    ) -> None:
        for key, value in locals().items():
            if key != 'self':   setattr(self, key, value)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        # Checks X data shape and values.
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got ({X.shape}).")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values, clean/impute them first.")
        return np.ascontiguousarray(X)

    def _activation_layer(self) -> nn.Module:
        # Returns the chosen activation function as a callable
        choices = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        return choices[self.activation]()

    def _make_mlp(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int|None=None) -> nn.Sequential:
        # Creates MLP-NN with of dimensions: input_dim → hidden_dims → output_dim
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(current, width), self._activation_layer()])
            if self.dropout > 0:    # dropout between each hidden layer
                layers.append(nn.Dropout(self.dropout))
            current = width
        if output_dim is not None:
            layers.append(nn.Linear(current, output_dim))
        return nn.Sequential(*layers)

    def _make_optimizer(self) -> torch.optim.Optimizer:
        # Creates and configures an optimizer: (SGD → lr, weight_decay, momentum), (ADAM|ADAMW → lr, weight_decay)
        params = list(self.client_models_.parameters()) + list(self.server_model_.parameters()) # MLP torch tensors
        if self.optimizer == "sgd":
            return torch.optim.SGD(params, momentum=self.momentum, lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        # Forward pass of each client (w_i @ X_i) → concat representations → server forward pass (w_p @ rep_lst)
        representations = [model(X[:, sl]) for model, sl in zip(self.client_models_, self.feature_slices_)]
        return self.server_model_(torch.cat(representations, dim=1))

    def _loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Evaluate loss function (MSE obj.fn for regression, CE for binary classif)
        if self.task == "classification":
            return nn.functional.cross_entropy(prediction, target.long())
        return nn.functional.mse_loss(prediction.squeeze(-1), target.float().reshape(-1))

    def _evaluate_loss(self, X: torch.Tensor, y: torch.Tensor) -> float:
        # Wrapper for loss fn evaluation (forward pass of 'X' and compares to 'y')
        self.client_models_.eval()
        self.server_model_.eval()
        with torch.no_grad():
            loss = self._loss(self._forward(X), y)
        return float(loss.item())

    def _check_X_y(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        # Collection of checks and validations before fitting
        X = self._validate_X(X)
        y = np.asarray(y).reshape(-1)
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} values.")
        if self.task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'.")
        if not 1 <= self.n_parties <= X.shape[1]:
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
        if self.early_stopping and not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1) when early_stopping is enabled.")
        if not np.isfinite(y.astype(np.float64)).all():
            raise ValueError("y contains NaN or infinite values.")
        return X, y

    def _save_snapshot(self, iter: int) -> None:
        # Aux. function to save weights of the splitNN-VFL MLP into weight_snapshots_ dict.
        self.weight_snapshots_[iter] = {
            "clients": [{k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
                        for model in self.client_models_],
            "server":   {k: v.detach().cpu().clone()
                            for k, v in self.server_model_.state_dict().items()}
            }

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # Fit VFL-splitNN for the given (X, y) dataset
        
        # [Check and validate inputs]
        X, y = self._check_X_y(X, y)
        client_dims = _validate_hidden_dims(self.hidden_dims_clients, "hidden_dims_clients")
        server_dims = _validate_hidden_dims(self.hidden_dims_server, "hidden_dims_server", allow_empty=True)

        # [VFL: m_i features, desired output encoding]
        self.n_features_in_ = X.shape[1]
        stride = X.shape[1] // self.n_parties
        starts = [i * stride for i in range(self.n_parties - 1)] + [(self.n_parties - 1) * stride]
        stops  = [(i + 1) * stride for i in range(self.n_parties - 1)] + [X.shape[1]]
        self.feature_slices_ = [slice(a, b) for a, b in zip(starts, stops)]
        self.classes_        = None
        if self.task == "classification":
            # Binary classif: no.classes and encoding
            self.classes_, y_numeric = np.unique(y, return_inverse=True)
            if len(self.classes_) < 2:
                raise ValueError("Classification requires at least two distinct classes.")
            y_numeric = y_numeric.astype(np.int64)
        else:
            # Regression
            y_numeric = y.astype(np.float32)

        # [RND SEED for deterministic reproduction]
        rng         = np.random.default_rng(self.random_state)
        generator   = torch.Generator().manual_seed(self.random_state)
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():   torch.cuda.manual_seed_all(self.random_state)

        # [Build client/server MLPs]
        self.device_        = torch.device(self.device)
        self.client_models_ = nn.ModuleList([self._make_mlp(sl.stop - sl.start, client_dims)
                                             for sl in self.feature_slices_]).to(self.device_)
        if self.task == "classification":   out_dim = len(self.classes_)
        else:                               out_dim = 1
        self.server_model_  = self._make_mlp(self.n_parties * client_dims[-1], server_dims, out_dim).to(self.device_)
        optimizer           = self._make_optimizer()

        # [Training & validation sets (early stop) → torch tensors]
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.as_tensor(y_numeric, dtype=torch.long if self.task == "classification" else torch.float32)
        if self.early_stopping:
            indices = rng.permutation(len(X))
            n_val   = max(1, int(round(len(X) * self.validation_fraction)))
            val_idx, train_indices = indices[:n_val], indices[n_val:]
            val_X = X_tensor[val_idx].to(self.device_)
            val_y = y_tensor[val_idx].to(self.device_)
        else:
            val_X = val_y = None
            train_indices = np.arange(len(X))

        # [Load dataset sampler & history record for epochs/batches]
        dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False, generator=generator)
        batch_records: list[dict] = []
        epoch_records: list[dict] = []
        self.weight_snapshots_: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

        # [Auxiliary variables]
        samples_seen            = 0
        best_loss               = float("inf")
        best_state              = None
        epochs_wo_improvement   = 0
        self.n_iter_            = 0
        snapshot_batch_iters    = {round((len(loader) / self.snapshot_frequency) * i)
                                   for i in range(1, self.snapshot_frequency + 1)}

        # [TRAIN LOOP - EPOCHS]
        fit_start = time.perf_counter()
        for epoch in range(self.epochs):

            # Set train mode
            self.client_models_.train()
            self.server_model_.train()
            loss_sum = 0.0
            seen = 0

            # [BATCH LOOP - ROUNDS]
            epoch_start = time.perf_counter()
            for batch_index, (batch_X, batch_y) in enumerate(loader, start=1):

                step_start = time.perf_counter()

                # [MAIN steps: forward/backward pass & optimizer]
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                prediction  = self._forward(batch_X)
                loss        = self._loss(prediction, batch_y)
                loss.backward()
                optimizer.step()
                if batch_index in snapshot_batch_iters: # save model weights
                    self._save_snapshot(self.n_iter_)

                # Update auxiliary variables                
                batch_n       = len(batch_X)
                samples_seen += batch_n
                seen         += batch_n
                loss_sum     += float(loss.item()) * batch_n
                self.n_iter_ += 1
                step_elapsed  = time.perf_counter() - step_start
                rep_values    =  batch_n * self.n_parties * client_dims[-1]
                backward_bytes = forward_bytes = rep_values * np.dtype(np.float32).itemsize
                # Update batch record history
                batch_records.append({
                    "iter":         self.n_iter_,
                    "epoch":        epoch + 1,
                    "batch":        batch_index,
                    "batch_size":   batch_n,
                    "loss":         float(loss.item()),
                    "samples_seen": samples_seen,
                    "step_time":    step_elapsed,
                    "elapsed_time": time.perf_counter() - fit_start,
                    "estimated_communication_bytes": int(forward_bytes + backward_bytes),
                })
                
            # Update auxiliary variables
            epoch_loss    = loss_sum / max(seen, 1)
            val_loss      = self._evaluate_loss(val_X, val_y) if val_X is not None else None
            epoch_elapsed = time.perf_counter() - epoch_start
            client_norm   = float(torch.sqrt(sum((p.detach()**2).sum() for p in self.client_models_.parameters())).item())
            server_norm   = float(torch.sqrt(sum((p.detach()**2).sum() for p in self.server_model_.parameters())).item())

            if self.verbose:
                suffix = f" - val_loss: {val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch + 1:03d}/{self.epochs} - loss: {epoch_loss:.4f}{suffix}")
            
            # Update epoch record history
            epoch_records.append({
                "epoch":            epoch + 1,
                "train_loss":       epoch_loss,
                "validation_loss":  val_loss,
                "epoch_time":       epoch_elapsed,
                "elapsed_time":     time.perf_counter() - fit_start,
                "optimizer_steps":  self.n_iter_,
                "samples_seen":     samples_seen,
                "client_w_l2_norm": client_norm,
                "server_w_l2_norm": server_norm,
                "total_w_l2_norm":  float(np.hypot(client_norm, server_norm))
            })

            # [Early stopping logic]
            if self.early_stopping and val_loss is not None:
                if val_loss < best_loss - self.min_delta:
                    best_loss  = val_loss
                    best_state = {"clients": copy.deepcopy(self.client_models_.state_dict()),
                                  "server": copy.deepcopy(self.server_model_.state_dict())}
                    self.best_epoch_ = epoch + 1
                    epochs_wo_improvement = 0
                else:
                    epochs_wo_improvement += 1
                    if epochs_wo_improvement >= self.patience:
                        if self.verbose:    print(f"Early stopping at epoch {epoch+1}; best epoch {self.best_epoch_}.")
                        break

        # [History dataframe epochs/batches]
        self.batch_history_ = pd.DataFrame.from_records(batch_records)
        self.epoch_history_ = pd.DataFrame.from_records(epoch_records)

        # [Early stopping restores best weights]
        if self.early_stopping and best_state is not None:
            self.client_models_.load_state_dict(best_state["clients"])
            self.server_model_.load_state_dict(best_state["server"])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict fitted model over X data.

        # Check and validate X
        if not hasattr(self, "server_model_"):
            raise RuntimeError("Call fit before predict.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        # Enable evaluation mode
        self.client_models_.eval()
        self.server_model_.eval()

        # Forward pass to obtain net input (aka predictions)
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = []
            for start in range(0, len(tensor), self.batch_size):
                outputs.append(self._forward(tensor[start:start + self.batch_size]).cpu())
            predictions = torch.cat(outputs, dim=0).numpy()

        # Return net input for regression or class of the predicted max value for binary classif.
        if self.task == "classification":
            return self.classes_[predictions.argmax(axis=1)]
        return predictions.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Predict probabilities for binary classification tasks (via softmax)

        # Check and validate X
        if self.task != "classification":
            raise AttributeError("predict_proba is available only for classification.")
        if not hasattr(self, "server_model_"):
            raise RuntimeError("Call fit before predict_proba.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        # Enable evaluation mode
        self.client_models_.eval()
        self.server_model_.eval()

        # Forward pass to obtain net input and softmax
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = [self._forward(tensor[i:i + self.batch_size]).cpu() for i in range(0, len(tensor), self.batch_size)]
            logits = torch.cat(outputs, dim=0)
            return torch.softmax(logits, dim=1).numpy()

    def _forward_with_snapshot(self, X: torch.Tensor, snapshot: dict) -> torch.Tensor:
        # Aux. fn. that applies a forward pass w/ the given MLP weight snapshot for data X
        
        # Client forward pass and concat representations
        representations = []
        for model, state, sl in zip(self.client_models_, snapshot["clients"], self.feature_slices_):
            state = {k: v.to(X.device) for k, v in state.items()}
            representations.append(torch.func.functional_call(model, state, (X[:, sl],)))
        # Server forward pass
        server_state = {k: v.to(X.device) for k, v in snapshot["server"].items()}
        return torch.func.functional_call(self.server_model_, server_state, (torch.cat(representations, dim=1),))

    def predict_proba_at(self, X: np.ndarray, iter: int) -> np.ndarray:
        # Same as predict_proba but with the saved weights for the given iter

        # Check and validate X
        if self.task != "classification":
            raise AttributeError("predict_proba_at is available only for classification.")
        if iter not in self.weight_snapshots_:
            raise ValueError(f"No weight snapshot available for iter {iter}.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        # Enable evaluation mode and restore saved weights
        snapshot = self.weight_snapshots_[iter]
        self.client_models_.eval()
        self.server_model_.eval()

        # Forward pass and softmax
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = [self._forward_with_snapshot(tensor[i:i + self.batch_size], snapshot).cpu()
                       for i in range(0, len(tensor), self.batch_size)]
            logits = torch.cat(outputs, dim=0)
            return torch.softmax(logits, dim=1).numpy()

    def predict_at(self, X: np.ndarray, iter: int) -> np.ndarray:
        # Same as predict but with the saved weights for the given iter

        # Check and validate X
        if iter not in self.weight_snapshots_:
            raise ValueError(f"No weight snapshot available for iter {iter}.")
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        # Enable evaluation mode and restore saved weights
        snapshot = self.weight_snapshots_[iter]
        self.client_models_.eval()
        self.server_model_.eval()

        # Forward pass and softmax
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(self.device_)
            outputs = [self._forward_with_snapshot(tensor[i:i + self.batch_size], snapshot).cpu()
                       for i in range(0, len(tensor), self.batch_size)]
            predictions = torch.cat(outputs, dim=0).numpy()
            # Return net input for regression or class of the predicted max value for binary classif.
            if self.task == "classification":
                return self.classes_[predictions.argmax(axis=1)]
            return predictions.reshape(-1)


    def get_results(self, train_Xy, test_Xy, name=None):
        # Returns dataframe with the evolution of different score metrics
        if self.snapshot_frequency == 0:
            raise AttributeError("No evolution results are possible w/ nil `snapshot_frequency`!")
        if not self.weight_snapshots_:
            raise ValueError("Empty weight dictionary. Train the model first!")

        Xtrain, ytrain  = train_Xy
        Xtest, ytest    = test_Xy
        iter = sorted(self.weight_snapshots_)
        results = []
        if self.task == 'classification':
            from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
            for i in iter:
                proba_train = self.predict_proba_at(Xtrain, iter=i)
                proba_test  = self.predict_proba_at(Xtest, iter=i)
                pred_train  = self.classes_[proba_train.argmax(axis=1)]
                pred_test   = self.classes_[proba_test.argmax(axis=1)]
                results.append({
                    "iter": i,
                    "acc_train": accuracy_score(ytrain, pred_train),
                    "auc_train": roc_auc_score(ytrain, proba_train[:, 1]),
                    "logloss_train": log_loss(ytrain, proba_train),
                    "acc_test": accuracy_score(ytest, pred_test),
                    "auc_test": roc_auc_score(ytest, proba_test[:, 1]),
                    "logloss_test": log_loss(ytest, proba_test),
                })
        else:
            from sklearn.metrics import root_mean_squared_error, r2_score
            for i in iter:
                pred_train = self.predict_at(Xtrain, iter=1)
                pred_test  = self.predict_at(Xtest,  iter=i)
                results.append({
                    "iter": i,
                    "rmse_train": root_mean_squared_error(ytrain, pred_train),
                    "r2_train": r2_score(ytrain, pred_train),
                    "rmse_test": root_mean_squared_error(ytest, pred_test),
                    "r2_test": r2_score(ytest, pred_test),
                })

        metrics_history = pd.DataFrame(results)
        return metrics_history.assign(name=name) if name else metrics_history

class SplitNNClassifier(ClassifierMixin, SplitNN):
    """Wrapper for classification. Explicit init params for sklearn support."""
    def __init__(
        self,
        # NN
        n_parties:              int = 2,
        hidden_dims_clients:    Sequence[int]  = (32, 64),
        hidden_dims_server:     Sequence[int]  = (64,),
        activation:             ActivationName = "relu",
        # FIT
        epochs:                 int = 25,
        batch_size:             int = 64,
        dropout:                float = 0.0,
        # OPTIM
        optimizer:              OptimizerName = "sgd",
        learning_rate:          float = 0.05,
        momentum:               float = 0.0,
        weight_decay:           float = 0.0,
        # TRAIN STOP
        early_stopping:         bool  = False,
        patience:               int   = 10,
        min_delta:              float = 0.0,
        validation_fraction:    float = 0.1,
        # MISC
        shuffle:                bool = True,
        random_state:           int  = 42,
        device:                 str  = "cpu",
        verbose:                bool = True,
        snapshot_frequency:     int  = 0
    ) -> None:
        params = locals().copy()
        params.pop('self', None)
        params.pop('__class__', None)
        super().__init__(task="classification", **params)


class SplitNNRegressor(RegressorMixin, SplitNN):
    """Wrapper for regression. Explicit init params for sklearn support."""
    def __init__(
        self,
        # NN
        n_parties:              int = 2,
        hidden_dims_clients:    Sequence[int]  = (32, 64),
        hidden_dims_server:     Sequence[int]  = (64,),
        activation:             ActivationName = "relu",
        # FIT
        epochs:                 int = 25,
        batch_size:             int = 64,
        dropout:                float = 0.0,
        # OPTIM
        optimizer:              OptimizerName = "sgd",
        learning_rate:          float = 0.05,
        momentum:               float = 0.0,
        weight_decay:           float = 0.0,
        # TRAIN STOP
        early_stopping:         bool  = False,
        patience:               int   = 10,
        min_delta:              float = 0.0,
        validation_fraction:    float = 0.1,
        # MISC
        shuffle:                bool = True,
        random_state:           int  = 42,
        device:                 str  = "cpu",
        verbose:                bool = True,
        snapshot_frequency:     int  = 0
    ) -> None:
        params = locals().copy()
        params.pop('self', None)
        params.pop('__class__', None)       
        super().__init__(task="regression", **params)
