"""Domain-generalization training wrappers: GroupDRO, IRM, and CORAL.

Each trainer wraps a user-supplied PyTorch classification model and trains it
with a domain-generalization objective, where *groups* correspond to disaster
events (or other domain indicators).

Usage pattern
-------------
>>> trainer = GroupDROTrainer(model, n_groups=5, lr=1e-3)
>>> trainer.fit(train_loader)            # DataLoader yields (x, y, group_idx)
>>> predictions = trainer.predict(X)     # numpy array
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.utils.data import DataLoader


# ======================================================================
# GroupDRO
# ======================================================================

class GroupDROTrainer:
    """Group Distributionally Robust Optimization (Sagawa et al., 2020).

    Maintains per-group weights and up-weights the worst-performing group at
    each step.

    Parameters
    ----------
    model : nn.Module
        PyTorch classifier returning logits of shape ``(B, C)``.
    n_groups : int
        Total number of groups (events).
    lr : float
        Learning rate for Adam optimizer.
    weight_decay : float
        L2 regularization.
    epochs : int
        Number of training epochs.
    group_adj_step : float
        Step size for the exponentiated-gradient update on group weights.
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        n_groups: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        group_adj_step: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.n_groups = n_groups
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.group_adj_step = group_adj_step
        self.device = torch.device(device)

        # Uniform initial group weights on simplex
        self._group_weights = torch.ones(n_groups, device=self.device) / n_groups

    def fit(self, train_loader: DataLoader) -> "GroupDROTrainer":
        """Train the model with GroupDRO.

        The *train_loader* must yield batches of ``(features, labels, group_ids)``
        where ``group_ids`` are integer indices in ``[0, n_groups)``.

        Parameters
        ----------
        train_loader : DataLoader

        Returns
        -------
        self
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

        self.model.train()
        for _epoch in range(self.epochs):
            for x, y, g in train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                g = g.to(self.device, dtype=torch.long)

                logits = self.model(x)
                per_sample_loss = criterion(logits, y)

                # Compute mean loss per group
                group_losses = torch.zeros(
                    self.n_groups, device=self.device, dtype=torch.float32
                )
                group_counts = torch.zeros(
                    self.n_groups, device=self.device, dtype=torch.float32
                )
                for gi in range(self.n_groups):
                    mask = g == gi
                    if mask.any():
                        group_losses[gi] = per_sample_loss[mask].mean()
                        group_counts[gi] = mask.sum().float()

                active_mask = group_counts > 0
                if not active_mask.any():
                    continue

                # Exponentiated gradient step on group weights for active groups in batch
                self._group_weights[active_mask] = self._group_weights[active_mask] * torch.exp(
                    self.group_adj_step * group_losses[active_mask].detach()
                )
                self._group_weights = self._group_weights / self._group_weights.sum()

                # Weighted loss: normalized across active groups in this minibatch
                # so that inactive groups do not artificially deflate the gradient norm
                active_w = self._group_weights[active_mask]
                active_w_norm = active_w / active_w.sum()
                loss = (group_losses[active_mask] * active_w_norm).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Return predicted class labels for numpy feature matrix *X*.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples,)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return softmax probabilities for numpy feature matrix *X*.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()


# ======================================================================
# IRM (Invariant Risk Minimization)
# ======================================================================

class IRMTrainer:
    """Invariant Risk Minimization (Arjovsky et al., 2019).

    Adds an IRM penalty that encourages the optimal linear classifier on top of
    the representation to be invariant across groups.

    Parameters
    ----------
    model : nn.Module
        PyTorch classifier returning logits of shape ``(B, C)``.
    n_groups : int
        Number of environment / group indices.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        L2 regularization.
    epochs : int
        Number of training epochs.
    irm_lambda : float
        Weight of the IRM penalty term (default ``1e2``).
    irm_anneal_epochs : int
        Number of warm-up epochs during which ``irm_lambda`` is set to ``1.0``
        to allow the representation to form before enforcing invariance.
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        n_groups: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        irm_lambda: float = 1e2,
        irm_anneal_epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.n_groups = n_groups
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.irm_lambda = irm_lambda
        self.irm_anneal_epochs = irm_anneal_epochs
        self.device = torch.device(device)

    @staticmethod
    def _irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the IRMv1 gradient penalty for a single environment."""
        scale = torch.ones(1, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, labels)
        (grad,) = torch.autograd.grad(loss, scale, create_graph=True)
        return (grad ** 2).sum()

    def fit(self, train_loader: DataLoader) -> "IRMTrainer":
        """Train the model with the IRMv1 objective.

        The *train_loader* must yield ``(features, labels, group_ids)``.

        Parameters
        ----------
        train_loader : DataLoader

        Returns
        -------
        self
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.epochs):
            lam = 1.0 if epoch < self.irm_anneal_epochs else self.irm_lambda

            for x, y, g in train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                g = g.to(self.device, dtype=torch.long)

                logits = self.model(x)
                erm_loss = F.cross_entropy(logits, y)

                # Penalty: sum of per-environment gradient penalties
                penalty = torch.tensor(0.0, device=self.device)
                n_envs_seen = 0
                for gi in range(self.n_groups):
                    mask = g == gi
                    if mask.sum() < 6:
                        continue
                    penalty = penalty + self._irm_penalty(logits[mask], y[mask])
                    n_envs_seen += 1

                if n_envs_seen > 0:
                    penalty = penalty / n_envs_seen

                loss = erm_loss + lam * penalty

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Return predicted class labels.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples,)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return softmax probabilities.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()


# ======================================================================
# CORAL (CORrelation ALignment)
# ======================================================================

class CORALTrainer:
    """Deep CORAL (Sun & Saenko, 2016) for domain generalization.

    Minimizes the cross-entropy loss plus a CORAL penalty that aligns the
    second-order feature statistics across groups.

    Parameters
    ----------
    model : nn.Module
        PyTorch classifier returning logits of shape ``(B, C)``.  The model
        is expected to expose a ``features`` attribute or method that returns
        intermediate representations; if it does not, logits are used directly
        as the representation for alignment.
    n_groups : int
        Number of domain / group indices.
    lr : float
        Learning rate for Adam.
    weight_decay : float
        L2 regularization.
    epochs : int
        Training epochs.
    coral_lambda : float
        Weight of the CORAL penalty (default ``1.0``).
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        model: nn.Module,
        n_groups: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        coral_lambda: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.n_groups = n_groups
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.coral_lambda = coral_lambda
        self.device = torch.device(device)

    @staticmethod
    def _covariance(X: torch.Tensor) -> torch.Tensor:
        """Compute the covariance matrix of rows of *X*."""
        n = X.size(0)
        X_centered = X - X.mean(dim=0, keepdim=True)
        return (X_centered.T @ X_centered) / max(n - 1, 1)

    @staticmethod
    def _coral_penalty(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """Frobenius-norm distance between covariance matrices."""
        cov_a = CORALTrainer._covariance(feat_a)
        cov_b = CORALTrainer._covariance(feat_b)
        d = feat_a.size(1)
        return ((cov_a - cov_b) ** 2).sum() / (4 * d * d)

    def _get_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features, logits)`` from the model.

        If the model has a callable ``features`` attribute, use it.  Otherwise,
        the logits themselves serve as both the representation and the output.
        """
        if hasattr(self.model, "features") and callable(self.model.features):
            feats = self.model.features(x)
            logits = self.model(x)
            return feats, logits
        logits = self.model(x)
        return logits, logits

    def fit(self, train_loader: DataLoader) -> "CORALTrainer":
        """Train the model with the CORAL penalty.

        The *train_loader* must yield ``(features, labels, group_ids)``.

        Parameters
        ----------
        train_loader : DataLoader

        Returns
        -------
        self
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for _epoch in range(self.epochs):
            for x, y, g in train_loader:
                x = x.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.long)
                g = g.to(self.device, dtype=torch.long)

                feats, logits = self._get_features(x)
                erm_loss = F.cross_entropy(logits, y)

                # Fast CORAL penalty: align each group covariance to pooled batch covariance
                coral_loss = torch.tensor(0.0, device=self.device)
                unique_groups = g.unique().tolist()
                n_groups_present = 0
                cov_pooled = self._covariance(feats)

                for gi in unique_groups:
                    mask_i = g == gi
                    if mask_i.sum() >= 4:
                        cov_i = self._covariance(feats[mask_i])
                        d = feats.size(1)
                        coral_loss = coral_loss + ((cov_i - cov_pooled) ** 2).sum() / (4 * d * d)
                        n_groups_present += 1

                if n_groups_present > 0:
                    coral_loss = coral_loss / n_groups_present

                loss = erm_loss + self.coral_lambda * coral_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Return predicted class labels.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples,)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return softmax probabilities.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
        """
        self.model.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()
