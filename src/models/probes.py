"""Probe classifiers on frozen feature representations.

Provides a lightweight scikit-learn logistic regression probe and a
PyTorch-based MLP probe, both exposing a unified ``fit`` / ``predict`` /
``predict_proba`` interface so they can be used interchangeably in the
benchmark evaluation pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Logistic Probe (scikit-learn)
# ---------------------------------------------------------------------------

class LogisticProbe:
    """Multinomial logistic regression on frozen features.

    Parameters
    ----------
    C : float
        Inverse regularization strength (default ``1.0``).
    max_iter : int
        Maximum solver iterations (default ``1000``).
    solver : str
        Optimization algorithm (default ``"lbfgs"``).
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: Optional[int] = None,
    ) -> None:
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )


    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> "LogisticProbe":
        """Fit the logistic probe.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``
            Frozen feature matrix.
        y : ndarray of shape ``(n_samples,)``
            Integer class labels.

        Returns
        -------
        self
        """
        self.model.fit(X, y)
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
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return class probability estimates.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
        """
        return self.model.predict_proba(X)


# ---------------------------------------------------------------------------
# MLP Probe (PyTorch)
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """Simple feed-forward network used by :class:`MLPProbe`."""

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (256,),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


class MLPProbe:
    """Multi-layer perceptron probe on frozen features.

    Parameters
    ----------
    hidden_dims : tuple[int, ...]
        Widths of hidden layers (default ``(256,)``).
    dropout : float
        Dropout probability (default ``0.1``).
    lr : float
        Learning rate for Adam (default ``1e-3``).
    weight_decay : float
        L2 regularization (default ``1e-4``).
    epochs : int
        Training epochs (default ``50``).
    batch_size : int
        Mini-batch size (default ``256``).
    device : str
        PyTorch device string (default ``"cpu"``).
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256,),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 256,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.random_state = random_state
        self._net: Optional[_MLPNet] = None
        self._classes: Optional[NDArray[np.integer]] = None

    # ----- training --------------------------------------------------------

    def fit(self, X: NDArray[np.floating], y: NDArray[np.integer]) -> "MLPProbe":
        """Train the MLP probe.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``
            Frozen feature matrix.
        y : ndarray of shape ``(n_samples,)``
            Integer class labels.

        Returns
        -------
        self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._classes = np.unique(y)
        n_classes = len(self._classes)
        # Re-map labels to 0..n_classes-1
        label_map = {int(c): i for i, c in enumerate(self._classes)}
        y_mapped = np.array([label_map[int(yi)] for yi in y], dtype=np.int64)

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y_mapped, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._net = _MLPNet(
            in_features=X.shape[1],
            n_classes=n_classes,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self._net.train()
        for _epoch in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self._net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return self

    # ----- inference -------------------------------------------------------

    def _forward(self, X: NDArray[np.floating]) -> torch.Tensor:
        """Run a forward pass and return logits (no grad)."""
        if self._net is None:
            raise RuntimeError("MLPProbe has not been fitted yet. Call fit() first.")
        self._net.eval()
        X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self._net(X_t)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Return predicted class labels.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples,)``
            Predicted labels in the original label space.
        """
        logits = self._forward(X)
        indices = logits.argmax(dim=1).cpu().numpy()
        assert self._classes is not None
        return self._classes[indices]

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return class probability estimates.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        ndarray of shape ``(n_samples, n_classes)``
            Softmax probabilities ordered by ``self._classes``.
        """
        logits = self._forward(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
