import os
import sys
from typing import Optional, List

import numpy as np
import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd

# Ensure local TabPFN package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TABPFN_SRC = os.path.join(PROJECT_ROOT, "TabPFN", "src")
if TABPFN_SRC not in sys.path:
    sys.path.insert(0, TABPFN_SRC)

from tabpfn.regressor import TabPFNRegressor  # type: ignore
from .base_estimator import BaseEstimator


class SamplePoolingHead(nn.Module):
    """
    Per-sample projection -> mean pool over samples -> scalar.
    Uses LazyLinear to avoid re-creating modules after optimizer init.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        # Lazy first layer adapts to embedding dim on first forward
        self.proj = nn.Sequential(
            nn.LazyLinear(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
        )
        self.out = nn.Linear(self.hidden_dim, 1)

    def forward(self, sample_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sample_embeddings: (S, E) or (B, S, E)
        Returns:
            scalar tensor (,) for one or (B,) for batch
        """
        if sample_embeddings.dim() == 2:
            z = self.proj(sample_embeddings)
            pooled = z.mean(dim=0)  # (H)
            out = self.out(pooled)
            return out.squeeze()
        elif sample_embeddings.dim() == 3:
            z = self.proj(sample_embeddings)
            pooled = z.mean(dim=1)  # (B, H)
            out = self.out(pooled)
            return out.squeeze(-1)
        else:
            raise ValueError("sample_embeddings must be 2D or 3D")


class TabPFNPoolingRegressor(pl.LightningModule, BaseEstimator):
    """
    Wrap TabPFNRegressor as a frozen feature extractor and train a learnable
    pooling head to predict dataset-level tau.

    For each dataset (seq_len x features), we:
      1) split (X, y) with y = Y column, X = remaining features (e.g., T and Zs)
      2) call TabPFNRegressor.fit(X, y) (no parameter updates to backbone)
      3) extract train embeddings via get_embeddings(X, data_source='train')
      4) average over ensemble, pool over samples with learnable head -> tau
    """

    def __init__(
        self,
        n_estimators: int = 4,
        head_hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        tabpfn_model_path: str = "auto",
        device: str = "auto",
        cache_dir: Optional[str] = None,
        expected_embedding_dim: int = 192,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.pool_head = SamplePoolingHead(hidden_dim=head_hidden_dim)
        # caching
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, np.ndarray] = {}

        # Initialize TabPFNRegressor (frozen backbone)
        self.tabpfn = TabPFNRegressor(
            n_estimators=n_estimators,
            model_path=tabpfn_model_path,
            device=device,
            fit_mode="fit_preprocessors",
            ignore_pretraining_limits=True,
        )

    def _split_features(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Split raw tensor (S, F) into (X_np, y_np). Assumes columns [T, Y, Z...]."""
        assert x.dim() == 2, "Expect (S, F) per dataset"
        y_np = x[:, 1].detach().cpu().numpy()
        if x.shape[1] > 2:
            X_np = torch.cat([x[:, :1], x[:, 2:]], dim=1).detach().cpu().numpy()
        else:
            X_np = x[:, :1].detach().cpu().numpy()
        return X_np, y_np

    def _embed_one(self, x: torch.Tensor) -> torch.Tensor:
        """Fit TabPFN on this dataset and return averaged sample embeddings (S, E), with caching."""
        key = self._dataset_key(x)
        emb_np: Optional[np.ndarray] = None
        # in-memory cache
        if key in self._mem_cache:
            emb_np = self._mem_cache[key]
        # disk cache
        if emb_np is None and self.cache_dir is not None:
            fpath = self.cache_dir / f"{key}.npy"
            if fpath.exists():
                try:
                    emb_np = np.load(fpath)
                except Exception:
                    emb_np = None
        # compute
        if emb_np is None:
            X_np, y_np = self._split_features(x)
            with torch.no_grad():
                self.tabpfn.fit(X_np, y_np)
                emb_np = self.tabpfn.get_embeddings(X_np, data_source="train")  # (N_est, S, E)
            if emb_np.ndim == 3:
                emb_np = emb_np.mean(axis=0)
            self._mem_cache[key] = emb_np
            if self.cache_dir is not None:
                try:
                    np.save(self.cache_dir / f"{key}.npy", emb_np)
                except Exception:
                    pass
        emb_t = torch.tensor(emb_np, dtype=torch.float32, device=self.device)
        return emb_t

    @staticmethod
    def _dataset_key(x: torch.Tensor) -> str:
        x_cpu = x.detach().to(dtype=torch.float32, device="cpu")
        h = hashlib.sha1()
        h.update(str(tuple(x_cpu.shape)).encode())
        h.update(x_cpu.numpy().tobytes())
        return h.hexdigest()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            emb = self._embed_one(x)  # (S, E)
            return self.pool_head(emb)
        elif x.dim() == 3:
            # batch of datasets
            preds: List[torch.Tensor] = []
            for i in range(x.shape[0]):
                emb = self._embed_one(x[i])
                preds.append(self.pool_head(emb))
            return torch.stack(preds, dim=0)
        else:
            raise ValueError("Input must be (S, F) or (B, S, F)")

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.pool_head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch  # x: (S,F) or (B,S,F), y: scalar or (B,)
        y_hat = self(x).squeeze()
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss


    def estimate(self, T, X, Z, Y, **kwargs) -> dict:
        """Estimate dataset-level tau for a single dataset.

        Expects pandas inputs. Concatenates columns into [T, Y, Z..., X...]
        layout to match this module's expected input ordering.
        """
        # Rebuild dataframe into Tensor; assumes this column order
        df = pd.concat([T, Y, Z, X], axis=1)
        x_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            tau_hat = self(x_tensor).squeeze().item()
        # No closed-form standard error available
        return {"tau": tau_hat, "se": None}
