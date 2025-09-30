import copy
from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from .transformer import MultiHeadAttentionBlock


class AxialEncoderBlock(nn.Module):
    """
    Axial encoder block that alternates attention across the sample axis (sequence length)
    and the feature axis.

    Input tensor shape: (batch_size, seq_len, num_features, d_model)
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        mode: str = "feature",  # "feature" or "sample"
    ) -> None:
        super().__init__()
        assert mode in ("feature", "sample")
        self.mode = mode
        self.attn_norm = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttentionBlock(
            n_heads=n_heads,
            in_dim=d_model,
            qk_dim=qk_dim or max(1, d_model // max(1, n_heads)),
            v_dim=v_dim or max(1, d_model // max(1, n_heads)),
            out_dim=d_model,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_model))
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, F, D)
        Returns:
            (B, S, F, D)
        """
        B, S, F, D = x.shape
        if self.mode == "sample":
            # Sample-axis attention: operate along S for each feature independently
            z = self.attn_norm(x)
            z = z.permute(0, 2, 1, 3).contiguous()  # (B, F, S, D)
            z = z.view(B * F, S, D)  # (B*F, S, D)
            z, _ = self.mha(z, z, z, padding_mask=None)  # (B*F, S, D)
            z = z.view(B, F, S, D).permute(0, 2, 1, 3).contiguous()  # (B, S, F, D)
            x = x + self.dropout(z)
        else:
            # Feature-axis attention: operate across features for each time step independently
            a = self.attn_norm(x)
            a = a.view(B * S, F, D)  # (B*S, F, D)
            a, _ = self.mha(a, a, a, padding_mask=None)  # (B*S, F, D)
            a = a.view(B, S, F, D)
            x = x + self.dropout(a)

        # MLP
        m = self.mlp_norm(x)
        m = self.mlp(m)
        x = x + self.mlp_dropout(m)
        return x


class AxialTransformer(pl.LightningModule):
    """
    Axial Transformer for tabular sequences (samples × features).

    Input: x of shape (batch_size, seq_len, num_features)
    Processing:
      - Per-cell projection to d_model
      - N axial encoder blocks alternating sample-axis and feature-axis attention
      - Global average pooling over (seq_len, num_features)
      - Linear head to scalar output (tau)
    """

    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        d_model: int,
        d_hidden: int,
        dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        feature_group_ids: Optional[list] = None,
        num_feature_groups: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Project scalar cell value -> d_model embedding
        self.input_proj = nn.Linear(1, d_model)

        # Build alternating blocks: feature-wise, sample-wise, feature-wise, ...
        blocks = []
        for i in range(n_blocks):
            mode = "feature" if i % 2 == 0 else "sample"
            blocks.append(
                AxialEncoderBlock(
                    n_heads=n_heads,
                    d_model=d_model,
                    d_hidden=d_hidden,
                    dropout=dropout,
                    qk_dim=qk_dim,
                    v_dim=v_dim,
                    mode=mode,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        self.final_linear = nn.Linear(d_model, 1)
        self.lr = lr
        self.weight_decay = weight_decay

        # Optional feature-group embeddings (e.g., T, Y, Z)
        self.feature_group_embed: Optional[nn.Embedding]
        if feature_group_ids is not None:
            fg_ids = torch.as_tensor(feature_group_ids, dtype=torch.long)
            self.register_buffer("feature_group_ids", fg_ids, persistent=False)
            if num_feature_groups is None:
                num_feature_groups = int(fg_ids.max().item()) + 1
            self.feature_group_embed = nn.Embedding(num_feature_groups, d_model)
        else:
            self.feature_group_ids = None
            self.feature_group_embed = None

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, S, F)
        Returns:
            (B, 1)
        """
        x = x.unsqueeze(-1)  # (B, S, F, 1)
        x = self.input_proj(x)  # (B, S, F, D)

        # Add feature group/type embeddings if provided
        if self.feature_group_embed is not None and self.feature_group_ids is not None:
            fg_ids = self.feature_group_ids.to(x.device)
            feat_emb = self.feature_group_embed(fg_ids)  # (F, D)
            x = x + feat_emb.unsqueeze(0).unsqueeze(1)  # broadcast to (B, S, F, D)
        x = self.encoder(x)  # (B, S, F, D)
        x = x.mean(dim=(1, 2))  # (B, D) global average pooling over samples and features
        out = self.final_linear(x)  # (B, 1)
        return out

    # Lightning hooks
    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

    def _step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.MSELoss()(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss


