import pandas as pd
import torch
import torch.nn as nn
from typing import Callable
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from huggingface_hub import PyTorchModelHubMixin

from .mine import MINE
from .mlp import LinearBlock
from .base_estimator import BaseEstimator

class ResLinearBlock(torch.nn.Module):
    """
    Linear Block with residual connection
    """
    def __init__(self, hidden_features: int, depth: int, activation: Callable):
        super().__init__()
        self.linear = nn.Sequential(*[LinearBlock(hidden_features, hidden_features, activation) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(x)


class PrePoolingBlock(torch.nn.Module):
    """
    Pre-Pooling Block projecting each sample to a vector
    """
    def __init__(self, in_features: int, hidden_features: int, depth: int, out_features: int):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.resblock = ResLinearBlock(hidden_features, depth, nn.ELU)
        self.fc_out = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.resblock(x)
        return self.fc_out(x)


class GlobalPoolingLayer(torch.nn.Module):
    """
    Global Pooling Layer pooling a dataset into a single vector
    """
    def __init__(self, pooling_type: str = 'mean'):
        super().__init__()
        if pooling_type == 'mean':
            self.pooling = Reduce('b l c -> b c', reduction='mean')
        elif pooling_type == 'max':
            self.pooling = Reduce('b l c -> b c', reduction='max')
        else:
            raise ValueError(f'Pooling type {pooling_type} not supported')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(x)


class PostPoolingBlock(torch.nn.Module):
    """
    Post-Pooling Block projecting the pooled vector of a dataset to a new vector
    """
    def __init__(self, in_features: int, hidden_features: int, depth: int, out_features: int):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.resblock = ResLinearBlock(hidden_features, depth, nn.ELU)
        self.fc_out = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.resblock(x)
        return self.fc_out(x)


class PoolingMLP(
    pl.LightningModule, 
    BaseEstimator, 
    PyTorchModelHubMixin, 
    repo_url="https://huggingface.co/learning-ivs/pooling-mlp"
):
    """
    Instrumental variable learner based on Multi-Layer Perceptron (MLP) with Pre- and Post-Pooling Blocks for
    dimensionality reduction.
    """

    def __init__(self, input_channels: int, hidden_channels: int, depth: int, num_classes: int, lr: float = 0.001,
                 weight_decay: float = 0.0) -> None:
        """
        Initialize the MLP learner with Convolutional Kernels.
        Args:
            input_channels: The number of input channels.
            hidden_channels: The number of hidden channels.
            num_classes: The number of output dimension.
            lr: The learning rate.
            weight_decay: The weight decay.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.depth = depth
        self.num_classes = num_classes
        self.mine = MINE(x_dim=2, y_dim=1)

        # build the network
        self.model = nn.Sequential(*[
            PrePoolingBlock(self.input_channels, self.hidden_channels, self.depth, self.hidden_channels),
            GlobalPoolingLayer(),
            PostPoolingBlock(self.hidden_channels, self.hidden_channels, self.depth, self.num_classes),
        ])
        self._initialize_weights(self.model)

    def _initialize_weights(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'val')

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'test')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        """
        Forward pass of the network.
        Args:
            batch: The input data.
            stage: The stage of the network.
        Returns:
            The output of the network.
        """
        x, y = batch
        y_hat = self(x).squeeze()
        loss = nn.MSELoss()(y_hat, y)

        # encode the Z _||_ Y | T
        # z_t = x[:, :2]
        # mine = self.mine(z_t, y)
        # loss += mine

        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss


    def estimate(self, T, X, Z, Y) -> dict:
        # rebuild dataframe into Tensor, assumes this same order
        df = pd.concat([T, Y, Z, X], axis=1)
        data = torch.tensor(df.to_numpy(), dtype=torch.float32)
        # create a batch dimension
        data = data.unsqueeze(0)
        tau = self.model(data).squeeze().item()
        
        # TODO we need a way to compute standard errors
        return {
            'tau': tau,
            'se': None
        }

