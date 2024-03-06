import torch
import torch.nn as nn
from typing import Callable
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from .mine import MINE

class LinearBlock(torch.nn.Module):
    def __init__(
            self, in_features: int, out_features: int, activation: Callable
    ) -> None:
        """
        Initialize the linear block.
        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            activation: The activation function to use
        """
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear block.
        Args:
            x: The input data.

        Returns:
            The output of the linear block.
        """
        return self.activation(self.batch_norm(self.linear(x)))


class AttentionBlock(torch.nn.Module):
    """
    Attention Block
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        """
        Attention Block

        Args:
            in_features: input feature size
            hidden_features: hidden feature size
            out_features: output feature size
        """
        super(AttentionBlock, self).__init__()
        self.q = nn.Linear(in_features, hidden_features, bias=False)
        self.k = nn.Linear(in_features, hidden_features, bias=False)
        self.v = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(torch.matmul(self.q(x), self.k(x).transpose(1, 2)), dim=1)
        v = torch.matmul(attention, self.v(x))
        return v


class MLP(pl.LightningModule):
    """
    Instrumental variable learner based on Multi-Layer Perceptron (MLP)
    """

    def __init__(self, input_channels: int, hidden_channels: list, num_classes: int, lr: float = 0.001,
                 weight_decay: float = 0.0) -> None:
        """
        Initialize the MLP learner with feedforward layers.
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
        self.num_classes = num_classes

        # build the network
        # b: batch size, l: n_samples, c: features
        # Concatenate along the l dimension
        layers = [Rearrange('b l c -> (b l) c'),
                  LinearBlock(self.input_channels, self.hidden_channels[0], nn.ReLU),
                  nn.BatchNorm1d(self.hidden_channels[0])
                  ]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(
                LinearBlock(self.hidden_channels[i], self.hidden_channels[i + 1], nn.ReLU)
            )
        layers.append(nn.Linear(self.hidden_channels[-1], self.num_classes))
        print(layers)
        self.model = nn.Sequential(*layers)
        self._initialize_weights(self.model)

    def _initialize_weights(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)

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
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss


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


class PoolingMLP(pl.LightningModule):
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
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)

        # encode the Z _||_ Y | T
        # z_t = x[:, :2]
        # mine = self.mine(z_t, y)
        # loss += mine

        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss
