import torch
import torch.nn as nn
from typing import Callable
import pytorch_lightning as pl
from einops.layers.torch import Rearrange


class ConvBlock(torch.nn.Module):
    def __init__(
            self, in_features: int, out_features: int, kernel_size: [int, tuple], activation: Callable
    ) -> None:
        """
        Initialize the convolutional block.
        Args:
            in_features: The number of input features.
            out_features: The number of output features.
            activation: The activation function to use
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=kernel_size, padding=0, stride=kernel_size//2)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear block.
        Args:
            x: The input data.

        Returns:
            The output of the linear block.
        """
        return self.activation(self.batch_norm(self.conv(x)))


class CNN(pl.LightningModule):
    """
    Instrumental variable learner based on Convolutional Neural Network (CNN)
    """

    def __init__(self, input_channels: int, hidden_channels: list, num_classes: int, lr: float = 0.001,
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
        self.num_classes = num_classes

        # build the network
        layers = [nn.BatchNorm1d(self.input_channels),
                  nn.Linear(self.input_channels, self.hidden_channels[0], bias=False),
                  Rearrange('b l c -> b c l'),
                  ConvBlock(self.hidden_channels[0], self.hidden_channels[0], kernel_size=256, activation=nn.ReLU)]
        for i in range(len(self.hidden_channels) - 1):
            layers.append(ConvBlock(self.hidden_channels[i], self.hidden_channels[i+1], kernel_size=4, activation=nn.ReLU))
        # projection layer
        layers.append(Rearrange('b c l -> b (c l)'))
        layers.append(nn.AdaptiveAvgPool1d(self.hidden_channels[-1]))
        layers.append(nn.Linear(self.hidden_channels[-1], self.num_classes))
        self.model = nn.Sequential(*layers)
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
        loss = nn.MSELoss()(y_hat.squeeze(), y)
        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss
