import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from .loss import SamplingRelevanceLoss


class ResidualBlock(torch.nn.Module):
    """
    Residual Block
    """
    def __init__(self, model: torch.nn.Module):
        """
        Residual Block
        """
        super(ResidualBlock, self).__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) + x


class FeedforwardBlock(torch.nn.Module):
    """
    Feedforward Block
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Feedforward Block
        """
        super(FeedforwardBlock, self).__init__()
        self.ffn = nn.Sequential(*[
                                 Rearrange('b n c -> b c n'),
                                 nn.BatchNorm1d(in_features),
                                 Rearrange('b c n -> b n c'),
                                 nn.Linear(in_features, in_features*10, bias=False),
                                 nn.GELU(),
                                 Rearrange('b n c -> b c n'),
                                 nn.BatchNorm1d(in_features*10),
                                 Rearrange('b c n -> b n c'),
                                 nn.Linear(in_features*10, out_features, bias=False),
                                 nn.GELU(),
                                 ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class GCNLayer(torch.nn.Module):
    """
    Graph Attention Layer
    """
    def __init__(
            self, in_features: int, out_features: int
    ) -> None:
        """
        Initialize the graph convolution layer.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.
        """
        super(GCNLayer, self).__init__()
        self.weight = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph convolution layer.
        Args:
            x: The input data.
            adj: The adjacency matrix.

        Returns:
            The output of the graph convolution layer.
        """
        hat_adj = adj + torch.eye(adj.shape[0]).to(adj.device)  # add self-connections
        deg = hat_adj.sum(dim=1, keepdim=True)  # degree matrix
        deg_sqrt = deg.pow(-0.5)
        hat_adj = deg_sqrt * hat_adj * deg_sqrt

        output = torch.matmul(hat_adj, x)  # graph convolution
        output = self.weight(output)
        output = nn.functional.gelu(output)

        return output


class GCN(pl.LightningModule):
    """
    Instrumental variable learner based on Graph Convolutional Neural Network (CNN)
    """

    def __init__(self, input_channels: int, hidden_channels: list, num_classes: int, lr: float = 0.001,
                 weight_decay: float = 0.0) -> None:
        """
        Initialize the MLP learner with Graph Convolutional Kernels.
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
        self.num_points = 10

        # build the network
        gcn = []
        kernel_size = 512
        # add embedding layer
        self.mat_embedding = nn.Sequential(*[nn.Conv1d(1, self.hidden_channels[0], kernel_size=kernel_size, stride=kernel_size, bias=False),
                                         Reduce('b c l -> b c', 'mean')])
        # add an embedding layer for single sample input
        self.row_embedding = nn.Linear(1, self.hidden_channels[0], bias=False)
        # add graph convolution layers
        layers = []
        for i in range(len(self.hidden_channels) - 1):
            layer = nn.ModuleDict({'gcn': GCNLayer(self.hidden_channels[i], self.hidden_channels[i]),
                                    'ffn': ResidualBlock(FeedforwardBlock(self.hidden_channels[i], self.hidden_channels[i]))})
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self._initialize_weights(self)
        self.mat_proj = nn.Linear(self.hidden_channels[-1], self.num_classes, bias=False)
        self.row_proj = nn.Linear(self.hidden_channels[-1], 1, bias=False)

    def _initialize_weights(self, model: nn.Module) -> None:
        for m in model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adj = torch.Tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]]).to(self.device)

        # forward for predicting the ATE
        if self.training:
            # get the embedding for each row
            xx = x[:, :self.num_points].reshape(-1, x.shape[-1])
            zz, tt, yy = xx[..., 0], xx[..., 1], xx[..., 2]
            zz = self.row_embedding(zz.unsqueeze(1))
            tt = self.row_embedding(tt.unsqueeze(1))
            yy = self.row_embedding(yy.unsqueeze(1))
            xx = torch.stack([zz, tt, yy], dim=1)
            for layer in self.layers:
                xx = layer['gcn'](xx, adj)
                xx = layer['ffn'](xx)
            xx = self.row_proj(xx.mean(dim=1))
            xx = [xxx.mean() for xxx in torch.split(xx, self.num_points, dim=0)]
            xx = torch.stack(xx, dim=0)
        else:
            xx = []

        z, t, y = x[..., 0], x[..., 1], x[..., 2]
        z = self.mat_embedding(z.unsqueeze(1))
        t = self.mat_embedding(t.unsqueeze(1))
        y = self.mat_embedding(y.unsqueeze(1))
        x = torch.stack([z, t, y], dim=1)
        for layer in self.layers:
            x = layer['gcn'](x, adj)
            x = layer['ffn'](x)
        x = self.mat_proj(x.mean(dim=1))

        return x, xx

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'val')

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'test')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=0.0)
        return [optimizer], [scheduler]

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
        y_group, y_point = self(x)
        if self.training:
            # add the loss for the relevance assumption
            relevance_loss = nn.MSELoss()(y_point.squeeze(), y)
            loss = nn.MSELoss()(y_group.squeeze(), y) + relevance_loss
        else:
            loss = nn.MSELoss()(y_group.squeeze(), y)

        self.log(f'{stage}_loss', loss, prog_bar=True)
        return loss
