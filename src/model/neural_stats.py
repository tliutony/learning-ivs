"""
Modified from Neural Statistician implementation:
https://github.com/conormdurkan/neural-statistician/blob/master/synthetic
"""

import os
import sys
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F, init
import pytorch_lightning as pl

from .utils import (kl_diagnormal_diagnormal, kl_diagnormal_stdnormal,
                   gaussian_log_likelihood)


class NeuralStats(pl.LightningModule):
    """
    Instrumental variable learner based on Neural Statistician.
    """

    def __init__(self, sample_size=200, n_features=1, num_classes=1,
                     c_dim=3, n_hidden_statistic=128, hidden_dim_statistic=3,
                     n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                     nonlinearity=F.relu, lr: float = 0.001,
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
        self.sample_size = sample_size
        self.num_classes = num_classes
        self.n_features = n_features
        self.c_dim = c_dim
        self.n_hidden_statistic = n_hidden_statistic
        self.hidden_dim_statistic = hidden_dim_statistic
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        self.lr = lr
        self.weight_decay = weight_decay

        # build the network
        self.model = Statistician(sample_size=self.sample_size, n_features=self.n_features,
                                    c_dim=self.c_dim, n_hidden_statistic=self.n_hidden_statistic,
                                    hidden_dim_statistic=self.hidden_dim_statistic, n_stochastic=self.n_stochastic,
                                    z_dim=self.z_dim, n_hidden=self.n_hidden, hidden_dim=self.hidden_dim,
                                    nonlinearity=self.nonlinearity)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.model(x, y)

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
        outputs = self(x, y)
        alpha = 0.5 ** self.trainer.current_epoch

        if stage == 'train':
            mse_loss, loss, vlb = self.model.loss(outputs, weight=(alpha + 1))
            self.log(f'{stage}_mse_loss', mse_loss, on_step=True, prog_bar=True)
            self.log(f'{stage}_loss', loss, on_step=True, prog_bar=True)
            self.log(f'{stage}_vlb', vlb, on_step=True, prog_bar=True)
            return loss
        else:
            y_hat, y = outputs[-1]
            # examine the correlation between y and y_hat
            loss = F.mse_loss(y_hat, y)
            self.log(f'{stage}_mse_loss', loss, prog_bar=True)


# Model
class Statistician(nn.Module):
    def __init__(self, sample_size=200, num_classes=1, n_features=1,
                 c_dim=3, n_hidden_statistic=128, hidden_dim_statistic=3,
                 n_stochastic=1, z_dim=16, n_hidden=3, hidden_dim=128,
                 nonlinearity=F.relu, print_vars=False):
        """

        :param sample_size:
        :param n_features:
        :param c_dim:
        :param n_hidden_statistic:
        :param hidden_dim_statistic:
        :param n_stochastic:
        :param z_dim:
        :param n_hidden:
        :param hidden_dim:
        :param nonlinearity:
        :param print_vars:
        """
        super(Statistician, self).__init__()
        # data shape
        self.sample_size = sample_size
        self.n_features = n_features
        self.num_classes = num_classes

        # context
        self.c_dim = c_dim
        self.n_hidden_statistic = n_hidden_statistic
        self.hidden_dim_statistic = hidden_dim_statistic

        # latent
        self.n_stochastic = n_stochastic
        self.z_dim = z_dim
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        # statistic network
        statistic_args = (self.sample_size, self.n_features,
                          self.n_hidden_statistic, self.hidden_dim_statistic,
                          self.c_dim, self.nonlinearity)
        self.statistic_network = StatisticNetwork(*statistic_args)

        z_args = (self.sample_size, self.n_features,
                  self.n_hidden, self.hidden_dim, self.c_dim, self.z_dim,
                  self.nonlinearity)
        # inference networks
        # one for each stochastic layer
        self.inference_networks = nn.ModuleList([InferenceNetwork(*z_args)
                                                 for _ in range(self.n_stochastic)])

        # TODO: add regression network for supervised learning p(y|c)
        self.regression_network = RegressionNetwork(self.c_dim, self.num_classes,
                                                    self.n_hidden, self.hidden_dim, self.nonlinearity)


        # TODO: add regularization network for maintaining graph assumption

        # latent decoders
        # again, one for each stochastic layer
        self.latent_decoders = nn.ModuleList([LatentDecoder(*z_args)
                                              for _ in range(self.n_stochastic)])

        # observation decoder
        observation_args = (self.sample_size, self.n_features,
                            self.n_hidden, self.hidden_dim, self.c_dim,
                            self.n_stochastic, self.z_dim, self.nonlinearity)
        self.observation_decoder = ObservationDecoder(*observation_args)

        # initialize weights
        self.apply(self.weights_init)

        # print variables for sanity check and debugging
        if print_vars:
            for i, pair in enumerate(self.named_parameters()):
                name, param = pair
                print("{} --> {}, {}".format(i + 1, name, param.size()))
            print()

    def forward(self, x, y=None):
        # statistic network
        c_mean, c_logvar = self.statistic_network(x)
        # sampling from MoG
        c = self.reparameterize_gaussian(c_mean, c_logvar)

        # inference networks
        # q(z_{t+1}|z_t, x, c)
        qz_samples = []
        qz_params = []
        z = None
        for inference_network in self.inference_networks:
            z_mean, z_logvar = inference_network(x, z, c)
            qz_params.append([z_mean, z_logvar])
            z = self.reparameterize_gaussian(z_mean, z_logvar)
            qz_samples.append(z)

        # latent decoders
        # p(z_{t+1}|z_t, c)
        pz_params = []
        z = None
        for i, latent_decoder in enumerate(self.latent_decoders):
            z_mean, z_logvar = latent_decoder(z, c)
            pz_params.append([z_mean, z_logvar])
            z = qz_samples[i]

        # observation decoder
        zs = torch.cat(qz_samples, dim=1)
        x_mean, x_logvar = self.observation_decoder(zs, c)

        # regression network
        y_hat = self.regression_network(c)

        outputs = (
            (c_mean, c_logvar),
            (qz_params, pz_params),
            (x, x_mean, x_logvar),
            (y_hat, y)
        )

        return outputs

    def loss(self, outputs, weight):
        c_outputs, z_outputs, x_outputs, y_outputs = outputs
        batch_size = y_outputs[0].shape[0]

        # 1. Reconstruction loss
        x, x_mean, x_logvar = x_outputs
        recon_loss = gaussian_log_likelihood(x.view(-1, self.n_features),
                                             x_mean, x_logvar)
        recon_loss /= (batch_size * self.sample_size)

        # 2. KL Divergence terms
        kl = 0

        # a) Context divergence
        c_mean, c_logvar = c_outputs
        kl_c = kl_diagnormal_stdnormal(c_mean, c_logvar)
        kl += kl_c

        # b) Latent divergences
        qz_params, pz_params = z_outputs
        shapes = (
            (batch_size, self.sample_size, self.z_dim),
            (batch_size, 1, self.z_dim)
        )
        for i in range(self.n_stochastic):
            args = (qz_params[i][0].view(shapes[0]),
                    qz_params[i][1].view(shapes[0]),
                    pz_params[i][0].view(shapes[1] if i == 0 else shapes[0]),
                    pz_params[i][1].view(shapes[1] if i == 0 else shapes[0]))
            kl_z = kl_diagnormal_diagnormal(*args)
            kl += kl_z

        kl /= (batch_size * self.sample_size)

        # supervised regression
        y_hat, y = y_outputs
        regression_loss = F.mse_loss(y_hat, y)

        # Variational lower bound and weighted loss
        vlb = recon_loss - kl
        loss = - ((weight * recon_loss) - (kl / weight)) - regression_loss

        return regression_loss, loss, vlb

    def step(self, batch, alpha, optimizer, clip_gradients=True):
        assert self.training is True

        inputs = Variable(batch.cuda())
        outputs = self.forward(inputs)
        loss, vlb = self.loss(outputs, weight=(alpha + 1))

        # perform gradient update
        optimizer.zero_grad()
        loss.backward()
        if clip_gradients:
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
        optimizer.step()

        # output variational lower bound
        return vlb.data[0]

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = Variable(torch.randn(std.size()).cuda())
        return mean + std * eps

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            init.xavier_normal(m.weight.data, gain=init.calculate_gain('relu'))
            init.constant(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm1d):
            pass

# ************************************************* Essential Blocks ************************************************* #

# Module for residual/skip connections
class FCResBlock(nn.Module):
    def __init__(self, width, n, nonlinearity):
        """

        :param width:
        :param n:
        :param nonlinearity:
        """
        super(FCResBlock, self).__init__()
        self.n = n
        self.nonlinearity = nonlinearity
        self.block = nn.ModuleList([nn.Linear(width, width) for _ in range(self.n)])

    def forward(self, x):
        e = x + 0
        for i, layer in enumerate(self.block):
            e = layer(e)
            if i < (self.n - 1):
                e = self.nonlinearity(e)
        return self.nonlinearity(e + x)


# PRE-POOLING FOR STATISTIC NETWORK
class PrePool(nn.Module):
    def __init__(self, n_features, n_hidden, hidden_dim, nonlinearity):
        super(PrePool, self).__init__()
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_initial = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)
        self.fc_final = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        # reshape and initial affine
        e = x.view(-1, self.n_features)
        e = self.fc_initial(e)
        e = self.nonlinearity(e)

        # residual block
        e = self.fc_block(e)

        # final affine
        e = self.fc_final(e)

        return e


# POST POOLING FOR STATISTIC NETWORK
class PostPool(nn.Module):
    """

    """

    def __init__(self, n_hidden, hidden_dim, c_dim, nonlinearity):
        super(PostPool, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden,
                                   nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.c_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, e):
        e = self.fc_block(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.c_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.c_dim)

        mean, logvar = e[:, :self.c_dim], e[:, self.c_dim:]

        return mean, logvar


# STATISTIC NETWORK q(c|D)
class StatisticNetwork(nn.Module):
    """

    """

    def __init__(self, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, nonlinearity):
        super(StatisticNetwork, self).__init__()
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.nonlinearity = nonlinearity

        # modules
        self.prepool = PrePool(self.n_features, self.n_hidden,
                               self.hidden_dim, self.nonlinearity)
        self.postpool = PostPool(self.n_hidden, self.hidden_dim,
                                 self.c_dim, self.nonlinearity)

    def forward(self, x):
        batch_size = x.shape[0]
        e = self.prepool(x)
        e = self.pool(e, batch_size)
        e = self.postpool(e)
        return e

    def pool(self, e, batch_size):
        e = e.view(batch_size, self.sample_size, self.hidden_dim)
        e = e.mean(1).view(batch_size, self.hidden_dim)
        return e


# INFERENCE NETWORK q(z|x, z, c)
class InferenceNetwork(nn.Module):
    """

    """

    def __init__(self, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(InferenceNetwork, self).__init__()
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_x = nn.Linear(self.n_features, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, x, z, c):
        # combine x, z, and c
        # embed x
        batch_size = x.shape[0]
        ex = x.view(-1, self.n_features)
        ex = self.fc_x(ex)
        ex = ex.view(batch_size, self.sample_size, self.hidden_dim)

        # embed z if we have more than one stochastic layer
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(ex.size()).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(batch_size, 1, self.hidden_dim).expand_as(ex)

        # sum and reshape
        e = ex + ez + ec
        e = e.view(batch_size * self.sample_size, self.hidden_dim)
        e = self.nonlinearity(e)

        # residual blocks
        e = self.fc_block1(e)
        e = self.fc_block2(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# LATENT DECODER p(z|z, c)
class LatentDecoder(nn.Module):
    """

    """

    def __init__(self, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, z_dim, nonlinearity):
        super(LatentDecoder, self).__init__()
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)
        self.fc_z = nn.Linear(self.z_dim, self.hidden_dim)

        self.fc_block1 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)
        self.fc_block2 = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                    nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.z_dim)
        self.bn_params = nn.BatchNorm1d(1, eps=1e-3, momentum=1e-2)

    def forward(self, z, c):
        # combine z and c
        # embed z if we have more than one stochastic layer
        batch_size = c.shape[0]
        if z is not None:
            ez = z.view(-1, self.z_dim)
            ez = self.fc_z(ez)
            ez = ez.view(batch_size, self.sample_size, self.hidden_dim)
        else:
            ez = Variable(torch.zeros(batch_size, 1, self.hidden_dim).cuda())

        # embed c and expand for broadcast addition
        ec = self.fc_c(c)
        ec = ec.view(batch_size, 1, self.hidden_dim).expand_as(ez)

        # sum and reshape
        e = ez + ec
        e = e.view(-1, self.hidden_dim)
        e = self.nonlinearity(e)

        # residual blocks
        e = self.fc_block1(e)
        e = self.fc_block2(e)

        # affine transformation to parameters
        e = self.fc_params(e)

        # 'global' batch norm
        e = e.view(-1, 1, 2 * self.z_dim)
        e = self.bn_params(e)
        e = e.view(-1, 2 * self.z_dim)

        mean, logvar = e[:, :self.z_dim].contiguous(), e[:, self.z_dim:].contiguous()

        return mean, logvar


# Observation Decoder p(x|z, c)
class ObservationDecoder(nn.Module):
    """

    """

    def __init__(self, sample_size, n_features,
                 n_hidden, hidden_dim, c_dim, n_stochastic, z_dim,
                 nonlinearity):
        super(ObservationDecoder, self).__init__()
        self.sample_size = sample_size
        self.n_features = n_features

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.c_dim = c_dim

        self.n_stochastic = n_stochastic
        self.z_dim = z_dim

        self.nonlinearity = nonlinearity

        # modules
        self.fc_zs = nn.Linear(self.n_stochastic * self.z_dim, self.hidden_dim)
        self.fc_c = nn.Linear(self.c_dim, self.hidden_dim)

        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)

        self.fc_params = nn.Linear(self.hidden_dim, 2 * self.n_features)

    def forward(self, zs, c):
        batch_size = c.shape[0]
        ezs = self.fc_zs(zs)
        ezs = ezs.view(batch_size, self.sample_size, self.hidden_dim)

        ec = self.fc_c(c)
        ec = ec.view(batch_size, 1, self.hidden_dim).expand_as(ezs)

        e = ezs + ec
        e = self.nonlinearity(e)
        e = e.view(-1, self.hidden_dim)

        e = self.fc_block(e)

        e = self.fc_params(e)

        mean, logvar = e[:, :self.n_features], e[:, self.n_features:]

        return mean, logvar


# Regression network p(y|c), probably p(y|c, z)
class RegressionNetwork(nn.Module):
    def __init__(self, n_features, num_classes,
                 n_hidden, hidden_dim, nonlinearity):
        super(RegressionNetwork, self).__init__()
        self.n_features = n_features
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.nonlinearity = nonlinearity

        # modules
        self.fc_c = nn.Linear(self.n_features, self.hidden_dim)

        self.fc_block = FCResBlock(width=self.hidden_dim, n=self.n_hidden - 1,
                                   nonlinearity=self.nonlinearity)

        self.fc_out = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_c(x)
        x = self.fc_block(x)
        x = self.fc_out(x)
        return x



# ************************************************* Essential Blocks ************************************************* #
