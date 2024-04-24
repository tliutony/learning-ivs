"""
Network that uses sufficient statistics as input to a multi-layer perceptron.
Specifically, the mean, variance, and covariance of the input data are used as input to the MLP.
"""

import torch
import torch.nn as nn
from .mlp import LinearBlock


class SufStatsBlock(nn.Module):
    """Calculate sufficient statistics of input data.

    Expects input data in the shape (batch, n_samples, channels),
    and outputs the mean, variance, and covariance of the input data in
    the shape (batch, suff_stats).
    """
