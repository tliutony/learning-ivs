"""Defines the LennonIVGenerator class."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from ..data.data_generator import DataGenerator


def generate_cov_matrix(n_vars, base=0.5) -> torch.Tensor:
    """Generates the covariance matrix for the instruments per Lennon et al. 2022"""
    cov = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            cov[i, j] = base ** (np.abs(i - j))
    return torch.Tensor(cov)


class LennonIVGenerator(nn.Module, DataGenerator):
    """
    nn.Module to generate IV datasets according to Lennon et al. 2022

    TODO add ability to generate IV controls to cover the Angrist/Frandsen case
    """

    def __init__(
        self,
        max_vars: int,
        n_instruments: int,
        instrument_strength: float,  # this currently translates to mu^2/n_samples
        instrument_cov_base: float = 0.5,
        tau_activation: str = "identity",
        instrument_activation: str = "identity",
        tau_range: list = [-5.0, 5.0],
        n_samples_range: list = [1000, 10000],
        base_seed: int = 42,
        # TODO add controls as well
        control_activation: str = "identity",
        control_covariance: torch.Tensor = None,
        control_str: float = 0,
        n_controls: int = 0,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.n_samples_range = n_samples_range
        self.tau_range = tau_range
        self.base_seed = base_seed
        torch.manual_seed(base_seed)

        self.max_vars = max_vars
        # TODO are these actually needed?
        # self.controls = nn.Linear(max_vars, 1)
        self.instruments = nn.Linear(max_vars, 1)

        # Nself.n_controls = n_controls
        self.n_instruments = n_instruments

        if control_covariance is not None:
            assert control_covariance.shape == (n_controls, n_controls)

        instrument_covariance = generate_cov_matrix(n_instruments, instrument_cov_base).to(self.device)
        self.instrument_covariance = instrument_covariance
        self.instrument_sampler = MultivariateNormal(
            torch.zeros(self.n_instruments, device=self.device), self.instrument_covariance
        )

        # currently follows the Beta pattern of Lennon et al. 2022, corresponds to their pi
        self.instrument_coefficients = torch.pow(
            torch.ones(self.n_instruments, device=self.device) * 0.5,
            torch.arange(0, self.n_instruments, device=self.device),
        )

        # shuffle coefficients
        self.instrument_coefficients = self.instrument_coefficients[
            torch.randperm(self.n_instruments, device=self.device)
        ]
        # determine C coefficient
        A = (
            torch.t(self.instrument_coefficients)
            @ self.instrument_covariance
            @ self.instrument_coefficients
        )
        C = torch.sqrt(instrument_strength / (instrument_strength * A + A))
        # print(f"C: {C}")
        self.instrument_coefficients = C * self.instrument_coefficients
        self.sigma_v = torch.sqrt(
            1
            - (
                torch.t(self.instrument_coefficients)
                @ self.instrument_covariance
                @ self.instrument_coefficients
            )
        )
        # print(self.sigma_v)
        self.sigma_y = 1
        eps = 1e-6

        # Lennon et al. 2022/Belloni et al. 2012 don't appear to guarantee this...
        self.confound_covariance = torch.tensor(
            [
                [self.sigma_y**2 + eps,       self.sigma_y * self.sigma_v],
                [self.sigma_y * self.sigma_v, self.sigma_v**2 + eps],  # current hack to get things positive definite
            ]
        , device=self.device)
        # print(self.confound_covariance)
        self.confound_sampler = MultivariateNormal(
            torch.zeros(2, device=self.device), self.confound_covariance
        )

        self.control_str = control_str

        self.activations = {
            "identity": lambda x: x,
            "relu": F.relu,
            "sigmoid": F.sigmoid,
            "tanh": F.tanh,
            "softplus": F.softplus,
            "leaky_relu": F.leaky_relu,
            "elu": F.elu,
        }

        self.tau_activation = self.activations[tau_activation]
        self.instrument_activation = self.activations[instrument_activation]
        # self.confounder_activation = self.activations[confounder_activation]

    def forward(self, tau: float):
        """Generates a single data sample"""

        # noise sample [\epislon_y, \epeilson_v] according to confounder covariance
        instrument_sample = torch.cat(
            [
                self.instrument_sampler.sample(),
                torch.zeros(self.max_vars - self.n_instruments),
            ]
        )

        epislon_y, epsilon_v = self.confound_sampler.sample()
        pi = torch.cat(
            [
                self.instrument_coefficients,
                torch.zeros(self.max_vars - self.n_instruments),
            ]
        )
        treat = self.instrument_activation(torch.t(pi) @ instrument_sample) + epsilon_v

        outcome = tau * self.tau_activation(treat) + torch.randn(1) + epislon_y

        # return data matrix of T, Y, Z
        return torch.cat([torch.Tensor([treat, outcome]), instrument_sample])

    def batch(self, tau: float, batch_size: int) -> torch.Tensor:
        """Generate a batch of samples using vectorized sampling and matrix ops.

        Returns a tensor of shape (batch_size, 2 + n_instruments): columns are [T, Y, Z0..Z{n_instruments-1}].
        """
        # Sample instruments: (batch_size, n_instruments)
        Z = self.instrument_sampler.sample((batch_size,))

        # Build full instrument matrix to size max_vars by padding zeros if needed
        if self.max_vars > self.n_instruments:
            pad = torch.zeros(batch_size, self.max_vars - self.n_instruments, device=Z.device, dtype=Z.dtype)
            instrument_matrix = torch.cat([Z, pad], dim=1)  # (batch_size, max_vars)
        else:
            instrument_matrix = Z  # (batch_size, n_instruments == max_vars)

        # pi coefficients to length max_vars
        pi = torch.cat([
            self.instrument_coefficients,
            torch.zeros(self.max_vars - self.n_instruments, device=Z.device, dtype=Z.dtype),
        ])  # (max_vars,)

        # Confounder noise: (batch_size, 2) -> eps_y, eps_v
        eps = self.confound_sampler.sample((batch_size,))
        epsilon_y = eps[:, 0]
        epsilon_v = eps[:, 1]

        # Treatment and outcome
        t_no_noise = instrument_matrix @ pi  # (batch_size,)
        treat = self.instrument_activation(t_no_noise) + epsilon_v  # (batch_size,)
        outcome = tau * self.tau_activation(treat) + torch.randn(batch_size, device=treat.device) + epsilon_y  # (batch_size,)

        # Return packed tensor [T, Y, Z0..]
        data = torch.cat([treat.unsqueeze(1), outcome.unsqueeze(1), Z], dim=1)
        return data

    def generate(self) -> dict:
        """
        Generate a dataset of n_samples
        """
        self.base_seed += 1
        np.random.seed(self.base_seed)
        n_samples = int(
            np.random.uniform(self.n_samples_range[0], self.n_samples_range[1])
        )
        tau = np.random.uniform(self.tau_range[0], self.tau_range[1])
        data = self.batch(tau, n_samples)

        return {
            "df": pd.DataFrame(
                data.detach().cpu().numpy(),
                columns=["T", "Y"] + [f"Z{i}" for i in range(self.n_instruments)],
            ),
            "treatment_effect": tau,
            "n_samples": n_samples,
        }
