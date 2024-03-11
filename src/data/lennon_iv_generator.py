"""Defines the LennonIVGenerator class."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data.data_generator import DataGenerator

def generate_cov_matrix(n_vars, base=0.5):
    """Generates the covariance matrix for the instruments per Lennon et al. 2022"""
    cov = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            cov[i, j] = base**(np.abs(i-j))
    return cov

class LennonIVGenerator(nn.Module, DataGenerator):
    """
    nn.Module to generate IV datasets according to Lennon et al. 2022

    TODO add ability to generate IV controls to cover the Angrist/Frandsen case
    """
    def __init__(self, 
                tau: float,
                max_vars: int,
                n_instruments: int,
                instrument_strength: float, # this currently translates to mu^2/n_samples
                instrument_cov_base: float = 0.5,
                tau_activation: str = 'identity',
                instrument_activation: str = 'identity',
                # TODO add controls as well
                control_activation: str = 'identity',
                control_covariance: torch.Tensor = None, 
                control_str: float = 0,
                n_controls: int = 0,
                ):
        super().__init__()
        self.max_vars = max_vars
        # TODO are these actually needed?
        #self.controls = nn.Linear(max_vars, 1)
        self.instruments = nn.Linear(max_vars, 1)

        #Nself.n_controls = n_controls
        self.n_instruments = n_instruments

        if control_covariance is not None:
            assert control_covariance.shape == (n_controls, n_controls)

        instrument_covariance = generate_cov_matrix(n_instruments, instrument_cov_base)
        self.instrument_covariance = instrument_covariance
        self.instrument_sampler = MultivariateNormal(torch.zeros(self.n_instruments), self.instrument_covariance)
        
        # currently follows the Beta pattern of Lennon et al. 2022, corresponds to their pi
        self.instrument_coefficients = torch.pow(torch.ones(self.n_instruments) * 0.5, 
                                         torch.arange(0, self.n_instruments))

        # shuffle coefficients
        self.instrument_coefficients = self.instrument_coefficients[torch.randperm(self.n_instruments)]
        # determine C coefficient
        A = torch.t(self.instrument_coefficients) @ self.instrument_covariance @ self.instrument_coefficients
        C = torch.sqrt(instrument_strength / (instrument_strength*A + A))
        print(f"C: {C}")
        self.instrument_coefficients = C * self.instrument_coefficients
        self.sigma_v = torch.sqrt(1 - (torch.t(self.instrument_coefficients) @ self.instrument_covariance @ self.instrument_coefficients))
        print(self.sigma_v)
        self.sigma_y = 1
        
        # TODO need to ensure that confounder covariance is positive definite, 
        # Lennon et al. 2022 doesn't appear to gaurantee this...
        self.confound_covariance = torch.Tensor([[self.sigma_y**2, self.sigma_y*self.sigma_v], 
                                                 [self.sigma_y*self.sigma_v, self.sigma_v**2]])
        print(self.confound_covariance)
        self.confound_sampler = MultivariateNormal(torch.zeros(2), self.confound_covariance)

        self.tau = tau
        self.control_str = control_str

        self.activations = {
            'identity': lambda x: x,
            'relu': F.relu,
            'sigmoid': F.sigmoid,
            'tanh': F.tanh,
            'softplus': F.softplus,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
        }
    
        self.tau_activation = self.activations[tau_activation]
        self.instrument_activation = self.activations[instrument_activation]
        #self.confounder_activation = self.activations[confounder_activation]


    def forward(self):
        """Generates a single data sample"""        

        # noise sample [\epislon_y, \epeilson_v] according to confounder covariance
        instrument_sample = torch.cat([self.instrument_sampler.sample(), torch.zeros(self.max_vars - self.n_instruments)])

        epislon_y, epsilon_v = self.confound_sampler.sample()
        pi = torch.cat([self.instrument_coefficients, torch.zeros(self.max_vars - self.n_instruments)])
        treat = self.instrument_activation(torch.t(pi) @ instrument_sample) + epsilon_v

        outcome = self.tau*self.tau_activation(treat) + torch.randn(1) + epislon_y

        # return data matrix of T, Y, Z
        return torch.cat([torch.Tensor([treat, outcome]), instrument_sample])
        
    def batch(self, batch_size: int):
        """Generate batch of examples"""
        return torch.stack([self.forward() for _ in range(batch_size)])
