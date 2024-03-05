import numpy as np
import pandas as pd
from tqdm import tqdm


class LinearNormalDataGenerator:
    """
    Data generator for generating linear multivariate normal data according to classical IV graph (Z, T, C, Y)
    """
    def __init__(self, n_samples_range: list = [1000, 10000], iv_strength_range: list = [0.1, 1.0],
                       conf_strength_range: list = [0.1, 1.0], treat_effect_range: list = [-1.0, 1.0],
                       conf_effect_range: list = [1.0, 5.0], beta_range: list = [-1.0, 1.0],
                       base_seed: int = 42) -> None:
        """
        Initialize the data generator

        Args:
            n_samples_range: range of sample size
            iv_strength_range: range of instrument strength
            conf_strength_range: range of confounder strength
            treat_effect_range: range of treatment effect
            conf_effect_range: range of confounder effect
            beta_range: range of beta (bias term)
            base_seed: random seed
        """
        self.n_samples_range = n_samples_range
        self.iv_strength_range = iv_strength_range
        self.conf_strength_range = conf_strength_range
        self.treat_effect_range = treat_effect_range
        self.conf_effect_range = conf_effect_range
        self.beta_range = beta_range
        self.base_seed = base_seed

    def generate(self) -> dict:
        """
        Sample parameters from given range and generate linear multivariate normal data following the formula:
        Y = beta + (tau * T) + (gamma * C) + epsilon

        Returns:
            dict: {'df': df, 'tau': treat_effect, 'n_samples': n_samples}
        """
        self.base_seed += 1
        np.random.seed(self.base_seed)
        n_samples = int(np.random.uniform(self.n_samples_range[0], self.n_samples_range[1]))
        z = np.random.normal(0, 1, size=n_samples)
        c = np.random.normal(0, 1, size=n_samples)
        iv_strength = np.random.uniform(self.iv_strength_range[0], self.iv_strength_range[1])
        conf_strength = np.random.uniform(self.conf_strength_range[0], self.conf_strength_range[1])
        conf_effect = np.random.uniform(self.conf_effect_range[0], self.conf_effect_range[1])
        treat_effect = np.random.uniform(self.treat_effect_range[0], self.treat_effect_range[1])

        # Generate treatment
        beta = np.random.uniform(self.beta_range[0], self.beta_range[1])
        t = beta + iv_strength * z + conf_strength * c + np.random.normal(0, 1, size=n_samples)

        # Generate outcome
        beta = np.random.uniform(self.beta_range[0], self.beta_range[1])
        y = beta + conf_effect * c + treat_effect * t + np.random.normal(0, 1, size=n_samples)

        data = np.concatenate([z.reshape(-1, 1), c.reshape(-1, 1), t.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
        df = pd.DataFrame(data, columns=['instrument', 'confounder', 'treatment', 'outcome'])

        return {'df': df, 'treatment_effect': treat_effect, 'n_samples': n_samples}

    def generate_all(self, n_datasets: int = 10000) -> list:
        """
        Generate multiple datasets

        Args:
            n_datasets: number of datasets to generate

        Returns:
            datasets: list of datasets
        """
        datasets = []
        for i in tqdm(range(n_datasets), desc="Generating data"):
            dataset = self.generate()
            datasets.append(dataset)

        return datasets
