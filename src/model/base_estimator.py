"""
Abstract class for existing IV estimators.
"""
import pandas as pd

from abc import abstractmethod, ABC
from tqdm import tqdm

class BaseEstimator(ABC):
    @abstractmethod
    def estimate(self, T, X, Z, Y, **kwargs) -> dict:
        """
        Estimate the treatment effect.

        TODO what other properties should be returned?
        Returns:
            dict: {'tau': treat_effect, 'se': standard_error}
        """
        pass

    def estimate_all(self, datasets: list, **kwargs) -> list:
        """Estimate the treatment effect for multiple datasets.

        Args:
            n_datasets: number of datasets to estimate

        Returns:
            pd.DataFrame: results of the estimation
        """        
        results = []
        for idx, (dataset, ground_truth_tau) in enumerate(tqdm(datasets, desc="tau estimation")):
            # pull X data, if the columns exist
            X_data = dataset.loc[:, dataset.columns.str.startswith('X')]
            Z_data = dataset.loc[:, dataset.columns.str.startswith('Z')]
            result = self.estimate(T=dataset['T'], X=X_data, Z=Z_data, Y=dataset['Y'], **kwargs)
            result['ground_truth'] = ground_truth_tau
            result['idx'] = idx
            results.append(result)

        return pd.DataFrame(results)