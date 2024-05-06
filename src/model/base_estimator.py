"Abstract class for existing IV estimators."

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

    @abstractmethod
    def estimate_all(self, datasets: list) -> list:
        """Estimate the treatment effect for multiple datasets.

        Args:
            n_datasets: number of datasets to estimate

        Returns:
            list of estimates
        """        
        results = []
        for dataset in tqdm(datasets, desc="Tau estimation"):
            results.append(self.estimate(**dataset))

        return results