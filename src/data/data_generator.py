"""Abstract class for data generators."""

from abc import abstractmethod, ABC
from tqdm import tqdm

class DataGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> dict:
        """
        dict: {'df': df, 'tau': treat_effect, 'n_samples': n_samples}
        """
        pass

    def generate_all(self, n_datasets: int = 10000) -> list:
        """Generate multiple datasets.

        Args:
            n_datasets: number of datasets to generate

        Returns:
            list of datasets
        """
        datasets = []
        for i in tqdm(range(n_datasets), desc="Generating data"):
            dataset = self.generate()
            datasets.append(dataset)

        return datasets
