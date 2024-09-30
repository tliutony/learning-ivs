import torch
import re
import pandas as pd
import numpy as np
from glob import glob
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

import src.data as data_generators  # lin_norm_generator as generators
from ..utils import Config


class TabularDataModule(pl.LightningDataModule):
    """
    DataModule for loading existed CSV data or to generate data according to some online generation scheme
    """

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        data_cfg = None,
        use_huggingface: bool = False,
        use_sequence: bool = False,  # New parameter to control sequence usage
        sequence_length: int = 1000,
        lazy_loading: bool = True,  # New parameter to control lazy loading
    ) -> None:
        """
        Initialize the csv data module

        Args:
        data_dir: data_dir to csv files, if None, will generate data online according to data_cfg
        data_cfg: directory to file specifying configuration details specific to data generation
        train_batch_size: training batch size
        val_batch_size: validation batch size
        test_batch_size: test batch size
        use_huggingface: whether to use Hugging Face datasets
        lazy_loading: whether to use lazy loading for datasets
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.data_cfg = data_cfg
        self.use_huggingface = use_huggingface
        self.use_sequence = use_sequence
        self.sequence_length = sequence_length if use_sequence else None
        self.lazy_loading = lazy_loading

        if data_dir is None and data_cfg is None:
            raise ValueError(
                f"data_cfg is not specified when online generation is True"
            )
        elif data_dir is None and data_cfg is not None:
            # load data config and setup online generation
            self.online_generation = True
<<<<<<< HEAD
            if isinstance(data_cfg, str):
                data_cfg = Config.fromfile(data_cfg)
            elif isinstance(data_cfg, Config):
                pass # continue as is
            else:
                raise TypeError(f"data_cfg should be either path to a data generation config file, or a Config object, received {type(data_cfg)}")

=======
>>>>>>> main
            generator = data_cfg.generation.pop("generator")
            self.online_generator = getattr(data_generators, generator)(
                **data_cfg.generation
            )
            print(self.online_generator)
        else:
            # only use offline data if data_cfg is not specified
            self.online_generation = False

    def setup(self, stage: [str, None] = None) -> None:
        """
        Load all csv files under data_dir and recognize the train/val/test split

        Args:
            stage: stage to setup
        """
        if self.use_huggingface:
            dataset = load_from_disk(self.data_dir)
            if stage == "fit" or stage is None:
                self.trainset = self.load_huggingface_data(dataset['train'], 'train')
                self.valset = self.load_huggingface_data(dataset['validation'], 'val')
            if stage == "test" or stage is None:
                self.testset = self.load_huggingface_data(dataset['test'], 'test')
        else:
            if stage == "fit" or stage is None:
                self.trainset = self.load_data("train")
                self.valset = self.load_data("val")
            if stage == "test" or stage is None:
                self.testset = self.load_data("test")

    def load_huggingface_data(self, dataset, stage) -> list:
        """
        Load data from Hugging Face dataset and divide it into specified number of datasets

        Args:
            dataset: Hugging Face dataset
            stage: current stage (train, val, or test)
        Returns:
            list of tuples (df, treatment_effect)
        """
        data = []
        
        dataset = dataset.to_pandas()
        
        # Group the dataset by Y values
        grouped = dataset.groupby('Y')
        print(dataset['Y'].unique())
        
        n_datasets = int(self.data_cfg.n_datasets * getattr(self.data_cfg, f"n_{stage}"))
        
        for _, group in tqdm(grouped, desc="loading huggingface data"):
            # If we have more groups than desired datasets, we might need to combine some
            if len(data) >= n_datasets:
                break
            
            treatment_effect = group['Y'].iloc[0]
            # Ensure we have at least sequence_length samples
            if len(group) >= self.sequence_length:
                data.append((group, treatment_effect))
        # If we have fewer groups than desired datasets, we might need to split some
        while len(data) < n_datasets:
            largest_group_idx = max(range(len(data)), key=lambda i: len(data[i][0]))
            largest_group, effect = data.pop(largest_group_idx)
            
            mid = len(largest_group) // 2
            if mid >= self.sequence_length:
                data.append((largest_group.iloc[:mid], effect))
                data.append((largest_group.iloc[mid:], effect))
            else:
                data.append((largest_group, effect))
        
        return data

    def load_data(self, stage: str) -> pd.DataFrame:
        """
        Prepare the data online or offline

        Args:
            stage: stage to setup
        Returns:
            dataframe of all datasets
        """
        if self.online_generation: # true online generation or online transformaton
            n_datasets = int(self.data_cfg.get(f"n_{stage}") * self.data_cfg.n_datasets)
            # generate/transform data online, add to parquets
            datasets = self.online_generator.generate_all(n_datasets)
            parquets = []
            for dataset in datasets:
                df = dataset["df"]
                parquets.append((df, dataset["treatment_effect"]))
        else:
            
            parquets = self.load_parquets_to_df(stage)

        return parquets

    def load_parquets_to_df(self, stage: str) -> pd.DataFrame:
        """
        Load all parquets under a directory and merge them into a dataframe

        Args:
            stage: stage to setup
        Returns
            dataframe of all parquets
        """
        parquets = []
        data_dir = f"{self.data_dir}/{stage}"
        for file_name in tqdm(glob(f"{data_dir}/*.parquet"), desc=f"loading {stage} parquets"):
            m = re.match(".*treatment_effect=(.*)-n.*", file_name)
            treatment_effect = float(m.groups()[0])
            if self.lazy_loading:
                parquets.append((file_name, treatment_effect))
            else:
                df = pd.read_parquet(file_name)
                parquets.append((df, treatment_effect))
        return parquets

    def train_dataloader(self) -> DataLoader:
        """
        Get train dataloader

        Returns:
            train dataloader
        """
        trainset = DataFrameDataset(self.trainset, self.use_sequence, self.sequence_length, self.lazy_loading)
        return DataLoader(
            trainset,
            batch_size=self.train_batch_size,
            num_workers=1, # was 4
            pin_memory=False, # was True
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get validation dataloader

        Returns:
            validation dataloader
        """
        valset = DataFrameDataset(self.valset, self.use_sequence, self.sequence_length, self.lazy_loading)
        return DataLoader(
            valset,
            batch_size=self.val_batch_size,
            num_workers=1, # was 4
            pin_memory=False, # was True
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Get test dataloader

        Returns:
            test dataloader
        """
        testset = DataFrameDataset(self.testset, self.use_sequence, self.sequence_length, self.lazy_loading)
        return DataLoader(
            testset,
            batch_size=self.test_batch_size,
            num_workers=1, # was 4
            pin_memory=False, # was True
            shuffle=False,
        )



class DataFrameDataset(Dataset):
    """
    Dataset for dataframe data
    """

    def __init__(self, dataset: list[tuple], use_sequence: bool, sequence_length: int = None, lazy_loading: bool = False) -> None:
        """
        Initialize the dataframe dataset

        Args:
            dataset: list of tuple consisted of df and corresponding treatment effect
            sequence_length: number of samples to group together for each sequence
            lazy_loading: whether to use lazy loading for datasets
        """
        super().__init__()
        self.dataset = dataset
        self.use_sequence = use_sequence
        self.sequence_length = sequence_length
        self.lazy_loading = lazy_loading

    def __getitem__(self, idx: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the item at idx

        Args:
            idx: index of the item

        Returns:
            x, y
        """
        if self.lazy_loading:
            file_name, y = self.dataset[idx]
            df = pd.read_parquet(file_name)
        else:
            df, y = self.dataset[idx]
        
        if self.use_sequence:
            if len(df) < self.sequence_length:
                repeats = self.sequence_length // len(df) + 1
                df = df.iloc[np.tile(np.arange(len(df)), repeats)[:self.sequence_length]]
            elif len(df) > self.sequence_length:
                df = df.sample(n=self.sequence_length, replace=False)
        
        x = torch.tensor(df.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def __len__(self) -> int:
        """
        Get the length of the dataset

        Returns:
            length of the dataset
        """
        return len(self.dataset)
