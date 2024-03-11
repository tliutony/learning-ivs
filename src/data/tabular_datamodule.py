import torch
import pandas as pd
from glob import glob
from copy import deepcopy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import src.data as data_generators #lin_norm_generator as generators
from ..utils import Config


class TabularDataModule(pl.LightningDataModule):
    """
    DataModule for loading existed CSV data
    """

    def __init__(
        self,
        data_dir: [str, None],
        train_batch_size: int = 32,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        data_cfg: [str, None] = None,
    ) -> None:
        """
        Initialize the csv data module

        Args:
        data_dir: data_dir to csv files, if None, will generate data online
        train_batch_size: training batch size
        val_batch_size: validation batch size
        test_batch_size: test batch size
        """
        super().__init__()
        self.save_hyperparameters()
        if data_dir is None and data_cfg is None:
            raise ValueError(
                f"data_cfg is not specified when online generation is True"
            )
        elif data_dir is None and data_cfg is not None:
            # load data config and setup online generation
            self.online_generation = True
            data_cfg = Config.fromfile(data_cfg)
            generator = data_cfg.generation.pop("generator")
            self.online_generator = getattr(data_generators, generator)(
                **data_cfg.generation
            )
            print(self.online_generator)
        elif data_dir is not None and data_cfg is None:
            # only use offline data if data_cfg is not specified
            self.online_generation = False
        else:
            raise ValueError(
                f"Only offline or online generation can be specified, not both"
            )

        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.data_cfg = data_cfg

    def setup(self, stage: [str, None] = None) -> None:
        """
        Load all csv files under data_dir and recognize the train/val/test split

        Args:
            stage: stage to setup
        """
        if stage == "fit" or stage is None:
            self.trainset = self.load_data("train")
            self.valset = self.load_data("val")
        if stage == "test" or stage is None:
            self.testset = self.load_data("test")

    def load_data(self, stage: str) -> pd.DataFrame:
        """
        Prepare the data online or offline

        Args:
            stage: stage to setup
        Returns:
            dataframe of all datasets
        """
        if self.online_generation:
            n_datasets = int(self.data_cfg.get(f"n_{stage}") * self.data_cfg.n_datasets)
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
        for file_name in glob(f"{data_dir}/*.parquet"):
            treatment_effect = float(
                file_name.split("/")[-1].split("-")[1].strip("treatment_effect=")
            )
            # n_samples = file_name.split('/')[-1].split('-')[2].strip('n_samples=')
            df = pd.read_parquet(file_name)
            parquets.append((df, treatment_effect))
        return parquets

    def train_dataloader(self) -> DataLoader:
        """
        Get train dataloader

        Returns:
            train dataloader
        """
        trainset = DataFrameDataset(self.trainset)
        return DataLoader(
            trainset, batch_size=self.train_batch_size, pin_memory=True, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Get validation dataloader

        Returns:
            validation dataloader
        """
        valset = DataFrameDataset(self.valset)
        return DataLoader(
            valset, batch_size=self.val_batch_size, pin_memory=True, shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        """
        Get test dataloader

        Returns:
            test dataloader
        """
        testset = DataFrameDataset(self.testset)
        return DataLoader(
            testset,
            batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )


class DataFrameDataset(Dataset):
    """
    Dataset for dataframe data
    """

    def __init__(self, dataset: list[tuple]) -> None:
        """
        Initialize the dataframe dataset

        Args:
            dataset: list of tuple consisted of df and corresponding treatment effect
        """
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx: int) -> [torch.Tensor, torch.Tensor]:
        """
        Get the item at idx

        Args:
            idx: index of the item

        Returns:
            x, y
        """
        df, y = self.dataset[idx]
        # this makes the assumption that any metadata has been removed from df
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
