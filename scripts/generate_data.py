import joblib
import random
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.data as data_generators
from src.utils import Config


def save_to_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Save df to parquet file

    Args:
        df: dataframe to save
        file_path: path to save
    """
    df.to_parquet(file_path, index=False)


def datasets_split_and_save(datasets: list, work_dir: str, n_train: float = 0.6, n_val: float = 0.2,
                            n_test: float = 0.2,) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Merge a list of datasets into three df: data train, val, and test sets, and save them to csv files

    Args:
        datasets: list of datasets
        work_dir: directory to save
        n_train: proportion of train set
        n_val: proportion of val set
        n_test: proportion of test set
    Returns:
        train, val, test sets
    """
    assert n_train + n_val + n_test == 1, "n_train + n_val + n_test should be 1"
    # Split datasets
    random.shuffle(datasets)
    trainset = datasets[:int(n_train * len(datasets))]
    valset = datasets[int(n_train * len(datasets)):int((n_train + n_val) * len(datasets))]
    testset = datasets[int((n_train + n_val) * len(datasets)):]

    # Save datasets to parquet files
    _ = joblib.Parallel(n_jobs=8)(joblib.delayed(save_to_parquet)(df['df'], os.path.join(work_dir, f"train/uid={i}-"
                                                                                                   f"treatment_effect={df['treatment_effect']}-"
                                                                                                   f"n_samples={df['n_samples']}"
                                                                                                   f".parquet"))
                                                                for i, df in tqdm(enumerate(trainset),
                                                                                    desc="Saving trainset...",
                                                                                    total=len(trainset)))
    _ = joblib.Parallel(n_jobs=8)(joblib.delayed(save_to_parquet)(df['df'], os.path.join(work_dir, f"val/uid={i}-"
                                                                                             f"treatment_effect={df['treatment_effect']}-"
                                                                                             f"n_samples={df['n_samples']}"
                                                                                             f".parquet"))
                                                                for i, df in tqdm(enumerate(valset),
                                                                                    desc="Saving valset...",
                                                                                    total=len(valset)))
    _ = joblib.Parallel(n_jobs=8)(joblib.delayed(save_to_parquet)(df['df'], os.path.join(work_dir, f"test/uid={i}-"
                                                                                             f"treatment_effect={df['treatment_effect']}-"
                                                                                             f"n_samples={df['n_samples']}"
                                                                                             f".parquet"))
                                                                for i, df in tqdm(enumerate(testset),
                                                                                    desc="Saving testset...",
                                                                                    total=len(testset)))


def generate_data_from_config(cfg_path: str, work_dir: str = '') -> None:
    """
    Generate data from config file (.py)

    Args:
        cfg_path: path to config file
        work_dir: path to save the data
    """
    cfg = Config.fromfile(cfg_path)
    if not work_dir:
        work_dir = cfg.work_dir
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(os.path.join(work_dir, 'train')):
        os.makedirs(os.path.join(work_dir, 'train'))
        os.makedirs(os.path.join(work_dir, 'val'))
        os.makedirs(os.path.join(work_dir, 'test'))

    # Initialize data generator
    generator = cfg.generation.pop('generator')
    if hasattr(data_generators, generator):
        generator = getattr(data_generators, generator)(**cfg.generation)
    else:
        raise NotImplementedError(f"Generator {generator} is not implemented")

    # Data generation
    datasets = generator.generate_all()
    datasets_split_and_save(datasets, work_dir, cfg.n_train, cfg.n_val, cfg.n_test)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../datasets/linear/linear_norm.py')
    parser.add_argument('--work_dir', type=str, default='')
    args = parser.parse_args()

    cfg = args.cfg
    work_dir = args.work_dir
    generate_data_from_config(cfg, work_dir)
