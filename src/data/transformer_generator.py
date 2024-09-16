# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
# %%
import src.data as data_generators
from ..data.data_generator import DataGenerator
from typing import Optional

import re
from glob import glob
# %%
DELIMITER = -999

class TransformerDataGenerator(DataGenerator):
    """
    Wraps another DataGenerator, and reformats each output dataset as a sequence of -999 delimited input examples.

    Requires the generation dict in data_cfg for generating Transformer input data to contain a base_generation dict itself for the base generator

    generation = dict(generator='TransformerDataGenerator',
                      base_generation = dict(generator='LinearNormalDataGenerator,
                                             n_samples_range=[1000, 1000],  # range of sample size
                                             iv_strength_range=[1.0, 2.0],  # range of instrument strength
                                             conf_strength_range=[0.0, 1.0],  # range of confounder strength
                                             treat_effect_range=[-5.0, 5.0],  # range of treatment effect
                                             conf_effect_range=[0.0, 1.0],  # range of confounder effect
                                             beta_range=[-1.0, 1.0],  # range of beta (constant bias term)
                                             base_seed=seed  # random seed
                                             ),
                       window_size = None,
                    ) 
    """
    def __init__(self, base_generation: dict = None, data_path: str = None, window_size: Optional[int] = None, stage: Optional[str] = None):
        """
        base_generation: dict, identical to `generation` attribute in data_cfg when using any other DataGenerator. used to initialize DataGenerator that this class wraps
        data_path: path to data to be transformed to a transformer-ready format

        window_size: int. a window size of k indicates that k data points one data point = (Z T C Y, tau delim) are input to the transformer as a single sequence.
        stage: specifies purpose dataset type. one of {'train', 'test'}, only needed for online transformation (loading data from hf and transforming it)
        """
        if base_generation is None and data_path is None:
            raise ValueError("must generate data on fly (specify base_generation) or transform existing data (specify data_path), neither specified")
        elif base_generation is not None and data_path is None: # use a base generator to generate data
            self.mode = 'online_generation'
            base_generator_str = base_generation.pop("generator") # specify base generator class to wrap
            self.base_generator = getattr(data_generators, base_generator_str)(**base_generation) # initialize base generator
        elif base_generation is None and data_path is not None: # transform existing data
            self.mode = 'online_transformation'
            self.data_path = data_path
        else:
            raise ValueError("must generate data on fly (specify base_generation) or transform existing data (specify data_path), not both")
        
        self.stage = stage
        self.window_size = window_size
    
    def generate(self):
        """
        Uses generate() method of base_generator to generate data. 
        """
        base_dict = self.base_generator.generate()
        return base_dict # {'df': df, 'treat_effect':tau, ..?}


    def generate_all(self, n_datasets: Optional[int] = 10000) -> list:
        """
        Either
        - generates n_datasets worth of vanilla data from scratch and converts it into transformer-ready format (online generation), or
        - loads data from self.data_path and transforms it to transformer-ready format (offline generation)
        self.mode determines this behavior (see init)

        (original functionality of generate_all was just to generate n_datasets worth of vanilla dataset in transformer format by wrapping generate function, which generates data from scratch. but extended to include loading and transforming data since generate_all used across project to 'create entire dataset' - sorry for bad code design!)
        """
        datasets = []
        if self.mode == 'online_generation':
            for i in tqdm(range(n_datasets), desc="Generating data"):
                base_dict = self.generate()
                base_df = base_dict['df']
                treat_effect = base_dict['treatment_effect']
                dataset_list = self.transform_to_transformer_ready(base_df, treat_effect)
                datasets.extend(dataset_list)
        elif self.mode == 'online_transformation':
            assert self.stage is not None, "stage must be specified for online transformation"
            parquets = self.load_parquets_to_df(self.stage)
            for i in tqdm(range(n_datasets), desc="Loading and transforming data"):
                base_df, treat_effect = parquets[i]
                # transform
                dataset_list = self.transform_to_transformer_ready(base_df, treat_effect)
                datasets.extend(dataset_list)
        return datasets


    def transform_to_transformer_ready(self, base_df, treat_effect) -> list:
        """
        Given a dataframe and a treatment effect corresponding to a single vanilla dataset, convert this to a transformer ready format. When a context window is used), each contiguous window is returned as its own mini dataset (data point).

        Returns a list of dictionaries of form

        {'df': df, 'treatment_effect': treat_effect, 'n_samples': n_samples}

        of length 1 if window_size is None, and n // window_size otherwise, where n = len(base_df). By default drops any remaining last points that don't fill a window.
        """
        dict_list = []
        if self.window_size is None:
            # if window size not specified, set window to be entire dataset
            self.window_size = len(base_df)
            n_samples = len(base_df)
        else:
            n_samples = len(base_df) // self.window_size
        
        if self.window_size != 1:
            # add tau and delimiter columns to base
            base_df['tau'] = treat_effect
            base_df['delimiter'] = DELIMITER
            # group dataframe into groups of 'window_size' consecutive rows
            grouped_by_window = base_df.groupby(np.arange(len(base_df)) // self.window_size)
            # merge each group into single list, returning a pandas Series of lists
            flattened_groups = grouped_by_window.apply(lambda x: x.values.ravel())
            # default behavior: drop remaining columns that don't fill a window
            if len(base_df) % self.window_size != 0:
                flattened_groups = flattened_groups[:-1]
            # create new column names
            new_columns = []
            for i in range(self.window_size):
                new_columns.extend([f'{prefix}_{i}' for prefix in base_df.columns])
            cols_todrop = [f'tau_{self.window_size-1}', f'delimiter_{self.window_size-1}']
            # create new dataframe
            reshaped_df = pd.DataFrame(flattened_groups.tolist(), columns=new_columns)
            reshaped_df.drop(columns=cols_todrop, inplace=True)
        else: # for window_size = 1, no modifications needed
            reshaped_df = base_df # use base_df, without tau or delimiters, as context has just one data entry

        reshaped_df = reshaped_df.T # transpose so we can extract columns as single data points; this allows us to batch data points to give batches of shape (bs, seq_len, data_dim=1) as required for Transformer input
        reshaped_df.columns = reshaped_df.columns.astype(str) # parquet requires columns to be strings
        dict_list = [
            {'df': reshaped_df[col].to_frame(), 'treatment_effect': treat_effect, 'n_samples':n_samples}
            for col in reshaped_df.columns
                    ]
        
        return dict_list  
    
    def load_parquets_to_df(self, stage: str) -> pd.DataFrame:
        """
        Copy of function with same name from TabularDataModule, but for use with HF self.data_path
        Load all parquets under a directory and merge them into a dataframe

        Args:
            stage: stage to setup
        Returns
            dataframe of all parquets
        """
        # TODO: add functionality to go through multiple 'train' directories eg train0, train1,...
        parquets = []
        data_dir = f"{self.data_path}/{self.stage}"
        for file_name in tqdm(glob(f"{data_dir}/*.parquet"), desc=f"loading {self.stage} parquets"):
            #print(file_name)
            #print(file_name.split("/")[-1].split("-")[1])
            # TODO temp fix, we need a more robust way to do this
            m = re.match(".*treatment_effect=(.*)-n.*", file_name)
            treatment_effect = float(
                m.groups()[0]
                #file_name.split("/")[-1].split("-")[1].strip("treatment_effect=")
            )
            # n_samples = file_name.split('/')[-1].split('-')[2].strip('n_samples=')
            df = pd.read_parquet(file_name)
            parquets.append((df, treatment_effect))
        return parquets

# %%
