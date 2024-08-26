# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
# %%
import src.data as data_generators
from ..data.data_generator import DataGenerator
from typing import Optional

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
    def __init__(self, base_generation, window_size: Optional[int] = None):
        """
        base_generation: dict, identical to `generation` attribute in data_cfg when using any other DataGenerator. used to initialize DataGenerator that this class wraps
        window_size: int, specifies length of window/sequence input to transformer, in terms of number of training examples per sequence
        """
        base_generator_str = base_generation.pop("generator") # specify base generator class to wrap
        self.base_generator = getattr(data_generators, base_generator_str)(**base_generation) # initialize base generator
        self.window_size = window_size
    
    def generate(self):
        """
        Uses generate() method of base_generator to generate data. When a context window is used), each context window section is returned as its own mini dataset (data point).

        Returns a list of dictionaries of form

        {'df': df, 'treatment_effect': treat_effect, 'n_samples': n_samples}

        of length 1 if window_size is None, and n // window_size otherwise, where n is the number of data points returned by base generator's generate function. By default drops any remaining last points that don't fill a window.
        """
        base_dict = self.base_generator.generate()
        base_dict_df = base_dict['df']
        treat_effect = base_dict['treatment_effect']

        base_dict_df['tau'] = treat_effect
        base_dict_df['delimiter'] = DELIMITER

        dict_list = []
        if self.window_size is None:
            # if window size not specified, set window to be entire dataset
            self.window_size = len(base_dict_df)
            n_samples = len(base_dict_df)
        else:
            n_samples = len(base_dict_df) // self.window_size
        
        # group dataframe into groups of 'window_size' consecutive rows
        grouped_by_window = base_dict_df.groupby(np.arange(len(base_dict_df)) // self.window_size)
        # merge each group into single list, returning a pandas Series of lists
        flattened_groups = grouped_by_window.apply(lambda x: x.values.ravel())
        # default behavior: drop remaining columns that don't fill a window
        if len(base_dict_df) % self.window_size != 0:
            flattened_groups = flattened_groups[:-1]
        # create new column names
        new_columns = []
        for i in range(self.window_size):
            new_columns.extend([f'{prefix}_{i}' for prefix in base_dict_df.columns])
        cols_todrop = [f'tau_{self.window_size-1}', f'delimiter_{self.window_size-1}']
        # create new dataframe
        reshaped_df = pd.DataFrame(flattened_groups.tolist(), columns=new_columns)
        reshaped_df.drop(columns=cols_todrop, inplace=True)

        reshaped_df = reshaped_df.T # transpose so we can extract columns as single data points; this allows us to batch data points to give batches of shape (bs, seq_len, data_dim=1) as required for Transformer input
        reshaped_df.columns = reshaped_df.columns.astype(str) # parquet requires columns to be strings
        dict_list = [
            {'df': reshaped_df[[col]], 'treatment_effect': treat_effect, 'n_samples':n_samples}
            for col in reshaped_df.columns
                    ]
        
        return dict_list


    def generate_all(self, n_datasets: int = 10000) -> list:
        datasets = []
        for i in tqdm(range(n_datasets), desc="Generating data"):
            dataset_list = self.generate()
            datasets.extend(dataset_list)
        
        return datasets  
# %%
