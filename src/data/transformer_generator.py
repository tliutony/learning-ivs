# %%
import pandas as pd
from tqdm import tqdm
# %%
import src.data as data_generators
from ..data.data_generator import DataGenerator

class TransformerDataGenerator(DataGenerator):
    """
    Wraps another DataGenerator, and reformats each output dataset as a sequence of -999 delimited input examples.

    Requires the generation dict in data_cfg to contain a base_generation dict itself for the base generator

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
    def __init__(self, base_generation, window_size):
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

        {'df': df, 'treatment_effect': treat_effect, 'n_samples': 1}

        of length 1 if window_size is None, and n // window_size otherwise, where n is the number of data points returned by base generator's generate function. By default drops any remaining last points that don't fill a window.
        """
        base_dict = self.base_generator.generate()
        base_dict_df = base_dict['df']
        treat_effect = base_dict['treat_effect']

        base_dict_df['tau'] = treat_effect
        base_dict_df['delimiter'] = -999

        # df = pd.DataFrame(base_dict_df.values.ravel()).T
        flattened_arr = base_dict_df.values.ravel()

        dict_list = []
        if self.window_size is not None:
            width = base_dict_df.shape[1]
            flattened_arr = flattened_arr[:-(len(flattened_arr) % (width * self.window_size))]
            windowed_arr = flattened_arr.reshape(-1, width * self.window_size) # divide into windows

            for row in windowed_arr:
                row_df = pd.DataFrame(row).T
                row_df = row_df.drop(row_df.columns[-2:], axis=1)
                dict_list.append({'df': row_df, 'treatment_effect': treat_effect, 'n_samples': 1})
        else:
            df = pd.DataFrame(flattened_arr).T
            df = df.drop(df.columns[-2:], axis=1) # drop last tau label and delimiter
            dict_list.append({'df': df, 'treatment_effect': treat_effect, 'n_samples': 1})
        return dict_list


    def generate_all(self, n_datasets: int = 10000) -> list:
        datasets = []
        for i in tqdm(range(n_datasets), desc="Generating data"):
            dataset_list = self.generate()
            datasets.extend(dataset_list)
        
        return datasets  
# %%
