"""Dataset configuration to match 100 instrument case from Lennon et al. 2022."""
seed = 42  # random seed

# data generation
generation = dict(generator='LennonIVGenerator',  # data generator
                  n_samples_range=[1000, 1000],  # range of sample size
                  max_vars=100,  # maximum number of variables
                  n_instruments=100,  # number of instruments
                  instrument_strength=180 / 1000,  # instrument strength, mu^2/n_samples
                  tau=1,
                  base_seed=seed  # random seed
                  )

# data split
n_datasets = 1000  # number of datasets to generate
n_train = 0.8  # proportion of data to use for training
n_val = 0.1  # proportion of data to use for validation
n_test = 0.1  # proportion of data to use for testing

# work directory
work_dir = '/project/learning-ivs/data/lennon100'  # directory to save data
