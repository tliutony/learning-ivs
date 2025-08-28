"""Dataset configuration to match 7 instrument case akin to learning-ivs/lennon7-range-tau-10k."""

seed = 42  # random seed

# data generation
generation = dict(
    generator="LennonIVGenerator",  # data generator
    n_samples_range=[1000, 1000],  # fixed sample size per dataset
    max_vars=7,  # maximum number of variables
    n_instruments=7,  # number of instruments
    instrument_strength=180 / 1000,  # mu^2 / n_samples, consistent with lennon100 config
    tau_range=[-5, 5],  # treatment effect range
    base_seed=seed,  # random seed
)

# data split
n_datasets = 10000  # number of datasets to generate
n_train = 0.8  # proportion of data to use for training
n_val = 0.1  # proportion of data to use for validation
n_test = 0.1  # proportion of data to use for testing

# work directory
work_dir = "./data/lennon7-range-tau-10k"  # directory to save data


