seed = 42  # random seed

# data generation
generation = dict(
    generator="LinearNormalDataGenerator",  # data generator
    n_samples_range=[1000, 1000],  # range of sample size
    iv_strength_range=[1.0, 2.0],  # range of instrument strength
    conf_strength_range=[0.0, 1.0],  # range of confounder strength
    treat_effect_range=[-5.0, 5.0],  # range of treatment effect
    conf_effect_range=[0.0, 1.0],  # range of confounder effect
    beta_range=[-1.0, 1.0],  # range of beta (constant bias term)
    base_seed=seed,  # random seed
)

# data split
n_datasets = 1000  # number of datasets to generate
n_train = 0.8  # proportion of data to use for training
n_val = 0.1  # proportion of data to use for validation
n_test = 0.1  # proportion of data to use for testing

# work directory
work_dir = "/project/learning-ivs/data/linear_norm"  # directory to save data
