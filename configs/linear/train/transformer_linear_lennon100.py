from src.utils.config import Config

_base_ = './base.py'

# model
model = dict(name='TransformerEncoder', n_blocks=4, n_heads=4, d_model=1, d_hidden=256, pooling='average')

# data
work_dir = './workdir'
# data_dir = './tmp_lennon100/range'
# prefer hf_dataset of data_dir
hf_dataset = 'learning-ivs/lennon100-range-tau-10k' # prefer hf_dataset over data_dir

# hf_dataset conditional args
transformer_transform = True # transform hf data to transformer ready format

# custom transformer data_cfg for online transformation
data_cfg = Config(dict(
    generation = dict(generator="TransformerDataGenerator",
                    data_path=None, # this gets updated in train.py at runtime to path where hf_dataset gets locally downloaded
                    window_size=1,
                    stage='train'),
    # data splitting 
    n_datasets = 10000,
    n_train = 0.1, # 0.8,  # reduced n_train for more tractable training. # proportion of data to use for training
    n_val = 0.1,  # proportion of data to use for validation
    n_test = 0.8 # 0.1  # proportion of data to use for testing
                    ))


# batch sizes
train_batch_size = 256
val_batch_size = 256
test_batch_size = 256

lazy_loading = False

# optimization
max_epochs = 35 # changed from 100 to 50 to 35
lr = 1e-4
weight_decay = 1e-4

# logging
logging = True
#exp_name = f'mlp_lennon100_bs{train_batch_size}_lr{lr}_eps{max_epochs}'x
exp_name = f'transformer_lennon100_bs{train_batch_size}_lr{lr}_wd{weight_decay}_eps{max_epochs}'
# exp_name = f'transformer_lennon100_lr{lr}_eps{max_epochs}'


# logging
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
