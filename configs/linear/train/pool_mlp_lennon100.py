_base_ = './base.py'

# model
model = dict(name='PoolingMLP', input_channels=102, 
                                hidden_channels=256, 
                                num_classes=1, 
                                depth=4)

# data
work_dir = './workdir'
use_huggingface = False
use_sequence = False
sequence_length = None
data_cfg = 'datasets/linear/lennon100_fixed_tau.py'
data_dir = '/data/shared/huggingface/hub/datasets--learning-ivs--lennon100-range-tau-10k/snapshots/cf012de277abb6146c84cf543a3b0819bbff1a3c' # activate online data generation by setting data_dir to None
lazy_loading = True
train_batch_size = 256
val_batch_size = 256
test_batch_size = 256

# optimization
max_epochs = 100
lr = 1e-3
weight_decay = 0.0001

# logging
logging = True
exp_name = f'pool_mlp_lennon100_10k_bs{train_batch_size}_lr{lr}_eps{max_epochs}_hidden{model["hidden_channels"]}_depth{model["depth"]}'