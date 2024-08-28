_base_ = './base.py'

# model
model = dict(name='PoolingMLP', input_channels=102, 
                                hidden_channels=256, 
                                num_classes=1, 
                                depth=4)

# data
work_dir = './workdir'
data_cfg = 'datasets/linear/lennon100_fixed_tau.py'
data_dir = None # './datasets/linear/lennon100' # activate online data generation by setting data_dir to None
train_batch_size = 256
val_batch_size = 256
test_batch_size = 256

# optimization
max_epochs = 100
lr = 4e-3
weight_decay = 0.0001

# logging
logging = False
exp_name = f'pool_mlp_lennon_bs{train_batch_size}_lr{lr}_eps{max_epochs}_hidden{model["hidden_channels"]}_depth{model["depth"]}'


