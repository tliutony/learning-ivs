_base_ = './base.py'

# model
model = dict(name='MLP', input_channels=102, hidden_channels=[64, 128, 64], num_classes=1)

# data
work_dir = './workdir'
data_dir = None # activate online data generation by setting data_dir to None
data_cfg = 'datasets/linear/lennon100.py'
train_batch_size = 512
val_batch_size = 2048
test_batch_size = 2048

# optimization
max_epochs = 5
lr = 0.01
weight_decay = 0.0001

# logging
logging = False
exp_name = f'mlp_lennon100_bs{train_batch_size}_lr{lr}_eps{max_epochs}'


