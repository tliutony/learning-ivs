_base_ = './base.py'

# model
model = dict(name='CNN', input_channels=3, hidden_channels=[128, 128], num_classes=1)

# data
# activate online data generation
data_dir = None
data_cfg = '/project/learning-ivs/datasets/linear/linear_norm.py'
train_batch_size = 2048
val_batch_size = 2048
test_batch_size = 2048

# optimization
max_epochs = 100
lr = 0.1
weight_decay = 0.0001

# logging
logging = False
exp_name = f'cnn_linear_normal_bs{train_batch_size}_lr{lr}_eps{max_epochs}'


