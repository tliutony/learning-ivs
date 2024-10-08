model = dict(name='TransformerEncoder', n_blocks=4, n_heads=4, d_model=1, d_hidden=256, pooling='average')

# data, online generation

# for now, locally generate data using generate data script for lennon fixed tau
# will create local directory - use this data_dir in data_dir below
data_dir = None
# need separate data_cfg for transformer ready input
# data_cfg = './datasets/linear/transformer_linear_norm.py'
train_batch_size = 2048
val_batch_size = 2048
test_batch_size = 2048

# optimization
max_epochs = 100
lr = 0.1
weight_decay = 0.0001

# logging
logging = False
project_name = 'iv_linear_normal'
# work_dir = './checkpoints/linear'
work_dir = './workdir'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
exp_name = f'transformer_linear_normal_bs{train_batch_size}_lr{lr}_eps{max_epochs}'