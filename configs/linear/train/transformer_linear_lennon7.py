_base_ = './base.py'

# model
model = dict(name='TransformerEncoder', n_blocks=2, n_heads=2, d_model=1, d_hidden=16, pooling='average')

# data
work_dir = './workdir'
# data_dir = './tmp_lennon100/range' # activate online data generation by setting data_dir to None and specifying data_cfg
data_cfg = None
# prefer hf_dataset over data_dir
hf_dataset = 'learning-ivs/lennon7-range-tau-10k'
# hf_dataset conditional args
window_size = 1
transformer_transform = True # transform hf data to transformer ready format

# batch sizes
train_batch_size = 256
val_batch_size = 256
test_batch_size = 256

# optimization
max_epochs = 50
lr = 3e-3
weight_decay = 1e-4

# logging
logging = True
exp_name = f'transformer_lennon7_bs{train_batch_size}_lr{lr}_wd{weight_decay}_eps{max_epochs}'

# logging
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
