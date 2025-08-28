from src.utils.config import Config

_base_ = './base.py'

# model: raw (unwindowed) Lennon7 dataset has features [T, Y, Z0..Z6] => d_model = 9
model = dict(name='TransformerEncoder', n_blocks=4, n_heads=4, d_model=9, d_hidden=256, pooling='average')

# data: use locally generated raw parquet datasets
data_dir = './data/lennon7-range-tau-10k'
use_huggingface = False
use_sequence = True         # enforce fixed-length sequences for batching
sequence_length = 1000      # matches n_samples per dataset in generator config
data_cfg = None

# batch sizes
train_batch_size = 256
val_batch_size = 256
test_batch_size = 256

lazy_loading = True

# optimization
max_epochs = 35
lr = 1e-4
weight_decay = 1e-4

# logging
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
exp_name = f'transformer_raw_lennon7_bs{train_batch_size}_lr{lr}_wd{weight_decay}_eps{max_epochs}'


