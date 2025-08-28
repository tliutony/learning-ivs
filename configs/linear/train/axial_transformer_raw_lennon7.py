from src.utils.config import Config

_base_ = './base.py'

# Axial Transformer over raw unwindowed samples; samples are tokens, features are attended in the feature-axis attention.
model = dict(
    name='AxialTransformer',
    n_blocks=4,
    n_heads=4,
    d_model=64,
    d_hidden=256,
    # Feature-type embedding: T=0, Y=1, all Z*=2 (lennon7 has 7 Z's)
    feature_group_ids=[0, 1] + [2] * 7,
    num_feature_groups=3,
)

# data: local raw parquet datasets
data_dir = './data/lennon7-range-tau-10k'
use_huggingface = False
use_sequence = True          # enforce fixed sequence length per dataset for batching
sequence_length = 500       # matches generator config
data_cfg = None

train_batch_size = 128
val_batch_size = 128
test_batch_size = 128

lazy_loading = True

max_epochs = 35
lr = 1e-3
weight_decay = 1e-4

logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
exp_name = f'axial_transformer_raw_lennon7_bs{train_batch_size}_lr{lr}_wd{weight_decay}_eps{max_epochs}'


