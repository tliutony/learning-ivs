# configs/linear/train/pooling_mlp_raw_lennon7.py
from src.utils.config import Config

_base_ = './base.py'

model = dict(name='PoolingMLP', input_channels=9, hidden_channels=256, depth=2, num_classes=1)

data_dir = './data/lennon7-range-tau-10k'
use_huggingface = False
use_sequence = True
sequence_length = 1000
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
exp_name = f'pooling_mlp_raw_lennon7_bs{train_batch_size}_lr{lr}_wd{weight_decay}_eps{max_epochs}'