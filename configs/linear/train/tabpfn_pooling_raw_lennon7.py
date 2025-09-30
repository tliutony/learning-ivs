from src.utils.config import Config

_base_ = './base.py'

# Freeze TabPFN backbone, learn a small pooling head for tau regression
model = dict(
    name='TabPFNPoolingRegressor',
    n_estimators=1,
    head_hidden_dim=128,
    cache_dir='./data/tabpfn_embed_cache',
    device='cuda',
)

data_dir = './data/lennon7-range-tau-10k'
use_huggingface = False
use_sequence = True
sequence_length = 500
data_cfg = None

train_batch_size = 128   # TabPFN.fit called per-dataset; keep small
val_batch_size = 128
test_batch_size = 128

lazy_loading = True

max_epochs = 50
lr = 1e-3
weight_decay = 1e-4
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=5)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
exp_name = f'tabpfn_pooling_lennon7_bs{train_batch_size}_eps{max_epochs}'


