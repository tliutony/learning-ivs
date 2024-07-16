_base_ = './base.py'
# figure this out, it shouldn't take too long
model = dict(name='TransformerEncoder', n_blocks=3, n_heads=4, d_model=1, d_hidden=10)

# data, online generation
# copied so far
data_dir = None
# need separate data_cfg for transformer ready input
data_cfg = '/project/learning-ivs/datasets/linear/transformer_linear_norm.py'
train_batch_size = 2048
val_batch_size = 2048
test_batch_size = 2048

# optimization
max_epochs = 100
lr = 0.1
weight_decay = 0.0001

# logging
logging = False
exp_name = f'transformer_linear_normal_bs{train_batch_size}_lr{lr}_eps{max_epochs}'