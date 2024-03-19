_base_ = './base.py'

# data
# activate online data generation
data_dir = None
data_cfg = 'datasets/linear/linear_norm.py'
train_batch_size = 512
val_batch_size = 512
test_batch_size = 512

# model
# TODO:
#  currently only finished debugging and integrating model class, need more check on if it is implemented correctly
#  hyperparameters are randomly chosen, need to tune them
#  The unsupervised loss part is NaN, might due to hyperparameter/gradient/forgetting... need to gradient clipping
#  and more detailed look in which part is going wrong
model = dict(name='NeuralStats', sample_size=1000, n_features=3,
                     c_dim=32, n_hidden_statistic=128, hidden_dim_statistic=64,
                     n_stochastic=3, z_dim=64, n_hidden=3, hidden_dim=128)

# optimization
max_epochs = 100
lr = 0.001
weight_decay = 0.0001

# logging
logging = False
exp_name = f'neuralstats_linear_normal_bs{train_batch_size}_lr{lr}_eps{max_epochs}'
early_stopping = dict(monitor='val_mse_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_mse_loss', mode='min', save_top_k=1)


