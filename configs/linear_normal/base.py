# model
model = dict(name='MLP', hidden_channels=[1024, 1024], num_classes=1)

# data
data_dir = './data/linear'
data_cfg = None
train_batch_size = 256
val_batch_size = 512
test_batch_size = 512

# optimization
max_epochs = 100
optimizer = dict(name='Adam', lr=0.001, weight_decay=0.0001)

# logging
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
