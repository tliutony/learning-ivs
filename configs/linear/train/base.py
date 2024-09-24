# model
model = dict(name='MLP', num_classes=1)

# data
data_dir = './data/linear'  # Local data directory or Hugging Face dataset name
use_huggingface = False  # Set to True if using Hugging Face dataset
use_sequence = False  # Set to True to use sequence sampling
data_cfg = None
train_batch_size = 256
val_batch_size = 512
test_batch_size = 512
sequence_length = 1000  # Only used if use_sequence is True

# optimization
max_epochs = 100
optimizer = dict(name='Adam', lr=0.001, weight_decay=0.0001)

# logging
logging = True
project_name = 'iv_linear_normal'
work_dir = './checkpoints/linear'
early_stopping = dict(monitor='val_loss', mode='min', patience=20)
checkpoint = dict(monitor='val_loss', mode='min', save_top_k=1)
