# model settings
num_classes = 1
# loss function, set multiple losses if needed, allow weighted sum
loss = [dict(type='TorchLoss', loss_name='MSELoss', multi_label=True, loss_weight=1.0)]

model = dict(
    # Base Encoder Decoder Architecture (Backbone+Head)
    type='CIEncoder',
    # Visual Backbone from Timm Models library
    backbone=dict(
        type='TimmModels',
        model_name='resnet18',
        features_only=False,
        remove_fc=True,
        pretrained=False
    ),
    # Linear Head for classification
    head=dict(
        type='BaseHead',
        in_index=-1,
        dropout=0.3,
        num_classes=num_classes,
        channels=None,
        losses=loss
    ),
    # auxiliary_head, only work in training for auxiliary loss
    # auxiliary_head=None,
    # metrics for evaluation, allow multiple metrics
    evaluation = dict(metrics=[dict(type='TorchMetrics', metric_name='MeanSquaredError', multi_label=True,
                                    prob=False),
                               dict(type='TorchMetrics', metric_name='R2Score', multi_label=True, prob=False)])
)

# dataset settings
dataset_type = 'CILiNGAM'

# training data preprocess pipeline
train_pipeline = [dict(type='ToTensor')]

# validation data preprocess pipeline
test_pipeline = [dict(type='ToTensor')]

data = dict(
    train_batch_size=256,  # for single card
    val_batch_size=256,
    test_batch_size=256,
    num_workers=4,
    train=dict(
        type=dataset_type,
        base_seed=42,
        mini_batch_size=500,
        n_datasets=10000,
        n_samples=[500, 10000],
        variable_cfg=dict(iv_strength=[0.1, 0.9],
                          conf_strength=[1, 5],
                          treat_effect=[-2, 2],
                          conf_effect=[1, 5]),
        sampler=None,  # None is default sampler, set to RandomSampler/DistributedSampler
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        base_seed=3407,
        mini_batch_size=500,
        n_datasets=2000,
        n_samples=[500, 10000],
        variable_cfg=dict(iv_strength=[0.1, 0.9],
                          conf_strength=[1, 5],
                          treat_effect=[-2, 2],
                          conf_effect=[1, 5]),
        sampler='SequentialSampler',
        pipeline=test_pipeline,
        ),
    test=dict(
        type=dataset_type,
        base_seed=3407,
        mini_batch_size=500,
        n_datasets=2000,
        n_samples=[500, 10000],
        variable_cfg=dict(iv_strength=[0.1, 0.9],
                          conf_strength=[1, 5],
                          treat_effect=[-2, 2],
                          conf_effect=[1, 5]),
        sampler='SequentialSampler',
        pipeline=test_pipeline
    ),
)

# yapf:disable
log = dict(
    # project name, used for cometml
    project_name='LCI',
    # work directory, used for saving checkpoints and loggings
    work_dir='/data2/charon/LCI',
    # explicit directory under work_dir for checkpoints and config
    exp_name='ci_lingam-n_datasets=100-n_samples=500_10000-model=resnet50-embedding=128-lr=0.1-epoch=100',
    logger_interval=50,
    # monitor metric for saving checkpoints
    monitor='val_mean_squared_error',
    # logger type, support TensorboardLogger, CometLogger
    logger=[dict(type='comet', key='oN1q8cGSIrH0zhorxKpNoenyc')],
    # checkpoint saving settings
    checkpoint=dict(type='ModelCheckpoint',
                    top_k=1,
                    mode='min',
                    verbose=True,
                    save_last=False,
                    ),
    # early stopping settings
    earlystopping=dict(
            mode='min',
            strict=False,
            patience=20,
            min_delta=0.0001,
            check_finite=True,
            verbose=True
    )

)

# yapf:enable
# resume from a checkpoint
resume_from = None
cudnn_benchmark = True

# optimization
optimization = dict(
    # running time unit, support epoch and iter
    type='epoch',
    # total running units
    max_iters=100,
    # optimizer
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.0005),
    # learning rate scheduler and warmup
    scheduler=dict(type='CosineAnnealing',
                   interval='step',
                   # warmup=dict(type='LinearWarmup', period=0.1)
                   )

)
