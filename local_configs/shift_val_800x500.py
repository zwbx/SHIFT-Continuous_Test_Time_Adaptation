_base_ = [
    './_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/default_runtime.py', 
    './_base_/schedules/schedule_160k_adamw.py',
]

# dataset settings
dataset_type = 'ShiftDataset'
data_split_type = 'videos_1x_val'
data_root = '.../SHIFT/continuous/videos/1x'
csv_root = data_root + '/val/front/seq.csv'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1280, 800)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 800), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 500),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75,2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','gt_semantic_seg']),
            dict(type='Collect', keys=['img','gt_semantic_seg']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/front/img',
        ann_dir='train/front/semseg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/front/img',
        ann_dir='val/front/semseg',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/front/img/',
        ann_dir='val/front/semseg/',
        pipeline=test_pipeline))

evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

YOUR_DATASET_CLASSES=14
model = dict( 
    decode_head=dict(num_classes=YOUR_DATASET_CLASSES), 
    auxiliary_head=dict(num_classes=YOUR_DATASET_CLASSES),
    train_cfg=dict(),)
    # test_cfg=dict(mode='slide', crop_size=(700,700), stride=(600,600)))



