# GESimOTAAssigner：gedm； CGPA include CGSF

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
data_root = 'F:/data/shiyanshi'
dataset_type = 'WorkpieceDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, save_best='coco/bbox_mAP_50', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        _delete_=True,
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        init_cfg=dict(
            checkpoint='F:/paper_learning/gmtnet/GMTNet-main/mmdet/swin_tiny_patch4_window7_224.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=False),
    bbox_head=dict(
        act_cfg=dict(type='Swish'),
        # conv_cfg=dict(type='DCBv2'),
        feat_channels=128,
        in_channels=256,
        loss_bbox=dict(
            eps=1e-16,
            loss_weight=5.0,
            mode='square',
            reduction='sum',
            type='IoULoss'),
        loss_cls=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        loss_l1=dict(loss_weight=1.0, reduction='sum', type='L1Loss'),
        loss_obj=dict(
            loss_weight=1.0,
            reduction='sum',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=1,
        stacked_convs=2,
        strides=(
            8,
            16,
            32,
        ),
        type='CGSFHead',
        use_depthwise=False,),
        data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            96,
            192,
            384,
            768,
        ],
        num_outs=5,
        out_channels=256,
        start_level=1,
        type='DFPN'),
    test_cfg=dict(
        max_per_img=500,  # 500
        min_bbox_size=0,
        nms=dict(iou_threshold=0.3, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(iou_weight=0.13, type='GESimOTAAssigner'),  # GESimOTAAssigner,   pos_iou_thr=0.5, neg_iou_thr=0.4,
        debug=False,
        pos_weight=-1),
    type='ATSS')
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.05))
param_scheduler = [
    dict(begin=0, by_epoch=False, end=50, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=48,  # 48
        gamma=0.1,
        milestones=[
            32,
            44,
        ],
        type='MultiStepLR'),
]
pretrained = 'F:/paper_learning/gmtnet/GMTNet-main/mmdet/swin_tiny_patch4_window7_224.pth'
resume = False
scale = (
    640,
    640,
)
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='F:/data/shiyanshi/annotations/instances_test.json',   # F:/data/shiyanshi/annotations/instances_test.json
        backend_args=None,
        data_prefix=dict(img='F:/data/shiyanshi/test'),  # F:/data/shiyanshi/test
        data_root='F:/data/shiyanshi',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='WorkpieceDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='F:/data/shiyanshi/annotations/instances_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric',
    outfile_prefix='E:/paper-learning/gmtnet/GMTNet-main/tools/work_dirs/GMTNet/val-result/results'
)
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=48, type='EpochBasedTrainLoop', val_interval=1, val_begin=30)  # max_epochs=48
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='F:/data/shiyanshi/annotations/instances_train.json',
        backend_args=None,
        data_prefix=dict(img='F:/data/shiyanshi/train'),
        data_root='F:/data/shiyanshi',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs',
                 ),
        ],
        type='WorkpieceDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_image = 'F:/data/shiyanshi/train'
train_json = 'F:/data/shiyanshi/annotations/instances_train.json'
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        640,
        640,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs',

         ),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='F:/data/shiyanshi/annotations/instances_val.json',
        backend_args=None,
        data_prefix=dict(img='F:/data/shiyanshi/val'),
        data_root='F:/data/shiyanshi',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='WorkpieceDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='F:/data/shiyanshi/annotations/instances_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_image = 'F:/data/shiyanshi/val'
val_json = 'F:/data/shiyanshi/annotations/instances_val.json'
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs\\GMTNet'


custom_imports = dict(imports=['gmt_hooks'], allow_failed_imports=False)

# 2. 在 custom_hooks 中使用它
# 因为我们在 gmt_hooks.py 里用了 @HOOKS.register_module()，
# 所以这里可以直接用 type='CostBalancePrinterHook' 字符串来引用
custom_hooks = [
    dict(type='CostBalancePrinterHook')
]

randomness = dict(seed=2027, deterministic=True)