_base_ = [
    '../_base_/default_runtime.py',
]

max_epochs = 300
num_last_epochs = 15
interval = 1

base_lr = 0.0001

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=max_epochs+1)

custom_hooks = [
    dict(
        type='CustomEvalHook',
        val_epochs=list(range(max_epochs - num_last_epochs, max_epochs))
    )
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
)

# learning rate
param_scheduler = [
    dict(
        # TODO: fix default scope in get function
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,
        begin=5,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=interval,
        save_best='coco/bbox_mAP',
        max_keep_ckpts=3,
    )
)

custom_hooks = [
    dict(
        type='MyModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')