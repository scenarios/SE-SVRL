_base_ = './pretraining_runtime_ucf.py'

data = dict(
    videos_per_gpu=32,  # total batch size is 32Gpus*8 == 256
    workers_per_gpu=32,
    train=dict(
        type='MANIFOLDDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='kinetics400_frame_zip/train.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='kinetics400_frame_zip/{}/RGB_frames.zip',
            frame_fmt='img_{:05d}.jpg',
        ),
    )
)

# optimizer
total_epochs = 200
optimizer = dict(type='SGD', lr=0.01*8, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.00001
)
checkpoint_config = dict(interval=10, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)
