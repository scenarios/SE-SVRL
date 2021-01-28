dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
syncbn = True

data = dict(
    videos_per_gpu=16,  # total batch size is 16Gpus*8 == 128
    workers_per_gpu=16,
    train=dict(
        type='MANIFOLDDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='ucf101/annotations/train_split_1.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='ucf101/zip/{}/RGB_frames.zip',
            frame_fmt='image_{:05d}.jpg',
        ),
        clip_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=16,
            strides=[1, 2, 3],
            temporal_jitter=True
        ),
        single_frame_sampler=dict(
            type='RandomFrameSampler',
            num_clips=1,
            clip_len=1,
            strides=1,
            temporal_jitter=False
        ),
        tcp_transform_cfg=[
            dict(type='GroupScale', scales=[112, 128, 144]),
            dict(type='GroupRandomCrop', out_size=112),
            dict(type='GroupFlip', flip_prob=0.50),
            dict(
                type='PatchMask',
                region_sampler=dict(
                    scales=[16, 24, 28, 32, 48, 64],
                    ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    scale_jitter=0.18,
                    num_rois=3,
                ),
                key_frame_probs=[0.5, 0.3, 0.2],
                loc_velocity=3,
                size_velocity=0.025,
                label_prob=0.8
            ),
            dict(type='RandomHueSaturation', prob=0.25, hue_delta=12, saturation_delta=0.1),
            dict(type='DynamicBrightness', prob=0.5, delta=30, num_key_frame_probs=(0.7, 0.3)),
            dict(type='DynamicContrast', prob=0.5, delta=0.12, num_key_frame_probs=(0.7, 0.3)),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ],
        manifold_transform_cfg=[
            #dict(type='GroupScale', scales=[112, 128, 144]),
            #dict(type='GroupRandomCrop', out_size=112),
            dict(type='GroupFlip', flip_prob=0.50),
            dict(type='GroupPermutation', bgr2rgb = True, hwc2chw = True),
            dict(type='GroupRandomResizedCrop',
                 size=112,
                 scale=(0.4, 1.0),
                 ratio=(0.75, 1.3333333333333333),
                 interpolation=2),
            dict(type='RandomColorJitter',brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, prob = 0.8),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=False,
                div255=True,
                hwc2chw = False,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ],
    )
)

# optimizer
total_epochs = 400
optimizer = dict(type='SGD', lr=0.01*4, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.00001
)
checkpoint_config = dict(interval=10, max_keep_ckpts=1, create_symlink=False)
workflow = [('train', 1)]
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)