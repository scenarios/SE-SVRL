_base_ = '../pretraining_runtime_ucf.py'

work_dir = './output/tcp/r3d_18_ucf101_224/pretraining/'

model = dict(
    type='TCP',
    backbone=dict(
        type='R3D',
        depth=18,
        num_stages=4,
        stem=dict(
            temporal_kernel_size=3,
            temporal_stride=1,
            in_channels=3,
            with_pool=False,
        ),
        down_sampling=[False, True, True, False],
        down_sampling_temporal=[False, True, True, True],
        channel_multiplier=1.0,
        bottleneck_multiplier=1.0,
        with_bn=True,
        zero_init_residual=True
    ),
    head=dict(
        in_channels=512,
        in_temporal_size=2,
        hidden_channels=512,
        roi_feat_size=5,
        spatial_stride=8.0,
        num_pred_frames=16,
        target_means=(0., 0., 0., 0.),
        target_stds=(1.6, 1.6, 0.08, 0.08),
    )
)

data = dict(
    train=dict(
        transform_cfg=[
            dict(type='GroupScale', scales=[224, 240, 256, 272, 288]),
            dict(type='GroupRandomCrop', out_size=224),
            dict(type='GroupFlip', flip_prob=0.50),
            dict(
                type='PatchMask',
                region_sampler=dict(
                    scales=[32, 48, 56, 64, 96, 128],
                    ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                    scale_jitter=0.18,
                    num_rois=3,
                ),
                key_frame_probs=[0.5, 0.3, 0.2],
                loc_velocity=8,
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
        ]
    )
)
