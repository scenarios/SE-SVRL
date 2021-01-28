_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_hmdb51.py']

work_dir = './output/tcp/r3d_18_ucf101_224/finetune_hmdb51/'

model = dict(
    backbone=dict(
        stem=dict(with_pool=True),
        pretrained='./output/tcp/r3d_18_ucf101_224/pretraining/epoch_300.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)

data = dict(
    train=dict(
        transform_cfg=[
                dict(type='GroupScale', scales=[224, 240, 256, 272, 288]),
                dict(type='GroupFlip', flip_prob=0.50),
                dict(type='RandomBrightness', prob=0.20, delta=32),
                dict(type='RandomContrast', prob=0.20, delta=0.20),
                dict(type='RandomHueSaturation', prob=0.20, hue_delta=12, saturation_delta=0.1),
                dict(type='GroupRandomCrop', out_size=224),
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=True,
                    div255=True,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
    ),
    val=dict(
        transform_cfg=[
            dict(type='GroupScale', scales=[256]),
            dict(type='GroupCenterCrop', out_size=224),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    ),
    test=dict(
        transform_cfg=[
            dict(type='GroupScale', scales=[256]),
            dict(type='GroupCenterCrop', out_size=224),
            dict(
                type='GroupToTensor',
                switch_rgb_channels=True,
                div255=True,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ]
    ),
)
