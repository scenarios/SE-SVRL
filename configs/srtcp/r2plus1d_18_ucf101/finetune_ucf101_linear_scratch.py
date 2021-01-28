_base_ = './finetune_ucf101.py'

work_dir = './output/srtcp/r2plus1d_18_ucf101/finetune_ucf101_linear_scratch/'

model = dict(
    backbone=dict(
        pretrained=None,
    ),
    FreezeBackbone=True
)
