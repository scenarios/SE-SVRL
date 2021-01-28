_base_ = './finetune_ucf101.py'

work_dir = './output/tcp/r2plus1d_18_kinetics/finetune_ucf101_linear/'

model = dict(
    cls_head=dict(
        dropout_ratio = 0.0
    ),
    FreezeBackbone=True
)
optimizer = dict(type='Adam', lr = 0.01)