_base_ = '../r3d_18_ucf101_224/pretraining.py'

work_dir = './output/tcp/r2plus1d_18_ucf101_224/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    )
)


