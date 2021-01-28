_base_ = '../r3d_18_ucf101_224/finetune_ucf101.py'

work_dir = './output/tcp/r2plus1d_18_ucf101_224/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/tcp/r2plus1d_18_ucf101_224/pretraining/epoch_300.pth',
    ),
)