_base_ = '../r3d_18_ucf101_224/finetune_hmdb51.py'

work_dir = './output/tcp/r2plus1d_18_ucf101_224/finetune_hmdb51/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/tcp/r2plus1d_18_ucf101_224/pretraining/epoch_300.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)