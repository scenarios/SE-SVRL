#_base_ = './finetune_ucf101.py'
_base_ = '../r3d_18_ucf101/finetune_ucf101.py'
work_dir = './output/manifoldlearning/r2plus1d_18_ucf101/finetune_ucf101_linear/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/manifoldlearning/r2plus1d_18_ucf101/pretraining/epoch_400.pth',
    ),
    cls_head=dict(
        dropout_ratio=0.0
    ),
    FreezeBackbone=True
)
optimizer = dict(type='Adam', lr=0.1)