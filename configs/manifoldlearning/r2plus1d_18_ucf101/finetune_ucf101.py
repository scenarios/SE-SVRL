_base_ = '../r3d_18_ucf101/finetune_ucf101.py'

work_dir = './output/manifoldlearning/r2plus1d_18_ucf101/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='./output/manifoldlearning/r2plus1d_18_ucf101/pretraining/epoch_400.pth',
    ),
    cls_head=dict(
        dropout_ratio = 0.8
    ),
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)