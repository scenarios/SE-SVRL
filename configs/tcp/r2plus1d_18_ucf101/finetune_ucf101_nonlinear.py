_base_ = './finetune_ucf101.py'

work_dir = './output/tcp/r2plus1d_18_ucf101/finetune_ucf101_nonlinear/'

model = dict(
    backbone=dict(
        pretrained='/mnt/data/projects/SR-SVRL/TCP_UCF101Pretrain-epoch300-bs32-tsize16-ssize112_ucfFinetune-epoch150-bs32-tsize16-ssize112_ModelR2p1dl18/pt-results/pt-TCP_UCF101Pretrain-epoch30-9e700479_1604380337_758242d5/tcp/r2plus1d_18_ucf101/pretraining/epoch_300.pth',
    ),
    cls_head=dict(
        non_linear = True,
        nonlinear_channels = 512,
        dropout_ratio = 0.0
    ),
    FreezeBackbone=True
)
