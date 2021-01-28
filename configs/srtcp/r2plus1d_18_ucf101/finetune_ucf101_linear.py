_base_ = './finetune_ucf101.py'

work_dir = './output/srtcp/r2plus1d_18_ucf101/finetune_ucf101_linear/'

model = dict(
    backbone=dict(
        pretrained='/mnt/data/projects/SR-SVRL/SRTCP-AuxDeepShare_UCF101Pre-epoch300-bs32-tsize16-ssize112-T0.07_ucfFinetune-epoch150-bs32-tsize16-ssize112_ModelR2p1dl18/pt-results/pt-SRTCP-AuxDeepShare_UCF101P-85a460f6_1605160845_7135b35a/srtcp/r2plus1d_18_ucf101/pretraining/epoch_300.pth',
    ),
    FreezeBackbone=True
)
