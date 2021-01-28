_base_ = './finetune_ucf101.py'

work_dir = './output/srtcp/r2plus1d_18_kinetics/finetune_ucf101_nonlinear/'

model = dict(
    backbone=dict(
        pretrained='/mnt/data/projects/SR-SVRL/SRTCP-AuxDeepShare_K400Pre-epoch200-bs64-tsize16-ssize112-T0.07_ucfFinetune-epoch150-bs32-tsize16-ssize112_ModelR2p1dl18/pt-results/pt-SRTCP-AuxDeepShare_K400Pre-191ef557_1605353476_dea30573/srtcp/r2plus1d_18_kinetics/pretraining/epoch_200.pth',
    ),
    cls_head=dict(
        non_linear = True,
        nonlinear_channels = 512,
        dropout_ratio = 0.0
    ),
    FreezeBackbone=True
)
