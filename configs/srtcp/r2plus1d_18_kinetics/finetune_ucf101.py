_base_ = '../r3d_18_kinetics/finetune_ucf101.py'

work_dir = './output/srtcp/r2plus1d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        pretrained='/mnt/data/projects/SR-SVRL/SRTCP-AuxDeepShare-bugfix_K400P-ep200-bs64-ts16-ss112-T0.07-proj512_ucf-ALFinetune-ep150-bs32-ts16-ss112_ModelR2p1dl18/pt-results/pt-SRTCP-AuxDeepShare-bugfix_-df35e302_1609329433_f161a174/srtcp/r2plus1d_18_kinetics/pretraining/epoch_200.pth',
        #pretrained='./output/srtcp/r2plus1d_18_kinetics/pretraining/epoch_200.pth',
    ),
    cls_head=dict(
        dropout_ratio = 0.8
    ),
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)