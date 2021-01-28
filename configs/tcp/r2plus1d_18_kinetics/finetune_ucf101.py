_base_ = '../r3d_18_kinetics/finetune_ucf101.py'

work_dir = './output/tcp/r2plus1d_18_kinetics/finetune_ucf101/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
        #pretrained='./output/tcp/r2plus1d_18_kinetics/pretraining/epoch_200.pth',
        pretrained='/mnt/data/projects/SR-SVRL/TCP_KineicsPre-epoch200-bs64-tsize16-ssize112_ucf-ALFinetuneAdam-ep150-bs32-tsize16-ssize112_ModelR2p1dl18/pt-results/pt-TCP_KineicsPre-epoch200-bs-be369022_1609331308_b7dbf433/tcp/r2plus1d_18_kinetics/pretraining/epoch_200.pth',
    ),
)
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4)