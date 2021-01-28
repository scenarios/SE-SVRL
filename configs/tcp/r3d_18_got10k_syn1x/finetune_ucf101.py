_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_ucf101.py']

work_dir = './output/tcp/r3d_18_got10k_syn1x/finetune_ucf101/'

model = dict(
    backbone=dict(
        pretrained='./output/tcp/r3d_18_got10k_syn1x/pretraining/epoch_300.pth',
    ),
)
