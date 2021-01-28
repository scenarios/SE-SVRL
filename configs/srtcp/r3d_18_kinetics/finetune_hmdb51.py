_base_ = ['../../recognizers/_base_/model_r3d18.py',
          '../../recognizers/_base_/runtime_hmdb51.py']

work_dir = './output/tcp/r3d_18_kinetics/finetune_hmdb51/'

model = dict(
    backbone=dict(
        # pretrained='./output/tcp/r3d_18_kinetics/pretraining/epoch_90.pth',
        pretrained='/mnt/data/projects/PyVRL/1024_tcp_r3d_k400/pt-results/pt-1024_tcp_r3d_k400-45d51286_1603524098_7b28951b/tcp/r3d_18_kinetics/pretraining/epoch_90.pth',
    ),
    cls_head=dict(
        num_classes=51
    )
)