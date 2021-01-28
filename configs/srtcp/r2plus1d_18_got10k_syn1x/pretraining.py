_base_ = '../r3d_18_got10k_syn1x/pretraining.py'

work_dir = './output/tcp/r2plus1d_18_got10k_syn1x/pretraining/'

model = dict(
    backbone=dict(
        type='R2Plus1D',
    )
)


