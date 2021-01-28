_base_ = './pretraining_runtime_ucf.py'

data = dict(
    train=dict(
        type='TCPDataset',
        data_source=dict(
            type='JsonClsDataSource',
            ann_file='got10k/annotations/got10k_train.json',
        ),
        backend=dict(
            type='ZipBackend',
            zip_fmt='got10k/zips/{}.zip',
            frame_fmt='{:08d}.jpg',
        ),
    )
)
