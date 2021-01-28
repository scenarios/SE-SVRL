import _init_paths
import os
import argparse

from pyvrl.builder import build_model, build_dataset
from pyvrl.apis import test_recognizer, get_root_logger
from mmcv import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation an action recognizer')
    parser.add_argument('--cfg', default='', type=str, help='config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument('--data_dir', default='data/', type=str, help='the dir that save training data')
    parser.add_argument('--checkpoint', help='the checkpoint file to resume from')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of gpus to use '
                             '(only applicable to non-distributed training)')
    parser.add_argument('--batchsize', type=int, default=6)
    parser.add_argument('--seed', type=int, default=7, help='random seed')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    cfg = Config.fromfile(args.cfg)
    # update configs according to CLI args
    cfg.gpus = args.gpus
    cfg.data.videos_per_gpu = args.batchsize
    if 'pretrained' in cfg['model']['backbone']:
        cfg['model']['backbone']['pretrained'] = None
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.data_dir is not None:
        if 'test' in cfg.data:
            cfg.data.test.root_dir = args.data_dir
    if args.checkpoint is not None:
        chkpt_list = [args.checkpoint]
    else:
        chkpt_list = [os.path.join(cfg.work_dir, fn)
                      for fn in os.listdir(cfg.work_dir) if fn.endswith('.pth')]

    # init logger before other steps
    logger = get_root_logger(log_level=cfg.log_level)

    # build a dataloader
    model = build_model(cfg.model, default_args=dict(train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg))
    dataset = build_dataset(cfg.data.test)
    results = test_recognizer(model,
                              dataset,
                              cfg,
                              chkpt_list,
                              logger=logger,
                              progress=args.progress)
