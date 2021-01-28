from __future__ import division

from collections import OrderedDict

import torch
from torch import distributed as dist

from mmcv.runner import DistSamplerSeedHook, build_optimizer, EpochBasedRunner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from .env import get_root_logger
from ..core import DistOptimizerHook, EvalHook, DistEvalHook
from ..datasets.dataloader import build_dataloader
from ..builder import build_dataset


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars


def batch_processor(model, data, train_mode):
    """Process a data batch.
    This method is required as an argument of Runner, which defines how to
    process a data batch and obtain proper outputs. The first 3 arguments of
    batch_processor are fixed.
    Args:
        model (nn.Module): A PyTorch model.
        data (dict): The data batch in a dict.
        train_mode (bool): Training mode or not. It may be useless for some
            models.
    Returns:
        dict: A dict containing losses and log vars.
    """
    losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['imgs']))

    return outputs


def train_network(model,
                  dataset,
                  cfg,
                  distributed=False,
                  validate=False,
                  logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    multiprocessing_context = None
    if cfg.get('numpy_seed_hook', True) and cfg.data.workers_per_gpu > 0:
        # https://github.com/pytorch/pytorch/issues/5059
        logger.info("Use spawn method for dataloader.")
        multiprocessing_context = 'spawn'
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.videos_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=cfg.gpus,
            dist=distributed,
            multiprocessing_context=multiprocessing_context)
    ]

    # start training
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # convert to syncbn
        if cfg.get('syncbn', False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()
        optimizer_config = cfg.optimizer_config

    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger)
    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
            drop_last=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    logger.info("Finish training... Exit... ")
