import torch
import logging
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
from mmcv.runner import load_state_dict


class BaseBackbone(nn.Module):
    """ Base class for backbone network.
    Args:
        bn_eval (bool): use the statistical means & vars for
            BatchNorm layers, even in .train() mode.
    """
    def __init__(self, bn_eval=False):

        super(BaseBackbone, self).__init__()
        self.bn_eval = bn_eval

    def init_from_pretrained(self, pretrained, logger):
        """ Initialization model weights from pretrained model.
        Args:
            pretrained (str): pretrained model path. (something like *.pth)
            logger (logging.Logger): output logger
        """
        logger.info(f"Loading pretrained backbone from {pretrained}")
        checkpoint = torch.load(pretrained)
        # get state_dict from checkpoint
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            raise RuntimeError(f'No state_dict found in checkpoint file {pretrained}')
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()}
        # strip prefix of backbone
        if any([s.startswith('backbone.') for s in state_dict.keys()]):
            state_dict = {k[9:]: v for k, v in checkpoint['state_dict'].items()
                          if k.startswith('backbone.')}
        load_state_dict(self, state_dict, strict=False, logger=logger)

    def train(self, mode=True):
        super(BaseBackbone, self).train(mode)
        if self.bn_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()