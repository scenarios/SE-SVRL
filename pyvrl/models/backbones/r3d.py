""" R3D & R(2+1)D model proposed in [1]
[1] A Closer Look at Spatiotemporal Convolutions for Action Recognition, CVPR 2018
"""

import numpy as np
import logging
import torch

from torch import nn
from collections import OrderedDict
from typing import List
from mmcv.cnn import kaiming_init, constant_init

from .base_backbone import BaseBackbone
from ...builder import BACKBONES

from typing import Iterable, Union


class build_3d_conv(nn.Module):

    def __init__(self,
                 block_type: str,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Iterable[int]],
                 stride: Union[int, Iterable[int]] = 1,
                 padding: Union[int, Iterable[int]] = 0,
                 dilation: Union[int, Iterable[int]] = 1,
                 groups: int = 1,
                 with_bn: bool = True,
                 deepshare: bool = True
                 ):
        """ build a 3D convolution block. The structure is
                conv -> (optional) bn -> relu
            Args:
                block_type (str): the block type. '3d' means the pure 3D conv
                    and '2.5d' means the R(2+1)D conv.
                in_channels (int): input channels
                out_channels (int): output channels
                kernel_size (int|list): convolution kernel size
                stride (int|list): convolution stride
                padding (int|list): padding size
                dilation (int|list): dilation size
                groups (int): number of the groups
                with_bn (bool): whether apply BatchNorm.
             """

        super(build_3d_conv, self).__init__()
        self.block_type = block_type

        # extend to 3d kernel size
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(padding, int):
            padding = [padding] * 3
        if isinstance(dilation, int):
            dilation = [dilation] * 3

        self.temporal_kernel_size = kernel_size[0]
        self.temporal_stride = stride[0]
        self.temporal_padding = padding[0]
        self.deepshare = deepshare

        _dict = OrderedDict()
        if block_type == '2.5d':
            # building block for R(2+1)D conv.
            mid_channels = 3 * in_channels * out_channels * kernel_size[1] * kernel_size[2]
            mid_channels /= (in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels)
            mid_channels = int(mid_channels)

            # build spatial convolution
            _dict['conv_s'] = nn.Conv3d(in_channels=in_channels,
                                        out_channels=mid_channels,
                                        kernel_size=[1, kernel_size[1], kernel_size[2]],
                                        stride=[1, stride[1], stride[2]],
                                        padding=[0, padding[1], padding[2]],
                                        dilation=[1, dilation[1], dilation[2]],
                                        groups=groups,
                                        bias=not with_bn)
            if with_bn:
                _dict['bn_s'] = nn.BatchNorm3d(mid_channels, eps=1e-3)
            _dict['relu_s'] = nn.ReLU()
            _dict['conv_t'] = nn.Conv3d(in_channels=mid_channels,
                                        out_channels=out_channels,
                                        kernel_size=[kernel_size[0], 1, 1],
                                        stride=[stride[0], 1, 1],
                                        padding=[padding[0], 0, 0],
                                        dilation=[dilation[0], 1, 1],
                                        groups=groups,
                                        bias=False)
            if self.deepshare:
                _dict['conv_s_aux'] = nn.Conv3d(in_channels=mid_channels,
                                                out_channels=out_channels,
                                                kernel_size=[1, 1, 1],
                                                stride=[1, 1, 1],
                                                padding=[0, 0, 0],
                                                dilation=[1, 1, 1],
                                                groups=groups,
                                                bias=False)

        elif block_type == '3d':
            # build spatial convolution
            _dict['conv'] = nn.Conv3d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=not with_bn)
        else:
            raise NotImplementedError

        self.operations = nn.ModuleDict(_dict)
        if self.deepshare and block_type == '2.5d':
            del self.operations['conv_s_aux'].weight

    def _update_aux(self):
        if self.deepshare:
            self.operations['conv_s_aux'].weight = torch.mean(self.operations['conv_t'].weight, dim=2, keepdim=True)
        else:
            self.operations['conv_s_aux'].weight = torch.nn.Parameter(torch.mean(self.operations['conv_t'].weight, dim=2, keepdim=True))

    def forward(self, x, disable_temporal=False):
        if self.block_type == '3d':
            return self.operations['conv'](x)
        x = self.operations['conv_s'](x)
        x = self.operations['bn_s'](x)
        x = self.operations['relu_s'](x)
        if disable_temporal:
            self._update_aux()
            x = self.operations['conv_s_aux'](x)
        else:
            #x_aux = self.operations['conv_s_aux'](x)
            #x_aux = nn.functional.avg_pool3d(x_aux, kernel_size=(1, 1, 1), stride=(self.temporal_stride, 1, 1))
            x = self.operations['conv_t'](x)# + x_aux

        return x


class BasicBlock(nn.Module):

    def __init__(self,
                 block_type: str,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_channels: int,
                 down_sampling: bool = False,
                 down_sampling_temporal: bool = None,
                 with_bn: bool = True,
                 deepshare: bool = True):
        super(BasicBlock, self).__init__()

        self.with_bn = with_bn

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        if down_sampling:
            if down_sampling_temporal:
                stride = [2, 2, 2]
            else:
                stride = [1, 2, 2]
        elif down_sampling_temporal:
            stride = [2, 1, 1]
        else:
            stride = [1, 1, 1]

        self.conv1 = build_3d_conv(block_type=block_type,
                                   in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=[3, 3, 3],
                                   stride=stride,
                                   padding=[1, 1, 1],
                                   with_bn=with_bn,
                                   deepshare=deepshare)
        if self.with_bn:
            self.bn1 = nn.BatchNorm3d(out_channels, eps=1e-3)
        self.conv2 = build_3d_conv(block_type=block_type,
                                   in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=[3, 3, 3],
                                   stride=[1, 1, 1],
                                   padding=[1, 1, 1],
                                   with_bn=with_bn,
                                   deepshare=deepshare)
        if self.with_bn:
            self.bn2 = nn.BatchNorm3d(out_channels, eps=1e-3)
        if down_sampling or in_channels != out_channels:
            self.downsample = build_3d_conv(block_type='3d',
                                            in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=[1, 1, 1],
                                            stride=stride,
                                            padding=[0, 0, 0],
                                            with_bn=with_bn,
                                            deepshare=deepshare)
            if self.with_bn:
                self.downsample_bn = nn.BatchNorm3d(out_channels, eps=1e-3)
        else:
            self.downsample = None

        self.relu = nn.ReLU()

    def forward(self, x, disable_temporal=False):
        identity = x

        out = self.conv1(x, disable_temporal)
        if self.with_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out, disable_temporal)
        if self.with_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity, disable_temporal)
            if self.with_bn:
                identity = self.downsample_bn(identity)
        out = out + identity
        out = self.relu(out)
        return out


class BaseResNet3D(BaseBackbone):

    BLOCK_CONFIG = {
        10: (1, 1, 1, 1),
        16: (2, 2, 2, 1),
        18: (2, 2, 2, 2),
        26: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
        50: (3, 4, 6, 3),
        101: (3, 4, 23, 3),
        152: (3, 8, 36, 3),
    }

    SHALLOW_FILTER_CONFIG = [
        [64, 64],
        [128, 128],
        [256, 256],
        [512, 512]
    ]
    DEEP_FILTER_CONFIG = [
        [256, 64],
        [512, 128],
        [1024, 256],
        [2048, 512]
    ]

    def __init__(self,
                 block_type: str,
                 depth: int,
                 num_stages: int,
                 stem: dict,
                 down_sampling: List[bool],
                 channel_multiplier: int,
                 bottleneck_multiplier: int,
                 down_sampling_temporal: List[bool] = None,
                 with_bn: bool = True,
                 bn_eval: bool = False,
                 return_indices: List = None,
                 zero_init_residual: bool = False,
                 pretrained: str = None,
                 deepshare: bool = True):

        super(BaseResNet3D, self).__init__(bn_eval=bn_eval)
        self.pretrained = pretrained
        self.return_indices = return_indices
        self.zero_init_residual = zero_init_residual

        self.block_type = block_type
        self.depth = depth
        self.num_stages = num_stages
        self.deepshare = deepshare

        self.stem = self.build_stem_block(stem_type=block_type,
                                          with_bn=with_bn,
                                          deepshare=deepshare,
                                          **stem)

        if self.deepshare:
            del self.stem['conv_s_aux'].weight
        self.stem_type = block_type
        self.stem_with_pool = stem['with_pool']
        self.stem_temporal_stride = stem['temporal_stride']

        stage_blocks = self.BLOCK_CONFIG[depth]
        if self.depth <= 18 or self.depth == 34:
            block_constructor = BasicBlock
        else:
            raise NotImplementedError("BottleNeckBlock is not supported yet.")

        if self.depth <= 34:
            filter_config = self.SHALLOW_FILTER_CONFIG
        else:
            filter_config = self.DEEP_FILTER_CONFIG
        filter_config = np.multiply(filter_config, channel_multiplier).astype(np.int)

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling
        in_channels = 64
        for i in range(num_stages):
            layer = self.build_res_layer(block_constructor, block_type,
                                         num_blocks=stage_blocks[i],
                                         in_channels=in_channels,
                                         out_channels=int(filter_config[i][0]),
                                         bottleneck_channels=int(filter_config[i][1] * bottleneck_multiplier),
                                         down_sampling=down_sampling[i],
                                         down_sampling_temporal=down_sampling_temporal[i],
                                         with_bn=with_bn,
                                         deepshare=deepshare)
            self.add_module('layer{}'.format(i+1), layer)
            in_channels = int(filter_config[i][0])

    def _update_stem_aux(self):
        if self.deepshare:
            self.stem['conv_s_aux'].weight = torch.mean(self.stem['conv_t'].weight, dim=2, keepdim=True)
        else:
            self.stem['conv_s_aux'].weight = torch.nn.Parameter(torch.mean(self.stem['conv_t'].weight, dim=2, keepdim=True))

    def _forward_stem(self, x, disable_temporal = False):
        if self.stem_type == '2.5d':
            x = self.stem['conv_s'](x)
            x = self.stem['bn_s'](x)
            x = self.stem['relu_s'](x)
            if disable_temporal:
                self._update_stem_aux()
                x = self.stem['conv_s_aux'](x)
                x = self.stem['bn_s_aux'](x)
                x = self.stem['relu_s_aux'](x)
            else:
                #x_aux = self.stem['conv_s_aux'](x)
                #x_aux = self.stem['bn_s_aux'](x_aux)
                #x_aux = self.stem['relu_s_aux'](x_aux)
                x = self.stem['conv_t'](x)
                x = self.stem['bn_t'](x)
                x = self.stem['relu_t'](x)
                #x = x + nn.functional.avg_pool3d(x_aux, kernel_size=(1, 1, 1), stride=(self.stem_temporal_stride, 1, 1))
        else:
            x = self.stem['conv'](x)
            x = self.stem['bn'](x)
            x = self.stem['relu'](x)
        if self.stem_with_pool:
            x = self.stem['pool'](x)

        return x

    def forward(self, x, disable_temporal = False, freeze_front = False):
        x = self._forward_stem(x, disable_temporal)
        feats = [x]
        for i in range(self.num_stages):
            stage = getattr(self, 'layer{}'.format(i + 1))
            for idx, l in enumerate(stage):
                x = l(x, disable_temporal)
                # Freeze all blocks except for the last block at the last stage
                if freeze_front:
                    if i == self.num_stages - 1 and idx == len(stage) - 2:
                        x = x.detach()
            feats.append(x)
        if self.return_indices is None:
            return feats[-1]
        else:
            return [feats[k] for k in self.return_indices]

    def init_weights(self):
        logger = logging.getLogger()
        if isinstance(self.pretrained, str):
            self.init_from_pretrained(self.pretrained, logger)
        elif self.pretrained is None:
            logger.info("Random init backbone network")
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm3d):
                    constant_init(m, 1)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, BasicBlock):
                        constant_init(m.bn2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    @staticmethod
    def build_res_layer(block,
                        block_type: str,
                        num_blocks: int,
                        in_channels: int,
                        out_channels: int,
                        bottleneck_channels: int,
                        down_sampling: bool = False,
                        down_sampling_temporal: bool = None,
                        with_bn: bool = True,
                        deepshare: bool = True):
        layers = list()
        layers.append(block(
            block_type=block_type,
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_channels=bottleneck_channels,
            down_sampling=down_sampling,
            down_sampling_temporal=down_sampling_temporal,
            with_bn=with_bn,
            deepshare=deepshare
        ))
        for i in range(num_blocks-1):
            layers.append(block(
                block_type=block_type,
                in_channels=out_channels,
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels,
                down_sampling=False,
                down_sampling_temporal=None,
                with_bn=with_bn,
                deepshare=deepshare
            ))
        return nn.ModuleList(layers)

    @staticmethod
    def build_stem_block(stem_type: str,
                         temporal_kernel_size: int,
                         temporal_stride: int,
                         in_channels: int = 3,
                         with_bn: bool = True,
                         with_pool: bool = True,
                         deepshare: bool = True) -> nn.Sequential:
        _dict = OrderedDict()
        if stem_type == '2.5d':
            _dict['conv_s'] = nn.Conv3d(in_channels=in_channels,
                                        out_channels=45,
                                        kernel_size=(1, 7, 7),
                                        stride=[1, 2, 2],
                                        padding=[0, 3, 3],
                                        bias=not with_bn)
            if with_bn:
                _dict['bn_s'] = nn.BatchNorm3d(45, eps=1e-3)
            _dict['relu_s'] = nn.ReLU()
            _dict['conv_t'] = nn.Conv3d(in_channels=45,
                                        out_channels=64,
                                        kernel_size=(temporal_kernel_size, 1, 1),
                                        stride=[temporal_stride, 1, 1],
                                        padding=[(temporal_kernel_size-1)//2, 0, 0],
                                        bias=not with_bn)
            if with_bn:
                _dict['bn_t'] = nn.BatchNorm3d(64, eps=1e-3)
            _dict['relu_t'] = nn.ReLU()

            if deepshare:
                _dict['conv_s_aux'] = nn.Conv3d(in_channels=45,
                                                out_channels=64,
                                                kernel_size=(1, 1, 1),
                                                stride=[1, 1, 1],
                                                padding=[0, 0, 0],
                                                bias=not with_bn)
                if with_bn:
                    _dict['bn_s_aux'] = nn.BatchNorm3d(64, eps=1e-3)
                _dict['relu_s_aux'] = nn.ReLU()
        elif stem_type == '3d':
            _dict['conv'] = nn.Conv3d(in_channels=in_channels,
                                      out_channels=64,
                                      kernel_size=(temporal_kernel_size, 7, 7),
                                      stride=[temporal_stride, 2, 2],
                                      padding=[(temporal_kernel_size-1)//2, 3, 3],
                                      bias=not with_bn)
            if with_bn:
                _dict['bn'] = nn.BatchNorm3d(64, eps=1e-3)
            _dict['relu'] = nn.ReLU()
        else:
            raise NotImplementedError

        if with_pool:
            _dict['pool'] = nn.MaxPool3d(kernel_size=[1, 3, 3],
                                         stride=[1, 2, 2],
                                         padding=[0, 1, 1])

        return nn.ModuleDict(_dict)


@BACKBONES.register_module()
class R2Plus1D(BaseResNet3D):
    def __init__(self, *args, **kwargs):
        super(R2Plus1D, self).__init__(block_type='2.5d', *args, **kwargs)


@BACKBONES.register_module()
class R3D(BaseResNet3D):
    def __init__(self, *args, **kwargs):
        super(R3D, self).__init__(block_type='3d', *args, **kwargs)
