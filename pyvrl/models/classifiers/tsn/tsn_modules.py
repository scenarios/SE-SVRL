import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSTModule(nn.Module):

    def __init__(self,
                 spatial_type: str = 'avg',
                 spatial_size: int = 7,
                 temporal_size: int = 1):
        super(SimpleSTModule, self).__init__()

        assert spatial_type in ['avg', 'max']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size
        if spatial_size != -1:
            self.spatial_size = (spatial_size, spatial_size)

        self.temporal_size = temporal_size

        assert not (self.spatial_size == -1) ^ (self.temporal_size == -1)

        if self.temporal_size == -1 and self.spatial_size == -1:
            self.pool_size = (1, 1, 1)
            if self.spatial_type == 'avg':
                self.pool_func = nn.AdaptiveAvgPool3d(self.pool_size)
            if self.spatial_type == 'max':
                self.pool_func = nn.AdaptiveMaxPool3d(self.pool_size)
        else:
            self.pool_size = (self.temporal_size,) + self.spatial_size
            if self.spatial_type == 'avg':
                self.pool_func = nn.AvgPool3d(self.pool_size, stride=1, padding=0)
            if self.spatial_type == 'max':
                self.pool_func = nn.MaxPool3d(self.pool_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, data):
        return self.pool_func(data)


class _SimpleConsensus(torch.autograd.Function):
    """Simplest segmental consensus module"""

    @staticmethod
    def forward(ctx, x, dim=1):
        ctx.shape = x.size()
        ctx.dim = dim
        output = x.mean(dim=ctx.dim, keepdim=True)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        return grad_in, None


class SimpleConsensus(nn.Module):
    def __init__(self, dim=1):
        super(SimpleConsensus, self).__init__()
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, data):
        return _SimpleConsensus.apply(data, self.dim)


class SimpleClsHead(nn.Module):

    def __init__(self,
                 with_avg_pool=True,
                 temporal_feature_size=1,
                 spatial_feature_size=7,
                 dropout_ratio=0.8,
                 in_channels=2048,
                 num_classes=101,
                 init_std=0.01,
                 non_linear = False,
                 nonlinear_channels = 2048):
        super(SimpleClsHead, self).__init__()

        self.with_avg_pool = with_avg_pool
        self.dropout_ratio = dropout_ratio
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        self.temporal_feature_size = temporal_feature_size
        self.spatial_feature_size = spatial_feature_size
        self.init_std = init_std
        self.num_classes = num_classes
        self.non_linear = non_linear
        self.nonlinear_channels = nonlinear_channels

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool3d((temporal_feature_size, spatial_feature_size, spatial_feature_size))

        if self.non_linear:
            self.fc_nl = nn.Sequential(
                nn.Identity()
                #nn.Linear(in_channels, nonlinear_channels),
                #nn.ReLU()
                #nn.Dropout(p=self.dropout_ratio),
                #nn.Linear(nonlinear_channels, nonlinear_channels),
                #nn.ReLU()
            )
        self.fc_cls = nn.Linear(nonlinear_channels if self.non_linear else in_channels, num_classes)

    def init_weights(self):
        if self.non_linear:
            for l in self.fc_nl:
                if isinstance(l, nn.Linear):
                    nn.init.normal_(l.weight, 0, self.init_std)
                    nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc_cls.weight, 0, self.init_std)
        nn.init.constant_(self.fc_cls.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            x = x.unsqueeze(2)
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == self.temporal_feature_size
        assert x.shape[3] == self.spatial_feature_size
        assert x.shape[4] == self.spatial_feature_size
        if self.with_avg_pool:
            x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        if self.non_linear:
            x = self.fc_nl(x)
        cls_logits = self.fc_cls(x)
        return cls_logits

    def loss(self,
             cls_logits,
             labels):
        if cls_logits.dim() == 3 and labels.dim() == 1:
            batch_size, num_segs, _ = cls_logits.size()
            assert batch_size == labels.size(0)
            labels = labels.view(batch_size, 1).repeat([1, num_segs]).contiguous()
            labels = labels.view(-1)
            cls_logits = cls_logits.view(batch_size * num_segs, -1)

        losses = dict()
        losses['loss_cls'] = F.cross_entropy(cls_logits, labels)

        with torch.no_grad():
            max_index = cls_logits.max(dim=1)[1]
            correct = (labels.eq(max_index.long())).sum()
            accuracy = correct.float() / cls_logits.size(0)
            losses['acc'] = accuracy
        return losses
