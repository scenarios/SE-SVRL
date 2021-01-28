import torch
import torch.nn as nn

from typing import Dict
from .tsn_modules import SimpleClsHead, SimpleSTModule
from ....apis import parse_losses
from ....builder import MODELS, build_backbone


@MODELS.register_module()
class TSN(nn.Module):

    def __init__(self,
                 backbone: dict,
                 st_module: dict,
                 cls_head: dict,
                 **kwargs):
        super(TSN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.st_module = SimpleSTModule(**st_module)
        self.cls_head = SimpleClsHead(**cls_head)
        self.init_weights()

        self.freeze_backbone = None
        self.non_linear = None
        if 'FreezeBackbone' in kwargs:
            self.freeze_backbone = kwargs['FreezeBackbone']
        if 'non_linear' in cls_head:
            self.non_linear = cls_head['non_linear']

    def init_weights(self):
        self.backbone.init_weights()
        if hasattr(self, 'st_module'):
            self.st_module.init_weights()
        if hasattr(self, 'seg_consensus'):
            self.seg_consensus.init_weights()
        if hasattr(self, 'cls_head'):
            self.cls_head.init_weights()

    def forward(self, return_loss=True, *args, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)

    def _forward(self, imgs: torch.Tensor):
        """ Predict the classification results.
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
        Returns:
            cls_logits (torch.Tensor): classification results, in shape of [N, M, num_class]
        """
        batch_size = imgs.size(0)
        num_segs = imgs.size(1)
        # unsqueeze the first dimension
        imgs = imgs.view((-1, ) + imgs.shape[2:])
        # backbone network
        feats = self.backbone(imgs, freeze_front=self.non_linear)  # [NM, C, T, H, W]
        if self.freeze_backbone and not self.non_linear:
            feats = feats.detach() # No update for the entire backbone

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if self.st_module is not None:
            feats = self.st_module(feats)  # [NM, C, 1, 1, 1]
        # feats = feats.view(batch_size * num_segs, -1)
        cls_logits = self.cls_head(feats)
        cls_logits = cls_logits.view(batch_size, num_segs, -1)
        return cls_logits

    def forward_train(self,
                      imgs: torch.Tensor,
                      gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """ Forward 3D-Net and then return the losses
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
            gt_labels (torch.Tensor): ground-truth label in shape of [N, 1]
        """

        cls_logits = self._forward(imgs)
        gt_labels = gt_labels.view(-1)
        losses = self.cls_head.loss(cls_logits, gt_labels)
        return losses

    def forward_test(self, imgs: torch.Tensor):
        """ Forward 3D-Net and then return the classification results
        Args:
            imgs (torch.Tensor): RGB image data in shape of [N, M, C, T, H, W]
        """
        cls_logits = self._forward(imgs)
        # average the classification logit
        cls_logits = cls_logits.mean(dim=1)
        cls_scores = torch.nn.functional.softmax(cls_logits, dim=1)
        cls_scores = cls_scores.cpu().numpy()
        return cls_scores

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['imgs']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['imgs']))

        return outputs
