import torch
from torch import nn
from typing import Iterable

from mmcv.ops.roi_align import RoIAlign
from mmcv.cnn import kaiming_init, normal_init

from ....apis import parse_losses
from ....builder import build_backbone, MODELS


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)

    return output


class TCPHead(nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 in_temporal_size: int,
                 roi_feat_size: int,
                 spatial_stride: float,
                 num_pred_frames: int,
                 target_means: Iterable[float] = (0.0, 0.0, 0.0, 0.0),
                 target_stds: Iterable[float] = (1.0, 1.0, 1.0, 1.0)):
        """ Head of temporal correspondence prediction.
        Args:
            in_channels (int): channels in input features
            hidden_channels (int): channels in hidden layer
            in_temporal_size (int): temporal size of input features
            roi_feat_size (int): the output size of RoI Align Operation
            spatial_stride (float): total stride of the backbone network
            num_pred_frames (int): number of input (and predicted) frames
            target_means (Iterable[float]): predict target means
            target_stds (Iterable[float]): predict target stds, the final
                predict target will minus the mean values and divide the
                std values.
        """

        super(TCPHead, self).__init__()
        self.in_channels = in_channels
        self.in_temporal_size = in_temporal_size
        self.hidden_channels = hidden_channels
        self.num_pred_frames = num_pred_frames

        self.target_means = torch.FloatTensor(target_means)
        self.target_stds = torch.FloatTensor(target_stds)

        self.temporal_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=(in_temporal_size, 1, 1),
            padding=0,
            bias=True
        )

        self.roi_align = RoIAlign(roi_feat_size, 1.0 / spatial_stride, sampling_ratio=2, aligned=True)
        self.fc1 = nn.Linear(hidden_channels * (roi_feat_size ** 2), hidden_channels)
        self.pred_head = nn.Linear(hidden_channels, num_pred_frames * 4)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        kaiming_init(self.temporal_conv)
        normal_init(self.fc1, 0, std=0.001)
        normal_init(self.pred_head, 0, std=0.001)

    def forward(self, feats: torch.Tensor, rois: torch.Tensor):
        """ Predict object trajectory according to the RoIs
        Args:
            feats (torch.Tensor): input features, in shape of [B, C, T, H, W]
            rois (torch.Tensor): input rois, in shape of [N, 5]
        """
        batch_size, in_channels, in_temporal_size, h, w = feats.size()
        assert in_temporal_size == self.in_temporal_size and in_channels == self.in_channels
        # Step 1, squeeze the temporal dimension of input features
        feats = self.temporal_conv(feats).squeeze(2)
        feats = self.relu(feats)

        # Step 2, apply RoI Align operation
        roi_feats = self.roi_align(feats, rois)

        # Step 3, apply two-layer MLP: fc-relu-fc
        fc_feats = roi_feats.view(rois.size(0), -1)
        fc_feats = self.relu(self.fc1(fc_feats))
        preds = self.pred_head(fc_feats)

        return preds

    def loss(self,
             preds: torch.Tensor,
             rois: torch.Tensor,
             gt_trajs: torch.Tensor):
        """ Calculate the loss function.

        Args:
            preds (torch.Tensor): the output of head,
                in shape of [Num_img * Num_traj, Num_frames * 4]
            rois (torch.Tensor): the query rois,
                in shape of [Num_img * Num_traj, 5]
            gt_trajs (torch.Tensor): ground-truth trajectory,
                in shape of [Num_img, Num_traj, Num_frames, 4]
        """
        bs, nt, nf, _ = gt_trajs.size()

        rois = rois.view(bs, nt, 5)
        roi_centers = (rois[..., 1:3] + rois[..., 3:5]) * 0.5
        roi_sizes = torch.clamp((rois[..., 3:5] - rois[..., 1:3]), min=1)  # [bs, nt, 2]

        gt_centers = (gt_trajs[..., 2:4] + gt_trajs[..., 0:2]) * 0.5
        gt_sizes = torch.clamp((gt_trajs[..., 2:4] - gt_trajs[..., 0:2]), min=1)

        gt_ctr_deltas = gt_centers - roi_centers.unsqueeze(2)
        gt_size_deltas = torch.log(gt_sizes) - torch.log(roi_sizes).unsqueeze(2)

        preds = preds.view(bs, nt, nf, 4)
        gt_targets = torch.cat((gt_ctr_deltas, gt_size_deltas), dim=-1)
        target_means = self.target_means.view(1, 1, 1, 4).type_as(gt_targets)
        target_stds = self.target_stds.view(1, 1, 1, 4).type_as(gt_targets)
        gt_targets = (gt_targets - target_means) / target_stds

        diff = self.smooth_l1_loss(preds, gt_targets)

        count = bs * nt * nf
        loss_ctr = diff[..., 0:2].sum() / count
        loss_size = diff[..., 2:4].sum() / count

        losses = dict(loss_ctr=loss_ctr, loss_size=loss_size)

        return losses

    @staticmethod
    def smooth_l1_loss(pred, target, beta=1.0):
        assert beta > 0
        assert pred.size() == target.size() and target.numel() > 0
        diff = torch.abs(pred - target)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                           diff - 0.5 * beta)
        return loss


@MODELS.register_module()
class SRTCP(nn.Module):

    def __init__(self,
                 backbone: dict,
                 head: dict,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 queue_size: int = 65536,
                 momentum: float = 0.999,
                 temperature: float = 0.07,
                 mlp: bool = False,
                 ):
        super(SRTCP, self).__init__()
        self.backbone = build_backbone(backbone)
        self.head = TCPHead(**head)

        self.K = queue_size
        self.m = momentum
        self.T = temperature
        self.key_encoder = build_backbone(backbone)

        if mlp:
            self.q_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
            self.k_fc = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.q_fc = nn.Linear(in_channels, out_channels)
            self.k_fc = nn.Linear(in_channels, out_channels)

        # create the queue
        self.register_buffer("queue", torch.randn(out_channels, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.init_weights()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.backbone.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for fc_param_q, fc_param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            fc_param_k.data = fc_param_k.data * self.m + fc_param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def init_weights(self):
        self.backbone.init_weights()
        self.head.init_weights()

        # Moco Copy
        for param_q, param_k in zip(self.backbone.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for fc_param_q, fc_param_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
            fc_param_k.data.copy_(fc_param_q.data)  # initialize
            fc_param_k.requires_grad = False  # not update by gradient

    def forward(self,
                imgs: torch.Tensor,
                gt_trajs: torch.Tensor,
                img_q: torch.Tensor,
                img_k: torch.Tensor,):
        """ Receive video clips and ground-truth trajectory as inputs.
        This function will return the loss values (in as dictionary)

        Args:
            imgs (torch.Tensor): input video clips, in shape of [Num_imgs, 3, T, H, W]
            gt_trajs (torch.Tensor): rois, in shape of [Num_imgs, Num_trajs, Num_frames, 4]
                the order of the last dimension is (x1, y1, x2, y2)

        """

        ########################COMPUTE TCP LOSS ON SPATIAL AND TEMPORAL PART############################
        # Step 1, extract spatial-temporal features
        feats = self.backbone(imgs)

        # Step 2, pick the bounding box on the starting frame as query
        bs, nt, nf, _ = gt_trajs.size()  # [B, N_trajectory, N_frames, 4]
        roi_inds = torch.arange(bs).view(bs, 1).repeat(1, nt).view(bs * nt, 1).type_as(imgs)
        boxes = gt_trajs[:, :, 0, :].contiguous().view(bs * nt, 4)
        rois = torch.cat((roi_inds, boxes), dim=1)  # [bs*nt, 5]

        # Step 3, feed into the prediction head
        preds = self.head(feats, rois)

        # Step 4, calculate loss function
        tcp_losses = self.head.loss(preds, rois, gt_trajs)

        ########################COMPUTE SINGLE FRAME MOCO LOSS ON SPATIAL PART############################
        q = self.backbone(img_q, disable_temporal=True)  # [N, C, t, h, w]
        q = self.q_fc(q.view(q.shape[0:2] + (-1,)).mean(dim=-1))  # [N, out_channels]
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.key_encoder(img_k, disable_temporal=True)  # keys: NxC
            k = self.k_fc(k.view(k.shape[0:2] + (-1,)).mean(dim=-1))
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        nce_loss = nn.functional.cross_entropy(logits, labels)
        moco_losses = dict(loss_moco=nce_loss)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        losses = dict(list(tcp_losses.items()) + list(moco_losses.items()))

        return losses

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
