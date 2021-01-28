import cv2
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class TrackingDataset(Dataset):
    """ We use real tracking dataset, e.g. GOT-10k to perform TCP task. """
    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 frame_sampler: dict,
                 transform_cfg: list,
                 tracking_cfg: dict,
                 test_mode: bool = False):
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.frame_sampler = builder.build_frame_sampler(frame_sampler)

        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode

        self.z_size = tracking_cfg.get('z_size', 64)
        self.x_size = tracking_cfg.get('x_size', 128)
        self.shift_aug_ratio = tracking_cfg.get('shift_aug_ratio', 0.18)
        self.scale_aug_ratio = tracking_cfg.get('scale_aug_ratio', 0.18)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)
        assert len(storage_obj) == len(video_info['gt_boxes'])
        frame_inds = self.frame_sampler.sample(len(storage_obj))
        assert frame_inds.shape[0] == 1, "Support single clip only."
        frame_inds = frame_inds.reshape(-1).astype(np.long)
        img_list = storage_obj.get_frame(frame_inds)

        gt_info = np.array(video_info['gt_boxes'], dtype=np.int)
        gt_labels = gt_info[frame_inds, -1]
        gt_boxes = gt_info[frame_inds, 0:4].astype(np.float32)

        img_list, gt_boxes = self.crop_wrt_gt_boxes(img_list, gt_boxes)
        # for some bounding boxes that are out of scope, we set the weight to be zeros.
        gt_ctrs = (gt_boxes[..., 0:2] + gt_boxes[..., 2:4]) * 0.5
        is_in_scope = (gt_ctrs >= 0) & (gt_ctrs <= self.x_size)
        is_in_scope = np.all(is_in_scope, axis=-1)
        gt_weights = np.ones((len(gt_labels), ), np.float32)
        gt_weights[gt_labels > 0] = 0.
        gt_weights[~is_in_scope] = 0.

        img_tensor, trans_params = self.img_transform.apply_image(img_list, return_transform_param=True)
        gt_boxes = self.img_transform.apply_boxes(gt_boxes, trans_params)
        gt_trajs = torch.FloatTensor(gt_boxes).unsqueeze(0)  # [1, 16, 4]
        gt_weights = torch.FloatTensor(gt_weights).unsqueeze(0)  # [1, 16]
        img_tensor = img_tensor.permute(1, 0, 2, 3).contiguous()

        data = dict(
            imgs=DataContainer(img_tensor, stack=True, pad_dims=1, cpu_only=False),
            gt_trajs=DataContainer(gt_trajs, stack=True, pad_dims=1, cpu_only=False),
            gt_weights=DataContainer(gt_weights, stack=True, pad_dims=1, cpu_only=False),
        )
        storage_obj.close()

        return data

    def crop_wrt_gt_boxes(self, img_list, gt_boxes):
        """ Crop the search region according the the first ground-truth box.
        We follow the same strategy as the SiamRPN.

        Args:
            img_list (list[np.ndarray]): image list
            gt_boxes (np.ndarray): ground-truth bounding boxes, in shape of [len(img_list), 4]
        """
        # step 1, generate the search region box
        sr_box = self.get_search_region_box(gt_boxes[0], self.z_size, self.x_size)

        # step 2, in training, we need to apply data augmentation to this sr_box
        if not self.test_mode:
            sr_wh = sr_box[2:4] - sr_box[0:2]
            sr_ctr = (sr_box[2:4] + sr_box[0:2]) * 0.5

            max_shift_pixels = sr_wh[0] * self.shift_aug_ratio
            shift_pixels = np.random.uniform(-max_shift_pixels, max_shift_pixels, size=(2, ))
            scale_ratios = np.exp(np.random.uniform(-self.scale_aug_ratio, self.scale_aug_ratio, size=(2, )))

            sr_ctr = sr_ctr + shift_pixels
            sr_wh = sr_wh * scale_ratios

            sr_box = np.concatenate((sr_ctr - sr_wh * 0.5, sr_ctr + sr_wh * 0.5), axis=0)

        # step 3, crop search regions from the raw image list
        k = (self.x_size - 1.0) / (sr_box[2:4] - sr_box[0:2])
        warp_mat = np.array([[k[0], 0, -sr_box[0] * k[0]],
                             [0, k[1], -sr_box[1] * k[1]]], np.float32)
        crop_img_list = []

        border_mode = random.choice([cv2.BORDER_REFLECT, cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT])

        for img in img_list:
            crop_img = cv2.warpAffine(img, warp_mat, dsize=(self.x_size, self.x_size), borderMode=border_mode)
            crop_img_list.append(crop_img)

        # step 4, adjust the ground-truth bounding box
        top_left = np.concatenate((sr_box[0:2], sr_box[0:2]), axis=0).reshape(1, 4)
        scale_factors = self.x_size / (sr_box[2:4] - sr_box[0:2])
        scale_factors = np.concatenate((scale_factors, scale_factors), axis=0).reshape(1, 4)
        gt_boxes = (gt_boxes - top_left) * scale_factors
        return crop_img_list, gt_boxes

    @staticmethod
    def get_search_region_box(box, z_size, x_size, context_amount=0.5):
        """ Find where to crop according to the target box.
        Args:
            box (np.ndarray): box coordinates [x1, y1, x2, y2]
            z_size (int): template size
            x_size (int): search region size
            context_amount (float): context amount in the template image

        Returns:
            crop_box (torch.Tensor): in shape of [num_scales, 4], [xc, yc, w, h]
        """
        ctr = (box[2:4] + box[0:2]) * 0.5
        wh = box[2:4] - box[0:2]

        base_z_context_size = wh + context_amount * wh.sum()
        base_z_size = np.sqrt(base_z_context_size.prod())
        base_x_size = (x_size / z_size) * base_z_size

        crop_box = np.concatenate([ctr - base_x_size * 0.5, ctr + base_x_size * 0.5], axis=0)  # [4, ]
        return crop_box
