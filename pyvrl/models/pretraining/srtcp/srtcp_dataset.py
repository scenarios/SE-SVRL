""" Temporal correspondence prediction for self-supervised video representation learning. """
import torch
import torchvision.transforms as torchvision_transforms
import cv2
import numpy as np

from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from PIL import Image

from ....builder import DATASETS
from ....datasets.transforms import Compose, single_frame_augmentation
from ....datasets import builder


@DATASETS.register_module()
class SRTCPDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 clip_sampler: dict,
                 single_frame_sampler:dict,
                 transform_cfg: list,
                 test_mode: bool = False):
        """ TCP dataset configurations.
        Args:
            data_source (dict): data source configuration dictionary
            data_dir (str): data root directory
            transform_cfg (list): data augmentation configuration list
            backend (dict): storage backend configuration
            test_mode (bool): placeholder, not available in TCP training.
        """
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.clip_sampler = builder.build_frame_sampler(clip_sampler)
        self.clip_transform = Compose(transform_cfg)
        self.single_frame_sampler = builder.build_frame_sampler(single_frame_sampler)
        self.single_frame_transform = torchvision_transforms.Compose(single_frame_augmentation)

        try:
            self.mask_trans_index = next(i for i, trans in enumerate(self.clip_transform.transforms)
                                         if trans.__class__.__name__ == 'PatchMask')
        except Exception:
            raise ValueError("cannot find PatchMask transformation in the data augmentation configurations.")
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)
        total_num_frames = len(storage_obj)
        assert total_num_frames > 0, "Bad data {}".format(video_info)
        frame_inds = self.clip_sampler.sample(total_num_frames)
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        frame_list = storage_obj.get_frame(frame_inds.reshape(-1))
        clip_tensor, trans_params = self.clip_transform.apply_image(frame_list, return_transform_param=True)
        clip_tensor = clip_tensor.permute(1, 0, 2, 3).contiguous()

        gt_trajs = torch.FloatTensor(trans_params[self.mask_trans_index]['traj_rois'])

        im_q, sampled_position = self.sample_single_img_tensor(storage_obj)
        im_k, _ = self.sample_single_img_tensor(storage_obj, position=sampled_position)

        data = dict(
            img_q=DataContainer(im_q, stack=True, pad_dims=1, cpu_only=False),
            img_k=DataContainer(im_k, stack=True, pad_dims=1, cpu_only=False),
            imgs=DataContainer(clip_tensor, stack=True, pad_dims=1, cpu_only=False),
            gt_trajs=DataContainer(gt_trajs, stack=True, pad_dims=1, cpu_only=False),
        )
        storage_obj.close()

        return data


    def sample_single_img_tensor(self, storage_obj, position = None):
        frame_ind = self.single_frame_sampler.sample_single(len(storage_obj), position)
        frame = storage_obj.get_frame(frame_ind)
        assert len(frame) == 1, "sample single frame only"
        # apply for transforms
        frame = cv2.cvtColor(frame[0], cv2.COLOR_BGR2RGB)
        _ = type(frame)
        frame = Image.fromarray(frame, mode='RGB')
        frame = self.single_frame_transform(frame)
        img_tensor = torch.unsqueeze(frame, dim=0).permute(1, 0, 2, 3).contiguous()
        return img_tensor, frame_ind[0]