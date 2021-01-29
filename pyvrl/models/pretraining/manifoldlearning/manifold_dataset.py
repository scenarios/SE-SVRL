""" Temporal correspondence prediction for self-supervised video representation learning. """
import torch
import torchvision.transforms as torchvision_transforms
import random

import numpy as np

from torch.utils.data import Dataset
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ....datasets.transforms import Compose, single_frame_augmentation
from ....datasets import builder


@DATASETS.register_module()
class MANIFOLDDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 clip_sampler: dict,
                 single_frame_sampler:dict,
                 tcp_transform_cfg: list,
                 manifold_transform_cfg: list,
                 test_mode: bool = False):
        """ MANIFOLD dataset configurations.
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
        self.tcp_clip_transform = Compose(tcp_transform_cfg)
        self.manifold_clip_transform = Compose(manifold_transform_cfg)
        self.single_frame_sampler = builder.build_frame_sampler(single_frame_sampler)
        self.single_frame_transform = torchvision_transforms.Compose(single_frame_augmentation)

        try:
            self.mask_trans_index = next(i for i, trans in enumerate(self.tcp_clip_transform.transforms)
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

        clip_tensor_with_transParam, clip_q, clip_k_p, clip_k_n, dataIndex = self.sample_query_key(storage_obj)
        clip_tensor, clip_tensor_trans_param = clip_tensor_with_transParam
        gt_trajs = torch.FloatTensor(clip_tensor_trans_param[self.mask_trans_index]['traj_rois'])

        data = dict(
            clip_q=DataContainer(clip_q, stack=True, pad_dims=1, cpu_only=False),
            clip_k_p=DataContainer(clip_k_p, stack=True, pad_dims=1, cpu_only=False),
            clip_k_n=DataContainer(clip_k_n, stack=True, pad_dims=1, cpu_only=False),
            dataIndex = DataContainer(dataIndex, stack=True, pad_dims=None, cpu_only=False),
            imgs=DataContainer(clip_tensor, stack=True, pad_dims=1, cpu_only=False),
            gt_trajs=DataContainer(gt_trajs, stack=True, pad_dims=1, cpu_only=False),
        )
        storage_obj.close()

        return data

    def sample_query_key(self, storage_obj):
        # TCP clip index
        total_num_frames = len(storage_obj)
        frame_inds = self.clip_sampler.sample(total_num_frames)
        num_segs, clip_len = frame_inds.shape
        assert num_segs == 1
        frame_inds = frame_inds.reshape(-1)

        # query clip index
        frame_q_inds = np.array([self.clip_sampler.sample_single(len(storage_obj), position=None)], np.int).reshape(-1)

        # key clip index
        isOverlap = random.uniform(0.0, 1.0) < 0.7
        frame_k1_inds = np.array(
            [self.clip_sampler.sample_single(len(storage_obj), position=frame_q_inds[0] if isOverlap else None)],
            np.int).reshape(-1)
        frame_k2_inds = np.array(
            [self.clip_sampler.sample_single(len(storage_obj), position=frame_q_inds[0] if isOverlap else None)],
            np.int).reshape(-1)

        clips_inds = [list(frame_inds), list(frame_q_inds), list(frame_k1_inds), list(frame_k2_inds)]
        clips, dataIndex = storage_obj.get_frame_with_index(clips_inds)
        frame_list, frame_q_list, frame_k1_list, frame_k2_list = clips
        dataIndex = torch.IntTensor(dataIndex)
        clip_tensor, trans_params = self.tcp_clip_transform.apply_image(frame_list, return_transform_param=True)

        clip_q_tensor, clip_q_trans_params = self.manifold_clip_transform.apply_image(frame_q_list,
                                                                             return_transform_param=True)

        clip_k1_tensor, clip_k1_trans_params = self.manifold_clip_transform.apply_image(frame_k1_list,
                                                                               return_transform_param=True)

        clip_k2_tensor, clip_k2_trans_params = self.manifold_clip_transform.apply_image(frame_k2_list,
                                                                               return_transform_param=True)

        clip_tensor = clip_tensor.permute(1, 0, 2, 3).contiguous()
        clip_q_tensor = clip_q_tensor.permute(1, 0, 2, 3).contiguous()
        clip_k1_tensor = clip_k1_tensor.permute(1, 0, 2, 3).contiguous()
        clip_k2_tensor = clip_k2_tensor.permute(1, 0, 2, 3).contiguous()

        # select close key & far key
        if abs(frame_k1_inds[0] - frame_q_inds[0]) < abs(frame_k2_inds[0] - frame_q_inds[0]):
            clip_k_positive_tensor = clip_k1_tensor
            clip_k_negtive_tensor = clip_k2_tensor
        else:
            clip_k_positive_tensor = clip_k2_tensor
            clip_k_negtive_tensor = clip_k1_tensor


        return (clip_tensor, trans_params), clip_q_tensor, clip_k_positive_tensor, clip_k_negtive_tensor, dataIndex