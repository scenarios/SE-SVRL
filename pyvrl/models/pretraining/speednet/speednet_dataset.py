import random
import torch
from torch.utils.data import Dataset
from typing import List
from mmcv.parallel import DataContainer

from ....builder import DATASETS
from ....datasets.transforms import Compose
from ....datasets import builder


@DATASETS.register_module()
class SpeedNetDataset(Dataset):

    def __init__(self,
                 data_dir: str,
                 data_source: dict,
                 backend: dict,
                 clip_len: int,
                 clip_strides: List[int],
                 transform_cfg: list,
                 test_mode: bool = False):
        self.data_dir = data_dir
        self.data_source = builder.build_data_source(data_source, dict(data_dir=data_dir))
        self.backend = builder.build_backend(backend, dict(data_dir=data_dir))
        self.img_transform = Compose(transform_cfg)
        self.test_mode = test_mode
        self.clip_len = clip_len
        self.clip_strides = clip_strides

    def _sample_frame_inds(self, num_frames):
        stride_idx = random.choice(list(range(len(self.clip_strides))))
        stride = self.clip_strides[stride_idx]
        pos_stride = stride if stride >= 0 else -stride

        total_len = pos_stride * (self.clip_len - 1) + 1
        if total_len >= num_frames:
            start_index = 0
        else:
            start_index = random.randint(0, num_frames - total_len)
        frame_inds = [min(start_index + i * pos_stride, num_frames - 1) for i in range(self.clip_len)]

        if stride < 0:
            frame_inds = list(reversed(frame_inds))

        return frame_inds, stride_idx

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        video_info = self.data_source[idx]
        # build video storage backend object
        storage_obj = self.backend.open(video_info)

        imgs, stride_idx = self.sample_single_img_tensor(storage_obj)

        imgs = imgs.unsqueeze(0)
        gt_label = torch.LongTensor([stride_idx])
        data = dict(
            imgs=DataContainer(imgs, stack=True, pad_dims=1, cpu_only=False),
            gt_labels=DataContainer(gt_label, stack=True, pad_dims=None, cpu_only=False)
        )
        storage_obj.close()

        return data

    def sample_single_img_tensor(self, storage_obj):
        frame_inds, stride_idx = self._sample_frame_inds(len(storage_obj))
        frames = storage_obj.get_frame(frame_inds)
        # apply for transforms
        frames, trans_params = self.img_transform.apply_image(frames, return_transform_param=True)
        img_tensor = frames.permute(1, 0, 2, 3).contiguous()
        return img_tensor, stride_idx
