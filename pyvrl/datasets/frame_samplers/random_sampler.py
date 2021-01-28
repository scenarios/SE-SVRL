import numpy as np
import random
from typing import List, Union


class RandomFrameSampler(object):

    def __init__(self,
                 num_clips: int,
                 clip_len: int,
                 strides: Union[int, List[int]],
                 temporal_jitter: bool,
                 sample_range = 7):
        self.num_clips = num_clips
        self.clip_len = clip_len
        if isinstance(strides, (tuple, list)):
            self.strides = strides
        else:
            self.strides = [strides]
        self.temporal_jitter = temporal_jitter
        self.base_sample_range = 2 * clip_len // 3

    def sample_single(self, num_frames: int, position=None):
        stride = random.choice(self.strides)
        if stride > 1 and self.temporal_jitter:
            index_jitter = [random.randint(0, stride) for i in range(self.clip_len)]
            this_sample_range = int(self.base_sample_range * stride * 1.5)
        else:
            index_jitter = [0 for i in range(self.clip_len)]
            this_sample_range = self.base_sample_range * stride * 1

        total_len = stride * (self.clip_len - 1) + 1
        if total_len >= num_frames:
            start_index = 0
        else:
            if position is not None:
                start_index = position + random.randint(-this_sample_range, this_sample_range)
                start_index = min(num_frames - 1, max(0, start_index))
            else:
                start_index = random.randint(0, num_frames - total_len)
        frame_inds = [min(start_index + i * stride + index_jitter[i], num_frames-1)
                      for i in range(self.clip_len)]
        return frame_inds

    def sample(self, num_frames: int):
        return np.array([self.sample_single(num_frames) for _ in range(self.num_clips)], np.int)
