import numpy as np
import random
from typing import List
from .base_transform import BaseTransform
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class GroupFlip(BaseTransform):

    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob

    def get_transform_param(self, data, *args, **kwargs):
        flag = random.uniform(0.0, 1.0) < self.flip_prob
        return dict(flag=flag)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        if transform_param['flag']:
            data = [self._flip_image(d) for d in data]
            transform_param['img_shape'] = data[0].shape
        return data

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        if transform_param['flag']:
            img_width = transform_param['img_shape'][1]
            transformed_boxes = []
            for i_boxes in boxes:
                i_trans_boxes = i_boxes.copy()
                i_trans_boxes[:, 0:4:2] = img_width - i_trans_boxes[:, [2, 0]]
                transformed_boxes.append(i_trans_boxes)
            return transformed_boxes
        else:
            return boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        if transform_param['flag']:
            transformed_flows = []
            for i_flow in flows:
                i_trans_flow = i_flow.copy()
                i_trans_flow[..., 0] = -i_trans_flow[..., 0]
                i_trans_flow = self._flip_image(i_trans_flow)
                transformed_flows.append(i_trans_flow)
            return transformed_flows
        else:
            return flows

    @staticmethod
    def _flip_image(img: np.ndarray):
        img = np.ascontiguousarray(img[:, ::-1, :])
        return img
