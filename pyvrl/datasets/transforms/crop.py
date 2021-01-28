import numpy as np
from typing import Union, Tuple, List, Iterable
import torch
from torchvision.transforms import RandomResizedCrop

from .base_transform import BaseTransform
from ...builder import TRANSFORMS


@TRANSFORMS.register_module()
class GroupRandomCrop(BaseTransform):

    def __init__(self,
                 out_size: Union[Tuple[int, int], int]):
        self.out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        if isinstance(data, list):
            img_h, img_w = data[0].shape[0:2]
        elif isinstance(data, np.ndarray):
            img_h, img_w = data.shape[0:2]
        else:
            raise TypeError("Unknown type {}".format(type(data)))
        delta_h = img_h - self.out_size[0]
        delta_w = img_w - self.out_size[1]
        assert delta_w >= 0 and delta_h >= 0
        y = int(np.random.randint(0, delta_h + 1))
        x = int(np.random.randint(0, delta_w + 1))
        return dict(img_shape=(img_h, img_w),
                    y=y, x=x)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        y, x = transform_param['y'], transform_param['x']
        data = [np.ascontiguousarray(img[y:y + self.out_size[0], x:x + self.out_size[1], :]) for img in data]
        return data

    def apply_boxes(self,
                    boxes: List[np.ndarray],
                    transform_param: dict):
        y, x = transform_param['y'], transform_param['x']
        delta = np.array([[x, y, x, y]], np.float32)
        transformed_boxes = []
        for i_boxes in boxes:
            i_trans_boxes = i_boxes.copy()
            i_trans_boxes[:, 0:4] = i_trans_boxes[:, 0:4] - delta
            transformed_boxes.append(i_trans_boxes)
        return transformed_boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        y, x = transform_param['y'], transform_param['x']
        flows = [np.ascontiguousarray(flow[y:y + self.out_size[0], x:x + self.out_size[1]]) for flow in flows]
        return flows


@TRANSFORMS.register_module()
class GroupCenterCrop(BaseTransform):

    def __init__(self,
                 out_size: Union[Tuple[int, int], int]):
        self.out_size = out_size if isinstance(out_size, (tuple, list)) else (out_size, out_size)

    def get_transform_param(self, data, *args, **kwargs):
        return dict(img_shape=data[0].shape)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict):
        data = [self.center_crop(d, self.out_size[0], self.out_size[1]) for d in data]
        return data

    def apply_boxes(self,
                    boxes: np.ndarray,
                    transform_param: dict):
        h, w = transform_param['img_shape'][0:2]
        y = (h - self.out_size[0]) // 2
        x = (w - self.out_size[1]) // 2
        delta = np.array([[x, y, x, y]], np.float32)
        transformed_boxes = []
        for i_boxes in boxes:
            i_trans_boxes = i_boxes.copy()
            i_trans_boxes[:, 0:4] = i_trans_boxes[:, 0:4] - delta
            transformed_boxes.append(i_trans_boxes)
        return transformed_boxes

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
        flows = [self.center_crop(d, self.out_size[0], self.out_size[1]) for d in flows]
        return flows

    @staticmethod
    def center_crop(t: np.ndarray,
                    crop_height: int,
                    crop_width: int):
        h, w, c = t.shape
        if h == crop_height and w == crop_width:
            return t
        sy = (h - crop_height) // 2
        ty = sy + crop_height
        sx = (w - crop_width) // 2
        tx = sx + crop_width
        crop_tensor = np.ascontiguousarray(t[sy:ty, sx:tx, ...])
        return crop_tensor

@TRANSFORMS.register_module()
class GroupRandomResizedCrop(BaseTransform):

    def __init__(self,
                 size: int,
                 scale=(0.08, 1.0),
                 ratio=(0.75, 1.3333333333333333),
                 interpolation=2):
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.proxy_transform = RandomResizedCrop(size=size, ratio=ratio, scale=scale, interpolation=interpolation)

    def get_transform_param(self, data, *args, **kwargs) -> dict:
        pass
        return dict(img_shape=(None, None),
                    y=None, x=None)

    def _apply_image(self,
                     data: List[np.ndarray],
                     transform_param: dict) -> List[np.ndarray]:
        """
        # Hacking the data format
        for i, img in enumerate(data):
            data[i] = np.transpose(img, (2, 0, 1))
        # data -> List[np.ndarray]
        # 1st choice, convert list of array to a single array piece at the first place
        # around 48 gpu days for 400 epochs
        data = torch.from_numpy(np.asarray(data))
        # 2nd choice, keep the list of array as it is before creating the tensor
        # around 500 gpu days for 400 epochs
        #data = torch.FloatTensor(data)

        data = self.proxy_transform(data).numpy()
        data = np.split(data, data.shape[0])

        # Hacking back the data format
        for i, img in enumerate(data):
            data[i] = np.transpose(np.squeeze(img, axis=0), (1, 2, 0))
        """
        return self.proxy_transform(data)

    def apply_boxes(self,
                    boxes: List[np.ndarray],
                    transform_param: dict):
        pass

    def apply_flow(self,
                   flows: List[np.ndarray],
                   transform_param: dict):
       pass
