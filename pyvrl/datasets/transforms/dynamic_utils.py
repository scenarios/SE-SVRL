import numpy as np
from typing import Union, Tuple, List, Iterable


def sample_key_frames(num_frames: int, key_frame_probs: List[float]) -> np.ndarray:
    """ Sample the indices of key frames.
    Args:
        num_frames (int): number of frames in whole video
        key_frame_probs (List[float]): decide the probabilities of how many key frames.
    Returns:
        frame_inds (np.ndarray): key frame index, in range of [0, num_frames - 1]
    """
    # how many key frames
    num_key_frames = np.random.choice(len(key_frame_probs), p=key_frame_probs)
    # if there is no inner key frame, we will directly sample the first frame and the last frame.
    if num_key_frames == 0:
        return np.array([0, num_frames - 1], dtype=np.int)
    avg_duration = num_frames / (num_key_frames + 1)
    ticks = np.array([int(avg_duration * i) for i in range(1, num_key_frames + 1)], dtype=np.int)

    # add random jitter
    jitter_range = int(avg_duration / 3)
    if jitter_range > 0:
        jitter = np.random.randint(-jitter_range, jitter_range, size=len(ticks))
    else:
        jitter = np.zeros((len(ticks),), np.int)

    ticks = ticks + jitter
    # add the first frame and last frame
    ticks = np.concatenate((ticks, np.array([0, num_frames - 1])), axis=0)
    # remove duplication and sort array
    ticks = np.sort(np.unique(ticks))
    return ticks


def extend_key_frame_to_all(array: np.ndarray,
                            key_frame_inds: np.ndarray,
                            interpolate='uniform'):

    def _uniform_interpolate(start_state, end_state, index_delta):
        delta_state = (end_state - start_state) * (1.0 / index_delta)
        return np.concatenate([start_state + _ * delta_state for _ in range(index_delta+1)], axis=0)

    def _accelerate_interpolate(start_state, end_state, index_delta):
        a = 2 * (end_state - start_state) / (index_delta ** 2)
        return np.concatenate([start_state + 0.5 * a * (_**2) for _ in range(index_delta+1)], axis=0)

    def _decelerate_interpolate(start_state, end_state, index_delta):
        a = 2 * (start_state - end_state) / (index_delta ** 2)
        return np.concatenate([end_state + 0.5 * a * ((index_delta-_)**2) for _ in range(index_delta+1)], axis=0)

    assert key_frame_inds[0] == 0 and key_frame_inds[-1] > 0
    num_key_frames = len(key_frame_inds)
    assert num_key_frames == len(array)
    num_frames = key_frame_inds[-1] + 1

    out_array = np.zeros((num_frames, ) + array.shape[1:], dtype=array.dtype)
    for i in range(num_key_frames - 1):
        # fill the values between i -> i+1
        st_idx, end_idx = key_frame_inds[i:i+2]
        if interpolate == 'uniform':
            inter_func = _uniform_interpolate
        elif interpolate == 'accelerate':
            inter_func = _accelerate_interpolate
        elif interpolate == 'decelerate':
            inter_func = _decelerate_interpolate
        elif interpolate == 'random':
            inter_index = np.random.choice(3, p=[0.7, 0.15, 0.15])
            if inter_index == 0:
                inter_func = _uniform_interpolate
            elif inter_index == 1:
                inter_func = _accelerate_interpolate
            else:
                inter_func = _decelerate_interpolate
        else:
            raise NotImplementedError
        i_out = inter_func(array[i:i+1],
                           array[i+1:i+2],
                           end_idx - st_idx)
        out_array[st_idx:end_idx+1] = i_out

    return out_array
