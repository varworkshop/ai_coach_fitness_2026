from typing import AnyStr, List
import numpy as np
from PIL import Image

def load_video_timestamps(file_path: AnyStr, is_competition_split: bool = False) -> np.array:
    """
    Return array of timestamps per video frame (features)

    :param file_path:
        Path of timestamp file.

    :return:
        List of timestamps in UNIX format
    """
    # Transform timestamps to seconds (from nanoseconds)
    timestamps = np.load(file_path).astype(np.double)
    # the competition split already comes with normalized timestamps
    if not is_competition_split:
        timestamps = np.array([int(xx / 1e9) + (xx / 1e9) % 1 for xx in timestamps]) + 28800.0
    return timestamps

def load_frames_into_np_array(
    video_frame_paths: List[str], 
    resolution: int = 384, 
    width: int = None, 
    height: int = None
) -> np.ndarray:
    """
    Load video frames from file paths into a numpy array.

    :param video_frame_paths:
        List of file paths to video frames.
    :param resolution:
        Resolution to resize frames to (both width and height). Defaults to 384. Used when width or height are not specified.
    :param width:
        Width to resize frames to. Defaults to resolution if not specified.
    :param height:
        Height to resize frames to. Defaults to resolution if not specified.

    :return:
        Numpy array of shape (num_frames, height, width, channels) containing the video frames.
    """
    if width is None:
        width = resolution
    if height is None:
        height = resolution
    video = []
    for video_frame_path in video_frame_paths:
        im = Image.open(video_frame_path).resize((width, height), resample=0)
        video.append(np.array(im))
    return np.array(video)
