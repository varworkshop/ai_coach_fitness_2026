import os
import cv2
import glob
import argparse
import numpy as np

from tqdm import tqdm

from utils import load_video_timestamps

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos in a dataset.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to the dataset root directory")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract (default: 5)")
    parser.add_argument("--competition", action="store_true", help="Extract frames for the competition split")
    return parser.parse_args()

def rotate(frame: np.ndarray) -> np.ndarray:
    """Rotate frame by 90 degrees and square pad with black borders.

    :param frame: numpy.array
        Single frame to rotate

    :return: numpy.array
        Frame rotated 90 degrees clockwise and squared-padded
    """
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width, _ = frame.shape
    square_size = max(height, width)

    pad_top = int((square_size - frame.shape[0]) / 2)
    pad_bottom = square_size - frame.shape[0] - pad_top
    pad_left = int((square_size - frame.shape[1]) / 2)
    pad_right = square_size - frame.shape[1] - pad_left
    frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
    return frame

def process_video(filepath: str, fps: int = 2, is_competition_split: bool = False) -> np.ndarray:
    """Extract frames from video at a specified FPS rate.

    :param filepath: str
        Path to the video file to process.

    :param fps: int, optional
        Frames per second to extract from the video. Default is 2.

    :return: numpy.ndarray
        List of extracted frames as numpy arrays in RGB format.
    """
    video_ts = load_video_timestamps(
        file_path = filepath.replace(".mp4", "_timestamps.npy"),
        is_competition_split = is_competition_split
    )
    if not is_competition_split:
        video_ts_normalized = video_ts - video_ts[0]
    else:
        video_ts_normalized = video_ts

    vidcap = cv2.VideoCapture(filepath)
    assert vidcap.isOpened()

    file_idx = int(filepath.split("/")[-1].split(".")[0])

    index_in = -1
    frames = []
    next_frame_timestamp = 0
    while True:
        success = vidcap.grab()
        if not success:
            break
        index_in += 1
        if video_ts_normalized[index_in] >= next_frame_timestamp:
            success, frame = vidcap.retrieve()
            if not success:
                break

            # Ensure all videos are vertical
            if file_idx <= 145 and not is_competition_split:
                frame = rotate(frame)

            # Save to array
            frames.append(frame)
            
            # Update timestamp of next frame to extract
            next_frame_timestamp += 1.0 / fps

    return frames

def save_frames_to_folder(video_output_folder: str, frames: np.ndarray) -> None:
    """Save a list of frames as JPEG images to a specified folder.

    :param video_output_folder: str
        Path to the folder where frames will be saved.

    :param frames: numpy.ndarray
        List of frames to save, where each frame is a numpy array.

    :return: None
    """
    for frame_idx, frame in enumerate(frames):
        frame_filename = os.path.join(video_output_folder, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(frame_filename, frame)


def extract_frames_from_videos(
    input_folder: str, output_folder: str, fps: int, is_competition_split: bool = False
) -> None:
    """Extract frames from all videos in a folder and save them to an output folder.

    :param input_folder: str
        Path to the folder containing input video files.

    :param output_folder: str
        Path to the folder where extracted frames will be saved.

    :param fps: int
        Frames per second to extract from each video.

    :return: None
    """
    os.makedirs(output_folder, exist_ok=True)

    video_files = glob.glob(os.path.join(input_folder, "0*.mp4"))

    print(f"Staring frame extraction of {len(video_files)} videos in: {input_folder}")
    for video_path in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_folder = os.path.join(output_folder, video_name)
        os.makedirs(video_output_folder, exist_ok=True)

        frames = process_video(video_path, fps, is_competition_split)
        
        save_frames_to_folder(video_output_folder, frames)
        
        print(f"Extracted {len(frames)} frames from {video_name}.mp4 at {fps} fps")

    print("Frame extraction complete.")

def run_extraction(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root
    fps = args.fps
    is_competition_split = args.competition
    input_folder = f"{dataset_root}/long_range_videos"
    output_folder = f"{dataset_root}/long_range_video_frames_{fps}fps"
    extract_frames_from_videos(
        input_folder, output_folder, fps, is_competition_split
    )


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    run_extraction(args)
    
