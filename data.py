from typing import List, Tuple, Dict
import os
import json
import numpy as np

from torch.utils.data import Dataset
from tqdm import tqdm

from utils import load_video_timestamps

class FitCoachDatset(Dataset):
    """
    A PyTorch Dataset class for the FitCoach benchmark dataset.

    This dataset handles long-range fitness coaching videos that are split into
    mini episodes, each corresponding to a specific exercise. It manages video
    frame timestamps, feedback annotations, and optional video frame paths.

    Attributes:
        data_root (str): Root directory of the dataset.
        model_fps (int): Frames per second used by the model.
        mini_episode_data (List[Dict]): List of dictionaries containing data for each mini episode.
        video_frame_timestamps_data (Dict): Dictionary mapping video IDs to their frame timestamps.
        video_frames_cache (Dict): Cache for video frame paths (populated if load_video_frame_paths=True).
    """

    def __init__(
        self, 
        data_root: str, 
        model_fps: int = 2,
        load_video_frame_paths: bool = False,
        remove_transition_feedbacks: bool = True
    ) -> None:
        """
        Initialize the FitCoachDataset.

        Args:
            data_root (str): Root directory containing the dataset files, including
                             the annotation JSON and video frame directories.
            model_fps (int): Frames per second that the model operates at. Used to
                             align video frame paths with model input requirements.
            load_video_frame_paths (bool, optional): Whether to load and cache video
                             frame file paths. Not needed for evaluation. Defaults to False.
        """
        self.data_root = data_root
        self.model_fps = model_fps
        # transition feedbacks need to removed for evaluation
        self.remove_transition_feedbacks = remove_transition_feedbacks

        annotation_file_path = os.path.join(data_root,"feedbacks_long_range.json")
        if "competition" in self.data_root.lower():
            self.load_annotations_competition(annotation_file_path)
        else:
            self.load_annotations(annotation_file_path)
        
        # not needed for evaluation
        if load_video_frame_paths:
            video_frames_root = os.path.join(data_root, f"long_range_video_frames_{self.model_fps}fps")
            assert os.path.exists(video_frames_root), f"Video frames extracted at the required fps does not exist: {video_frames_root}"
            self.load_video_frame_paths(video_frames_root)

    def extract_video_id(self, long_range_video_file: str) -> str:
        """
        Extract the video ID from a video file path by removing the file extension
        and directory components.

        Args:
            long_range_video_file (str): Full path or filename of the video file.

        Returns:
            str: The video ID, which is the filename without its extension.
        """
        return os.path.splitext(os.path.basename(long_range_video_file))[0]
    
    def extract_exercise_names(
        self, 
        feedbacks: List[str], 
        is_transition: List[bool]
    ) -> List[str]:
        """
        Extract exercise names from transition feedbacks in a video.
        The first feedback in each mini episode is a transition feedback that contains
        the exercise name. This method parses those transition feedbacks to extract
        clean exercise names by removing common prefix phrases.
        Note: The last feedback marks the end of the video and is excluded.

        Args:
            feedbacks (List[str]): List of all feedback strings in the video.
            is_transition (List[bool]): Boolean list indicating which feedbacks are
                                        transition feedbacks (marking new exercises).

        Returns:
            List[str]: List of exercise names extracted from transition feedbacks,
                       with introductory phrases removed.
        """
        # the first feedback in each mini episode contains the exercise name
        # (this feedback is not used for evaluation)
        exercise_names = []
        # last feedback marks the end of the video
        is_transition_idxs = np.where(np.array(is_transition))[0][:-1]
        for is_transition_idx in is_transition_idxs:
            transition_feedback = feedbacks[is_transition_idx]
            exercise_name = (
                transition_feedback.replace("First up are ", "")
                .replace("Moving on to ", "")
                .replace("!", "")
            )
            exercise_names.append(exercise_name)
        return exercise_names

    def collapse_feedbacks_and_get_feedback_timestamps(
        self, 
        frame_aligned_feedbacks: List[str], 
        video_frame_timestamps: List[float]
    ) -> Tuple[List[str],List[float]]:
        """
        Collapse frame-aligned feedbacks into unique feedbacks with their first occurrence timestamps.

        Since feedbacks are aligned per frame, the same feedback string may appear across
        multiple consecutive frames. This method deduplicates consecutive identical feedbacks
        and records the timestamp of their first occurrence.

        Args:
            frame_aligned_feedbacks (List[str]): List of feedback strings aligned to each
                                                  video frame, where the same feedback may
                                                  repeat across consecutive frames.
            video_frame_timestamps (List[float]): List of timestamps (in seconds) corresponding
                                                   to each video frame.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - feedbacks (List[str]): Deduplicated list of unique consecutive feedbacks.
                - feedback_timestamps (List[float]): Timestamps of the first occurrence of
                                                      each unique feedback.
        """
        feedbacks, feedback_timestamps = [], []
        last_feedback = None
        for frame_aligned_feedback, video_frame_timestamp in \
            zip(frame_aligned_feedbacks,video_frame_timestamps):
            if len(frame_aligned_feedback) > 0 and frame_aligned_feedback != last_feedback:
                feedbacks.append(frame_aligned_feedback)
                feedback_timestamps.append(video_frame_timestamp)
                last_feedback = frame_aligned_feedback
        return feedbacks, feedback_timestamps

    def split_feedbacks_into_mini_episodes(
        self, 
        feedbacks: List[str], 
        feedback_timestamps: List[float], 
        is_transition: List[bool]
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Split feedbacks and their timestamps into mini episodes based on transition markers.

        Each mini episode corresponds to a single exercise segment. Transitions mark the
        boundaries between episodes. The last mini episode is handled specially - it extends
        until the end of the video, and its end timestamp is returned separately.

        Args:
            feedbacks (List[str]): List of deduplicated feedback strings.
            feedback_timestamps (List[float]): List of timestamps corresponding to each feedback.
            is_transition (List[bool]): Boolean list indicating which feedbacks are transitions
                                         (i.e., mark the start of a new mini episode).

        Returns:
            Tuple[List[List[str]], List[List[float]], float]: A tuple containing:
                - mini_episode_feedbacks (List[List[str]]): Feedbacks grouped by mini episode.
                - mini_episode_feedback_timestamps (List[List[float]]): Timestamps grouped
                                                                         by mini episode.
                - last_mini_episode_end_timestamp (float): The end timestamp of the last
                                                            mini episode (from the final
                                                            transition marker).
        """
        split_idxs = np.where(is_transition)[0]

        # Add start and end indices for slicing
        slice_boundaries = split_idxs.tolist()[1:] + [len(feedbacks)]

        # init ds
        mini_episode_feedbacks = []
        mini_episode_feedback_timestamps = []
        start_idx = 0

        # split based on boundaries
        for end_idx in slice_boundaries:
            mini_episode_feedbacks.append(feedbacks[start_idx:end_idx])
            mini_episode_feedback_timestamps.append(feedback_timestamps[start_idx:end_idx])
            start_idx = end_idx

        # Keep rolling the last mini episode till the end of the video
        assert len(mini_episode_feedback_timestamps[-1]) == 1
        last_mini_episode_end_timestamp = mini_episode_feedback_timestamps[-1][0]

        # remove the last session end feedback
        mini_episode_feedbacks = mini_episode_feedbacks[:-1]
        mini_episode_feedback_timestamps = mini_episode_feedback_timestamps[:-1]

        return mini_episode_feedbacks, mini_episode_feedback_timestamps, last_mini_episode_end_timestamp

    def get_video_frame_timestamps_per_mini_episodes(
        self, 
        video_frame_timestamps: List[float], 
        mini_episode_feedback_timestamps: List[float],
        last_mini_episode_end_timestamp: float
    ) -> List[List[float]]:
        """
        Get the video frame timestamps that belong to each mini episode.
        Each mini episode spans from its first transition feedback timestamp to the
        start of the next mini episode (or the end of the video for the last episode).
        This method filters the full video frame timestamps to only include those
        within each mini episode's time range.

        Args:
            video_frame_timestamps (List[float]): Array of all video frame timestamps.
            mini_episode_feedback_timestamps (List[List[float]]): List of feedback timestamp
                                                                   lists, one per mini episode.
            last_mini_episode_end_timestamp (float): The end timestamp for the last mini
                                                      episode, used to bound the final segment.

        Returns:
            List[List[float]]: A list of timestamp lists, where each inner list contains
                                the video frame timestamps belonging to the corresponding
                                mini episode.
        """
        # each mini episode starts at the first (transition) feedback
        split_timestamps = [_timestamps[0] for _timestamps in mini_episode_feedback_timestamps]
        split_timestamps_begin_end = [[t_1, t_2] for t_1, t_2 in zip(split_timestamps[:-1],split_timestamps[1:])]
        # Keep rolling the last mini episode till the end of the video
        split_timestamps_begin_end += [[split_timestamps[-1],last_mini_episode_end_timestamp]]
        
        # init ds
        mini_episode_video_frame_timestamps = []
        # split based on boundaries
        for t_start, t_end in split_timestamps_begin_end:
            _timestamp_idxs = np.where(
                np.logical_and(
                    video_frame_timestamps >= t_start,
                    video_frame_timestamps <=t_end
                )
            )[0]
            mini_episode_video_frame_timestamps.append(
                video_frame_timestamps[_timestamp_idxs].tolist()
            )
        return mini_episode_video_frame_timestamps

    def get_video_frame_timestamps_per_mini_episodes_competition(
        self, 
        video_frame_timestamps: List[float], 
        mini_episode_start_timestamps: List[float], 
        mini_episode_end_timestamps: List[float],
        video_end_timestamp: float
    ) -> List[List[float]]:
        """
        Get the video frame timestamps that belong to each mini episode for the competition split.

        Unlike the standard split method, this version uses explicit start and end timestamps
        provided per mini episode rather than deriving boundaries from transition feedbacks.
        The last mini episode is extended to the end of the video.
        Args:
            video_frame_timestamps (List[float]): Array of all video frame timestamps.
            mini_episode_start_timestamps (List[float]): List of start timestamps for each
                                                       mini episode.
            mini_episode_end_timestamps (List[float]): List of end timestamps for each
                                                        mini episode. The last entry will be
                                                        overwritten with video_end_timestamp.
            video_end_timestamp (float): The timestamp of the last frame in the video, used
                                          to extend the final mini episode to the end of the video.

        Returns:
            List[List[float]]: A list of timestamp lists, where each inner list contains
                                the video frame timestamps belonging to the corresponding
                                mini episode.
        """
        # split mini episodes using mini_episode_start_timestamps and mini_episode_end_timestamps
        # roll last mini episode till end of video
        mini_episode_end_timestamps[-1] = video_end_timestamp
        split_timestamps_begin_end = [
            [t_1, t_2] 
            for t_1, t_2 in zip(mini_episode_start_timestamps,mini_episode_end_timestamps)
        ]

        # init ds
        mini_episode_video_frame_timestamps = []
        # split based on boundaries
        for t_start, t_end in split_timestamps_begin_end:
            _timestamp_idxs = np.where(
                np.logical_and(
                    video_frame_timestamps >= t_start,
                    video_frame_timestamps <=t_end
                )
            )[0]
            mini_episode_video_frame_timestamps.append(
                video_frame_timestamps[_timestamp_idxs].tolist()
            )
        return mini_episode_video_frame_timestamps

    def load_annotations(self, annotation_file_path: str) -> None:
        """
        Load and process annotations from the JSON annotation file.

        This method reads the annotation file and processes each video's annotations
        to populate `self.mini_episode_data` with per-mini-episode information including
        exercise names, feedbacks, feedback timestamps, and video frame timestamps.
        It also populates `self.video_frame_timestamps_data` for later use when loading
        video frame paths.

        The processing pipeline for each video:
            1. Load frame-aligned feedbacks and video frame timestamps.
            2. Collapse frame-aligned feedbacks to unique feedbacks with timestamps.
            3. Extract exercise names from transition feedbacks.
            4. Split feedbacks into mini episodes based on transition markers.
            5. Get video frame timestamps per mini episode.
            6. Store all data per mini episode in `self.mini_episode_data`.

        Args:
            annotation_file_path (str): Path to the JSON annotation file containing
                                         feedback and timestamp data for all videos.

        Returns:
            None: Populates `self.mini_episode_data` and `self.video_frame_timestamps_data`
                  as side effects.
        """
        with open(annotation_file_path, 'r') as file:
            annotations = json.load(file)

        self.mini_episode_data = []
        # save the video frame timestamps data for loading video frames
        self.video_frame_timestamps_data = {}
        for annotation in tqdm(annotations, desc="Loading annotations ..."):
            # load data from json
            long_range_video_file = annotation['long_range_video_file']
            frame_aligned_feedbacks = annotation['feedbacks']
            is_transition = annotation['is_transition']

            # get video id
            video_id = self.extract_video_id(long_range_video_file)

            # load video frame timestamps, the json stores only the file name
            video_frame_timestamps_path = os.path.join(
                self.data_root,
                annotation['video_timestamps']
            )
            video_frame_timestamps = load_video_timestamps(video_frame_timestamps_path)

            # save for future use while loading video frame paths
            self.video_frame_timestamps_data[video_id] = video_frame_timestamps

            # the json stores feedbacks aligned per frame
            # the display duration can be ignored as it is not part of the evaluation
            # (can be derived independently based on speech speed if necessary)
            feedbacks, feedback_timestamps = self.collapse_feedbacks_and_get_feedback_timestamps(
                frame_aligned_feedbacks,
                video_frame_timestamps.tolist()
            )

            # get names of exercise per mini episode
            mini_episode_exercise_names = self.extract_exercise_names(
                feedbacks,
                is_transition
            )
            
            # split feedbacks per mini episode
            mini_episode_feedbacks, mini_episode_feedback_timestamps, last_mini_episode_end_timestamp = \
            self.split_feedbacks_into_mini_episodes(
                feedbacks, 
                feedback_timestamps, 
                is_transition
            )

            # get timestamps of video frames corresponding to each mini episode
            mini_episode_video_frame_timestamps = self.get_video_frame_timestamps_per_mini_episodes(
                video_frame_timestamps,
                mini_episode_feedback_timestamps,
                last_mini_episode_end_timestamp
            )
            
            # store data per mini episode
            num_mini_episodes = len(mini_episode_feedbacks)
            for idx in range(num_mini_episodes):
                _mini_episode_exercise_name = mini_episode_exercise_names[idx]
                _mini_episode_feedbacks = mini_episode_feedbacks[idx]
                _mini_episode_feedback_timestamps = mini_episode_feedback_timestamps[idx]
                _mini_episode_video_frame_timestamps = mini_episode_video_frame_timestamps[idx]
                _mini_episode_id = f"{video_id}_{idx}"

                if self.remove_transition_feedbacks:
                    # the first feedback is a transition feedback
                    _mini_episode_feedbacks = _mini_episode_feedbacks[1:]
                    _mini_episode_feedback_timestamps = _mini_episode_feedback_timestamps[1:]
                
                # do not load empty mini episodes
                if len(_mini_episode_feedbacks) == 0 and \
                    len(_mini_episode_feedback_timestamps) == 0:
                    continue

                self.mini_episode_data.append({
                    "video_id": video_id,
                    "mini_episode_id": _mini_episode_id,
                    "mini_episode_exercise_name": _mini_episode_exercise_name,
                    "mini_episode_feedbacks": _mini_episode_feedbacks,
                    "mini_episode_feedback_timestamps": _mini_episode_feedback_timestamps,
                    "mini_episode_video_frame_timestamps": _mini_episode_video_frame_timestamps,
                })

        print("Done!")

    def load_annotations_competition(self, annotation_file_path: str) -> None:
        """
        Load and process annotations from the JSON annotation file for the competition split.

        Unlike the standard annotation loader, this method handles competition-format annotations
        where mini episode boundaries are provided as explicit start and end timestamps rather
        than being derived from transition feedbacks. Exercise names are also provided directly
        in the annotation file.

        This method populates `self.mini_episode_data` with per-mini-episode information including
        exercise names and video frame timestamps. It also populates `self.video_frame_timestamps_data`
        for later use when loading video frame paths.

        The processing pipeline for each video:
            1. Load exercise names and video frame timestamps directly from the annotation.
            2. Use explicit start and end timestamps to assign video frames to mini episodes.
            3. Store all data per mini episode in `self.mini_episode_data`.

        Args:
            annotation_file_path (str): Path to the JSON annotation file containing
                                         exercise names, start/end timestamps, and video
                                         frame timestamp references for all competition videos.

        Returns:
            None: Populates `self.mini_episode_data` and `self.video_frame_timestamps_data`
                  as side effects.
        """
        with open(annotation_file_path, 'r') as file:
            annotations = json.load(file)

        self.mini_episode_data = []
        # save the video frame timestamps data for loading video frames
        self.video_frame_timestamps_data = {}
        for annotation in tqdm(annotations, desc="Loading competition annotations ..."):
            # load data from json
            long_range_video_file = annotation['long_range_video_file']
            mini_episode_exercise_names = annotation['exercises']

            # get video id
            video_id = self.extract_video_id(long_range_video_file)

            # load video frame timestamps, the json stores only the file name
            video_frame_timestamps_path = os.path.join(
                self.data_root,
                annotation['video_timestamps']
            )

            # the timestamps in the competition split are already normalized
            video_frame_timestamps = load_video_timestamps(
                video_frame_timestamps_path,
                is_competition_split = True
            )

            # save for future use while loading video frame paths
            self.video_frame_timestamps_data[video_id] = video_frame_timestamps

            # get timestamps of video frames corresponding to each mini episode
            mini_episode_start_timestamps = annotation['exercise_start_timestamps']
            mini_episode_end_timestamps = annotation['exercise_end_timestamps']
            
            mini_episode_video_frame_timestamps = self.get_video_frame_timestamps_per_mini_episodes_competition(
                video_frame_timestamps,
                mini_episode_start_timestamps,
                mini_episode_end_timestamps,
                video_end_timestamp = float(video_frame_timestamps[-1])
            )

            # store data per mini episode
            num_mini_episodes = len(mini_episode_exercise_names)
            for idx in range(num_mini_episodes):
                _mini_episode_id = f"{video_id}_{idx}"
                _mini_episode_video_frame_timestamps = mini_episode_video_frame_timestamps[idx]
                _mini_episode_exercise_name = mini_episode_exercise_names[idx]
                self.mini_episode_data.append({
                    "video_id": video_id,
                    "mini_episode_id": _mini_episode_id,
                    "mini_episode_exercise_name": _mini_episode_exercise_name,
                    "mini_episode_video_frame_timestamps": _mini_episode_video_frame_timestamps,
                })

        print("Done!")

    def load_video_frame_paths(self, video_frames_root: str) -> None:
        """
        Load and cache video frame file paths for each mini episode.

        This method iterates through all mini episodes and determines which extracted
        video frame files correspond to each mini episode's time range. It accounts for
        the model's FPS by incrementing through frames at the model's frame rate rather
        than the video's native frame rate.

        The frame paths are stored directly in each mini episode's data dictionary
        under the key "mini_episode_video_frame_paths".

        Frame files are expected to follow the naming convention:
            `{video_frames_root}/{video_id}/frame_{index:06d}.jpg`

        Args:
            video_frames_root (str): Root directory containing subdirectories for each
                                      video, which in turn contain the extracted frame
                                      image files.

        Returns:
            None: Updates each entry in `self.mini_episode_data` with a
                  "mini_episode_video_frame_paths" key as a side effect.
        """
        self.video_frames_cache = {}
        for _mini_episode_data in tqdm(self.mini_episode_data, desc="Loading video frame paths..."):
            video_id = _mini_episode_data["video_id"]
            video_frame_timestamps = self.video_frame_timestamps_data[video_id]
            mini_episode_video_frame_timestamps = \
                _mini_episode_data["mini_episode_video_frame_timestamps"]
            
            mini_episode_start_timestamp = mini_episode_video_frame_timestamps[0]
            mini_episode_end_timestamp = mini_episode_video_frame_timestamps[-1]

            # loop through video timestamps and increment index at self.model_fps ...
            # to align with timestamps of extracted frames
            # update mini_episode_video_frame_timestamps to align with frames
            mini_episode_video_frame_paths, mini_episode_video_frame_timestamps = [], []
            extracted_frame_idx, next_frame_timestamp = 0, video_frame_timestamps[0]
            for video_frame_timestamp in video_frame_timestamps:
                if video_frame_timestamp >= next_frame_timestamp:
                    # filter mini episode timestamps
                    if mini_episode_start_timestamp <= video_frame_timestamp <= mini_episode_end_timestamp:
                        frame_path = f"frame_{extracted_frame_idx:06d}.jpg"
                        frame_path = os.path.join(video_frames_root, video_id, frame_path)
                        mini_episode_video_frame_paths.append(frame_path)
                        mini_episode_video_frame_timestamps.append(video_frame_timestamp)
                    extracted_frame_idx += 1
                    next_frame_timestamp += 1.0 / self.model_fps

            _mini_episode_data["mini_episode_video_frame_paths"] = mini_episode_video_frame_paths
            _mini_episode_data["mini_episode_video_frame_timestamps"] = mini_episode_video_frame_timestamps

        print("Done!")

    def load_video_frame_paths_comnpetition(self, video_frames_root: str) -> None:
        """
        Load and cache video frame file paths for each mini episode in the competition split.

        This method is intended to mirror the functionality of `load_video_frame_paths` but
        adapted for the competition split format. It is currently a placeholder and will
        trigger a breakpoint when called, indicating it has not yet been fully implemented.

        Args:
            video_frames_root (str): Root directory containing subdirectories for each
                                      video, which in turn contain the extracted frame
                                      image files.

        Returns:
            None
        """
        breakpoint()

    def __len__(self,) -> int:
        """
        Return the total number of mini episodes in the dataset.

        Returns:
            int: The number of mini episodes across all videos in the dataset.
        """
        return len(self.mini_episode_data)

    def get_responses_and_timestamps_of_miniepisode(
        self, 
        mini_episode_id: str
    ) -> Tuple[List[str],List[float]]:
        """
        Retrieve the data for a specific mini episode by its ID.

        Args:
            mini_episode_id (str): The unique identifier of the mini episode
                                    in the format "{video_id}_{index}".

        Returns:
            Tuple[List[str], List[float]]: A tuple containing:
                - feedbacks (List[str]): List of feedback strings for the mini episode.
                - feedback_timestamps (List[float]): Timestamps corresponding to each feedback.

        Raises:
            ValueError: If the provided mini_episode_id does not exist in the dataset.
        """
        for _data in self.mini_episode_data:
            if _data["mini_episode_id"] == mini_episode_id:
                return _data["mini_episode_feedbacks"], _data["mini_episode_feedback_timestamps"]
        raise ValueError(f"mini_episode_id '{mini_episode_id}' does not exist in self.mini_episode_data")

    def __getitem__(self, video_idx: int) -> Dict[str, str | List[str] | List[float]]:
        """
        Retrieve a mini episode's data by index.

        Args:
            video_idx (int): Index of the mini episode to retrieve.

        Returns:
            Dict[str, str | List[str] | List[float]]: A dictionary containing the
                mini episode's data with the following keys:
                - "video_id" (str): The ID of the source video.
                - "mini_episode_id" (str): Unique identifier for the mini episode
                                            in the format "{video_id}_{index}".
                - "mini_episode_exercise_name" (str): Name of the exercise in this
                                                       mini episode.
                - "mini_episode_feedbacks" (List[str]): List of feedback strings
                                                         for this mini episode.
                - "mini_episode_feedback_timestamps" (List[float]): Timestamps
                                                                      corresponding to each feedback.
                - "mini_episode_video_frame_timestamps" (List[float]): Timestamps
                                                                         of video frames in this episode.
                - "mini_episode_video_frame_paths" (List[str], optional): File paths
                                                                            to video frames, present only
                                                                            if load_video_frame_paths=True
                                                                            was set during initialization.
        """
        return self.mini_episode_data[video_idx]
