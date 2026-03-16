from typing import Tuple, List, Dict
import torch
import json
import os
import ast
import re
import numpy as np
import argparse

from tqdm import tqdm
from json_repair import repair_json
from torch.utils.data import Dataset

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from data import FitCoachDatset
from utils import load_frames_into_np_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FitCoach Video LLM Inference")
    parser.add_argument(
        "--data_root",
        required=True,
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="Model ID to load from HuggingFace"
    )
    parser.add_argument(
        "--model_fps",
        type=int,
        default=2,
        help="Frames per second for the model"
    )
    parser.add_argument(
        "--prediction_interval",
        type=float,
        default=5.0,
        help="Prompting interval in seconds"
    )
    parser.add_argument(
        "--llm_max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of new tokens for the LLM"
    )
    parser.add_argument(
        "--predictions_save_root",
        type=str,
        default="./predictions",
        help="Directory to save predictions"
    )
    parser.add_argument(
        "--predictions_file_name",
        type=str,
        default="predictions.json",
        help="File name for saving predictions"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=384,
        help="Resolution (used as both width and height) for loading video frames. Ignored if --width and --height are set."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width for loading video frames. Overrides --resolution if set together with --height."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height for loading video frames. Overrides --resolution if set together with --width."
    )
    return parser.parse_args()


def load_modal_and_processor(
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
) -> Tuple[Qwen3VLForConditionalGeneration, AutoProcessor]:
    #default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    print("Model device map:", model.hf_device_map)
    processor = AutoProcessor.from_pretrained(
        model_id
    )
    return model, processor

def load_dataset(data_root: str, model_fps: int = 2) -> Dataset:
    ds = FitCoachDatset(
        data_root,
        model_fps,
        load_video_frame_paths = True
    )
    return ds

def save_predictions_file(
    predictions_dict: dict,
    predictions_save_root: str,
    predictions_file_name: str
) -> None:
    os.makedirs(predictions_save_root, exist_ok=True)
    if predictions_file_name.endswith(".json"):
        out_file_path = os.path.join(predictions_save_root, predictions_file_name)
    else:
        out_file_path = os.path.join(
            predictions_save_root,
            predictions_file_name + ".json"
        )
    with open(out_file_path, "w") as f:
        json.dump(predictions_dict, f)


@torch.no_grad()
def get_feedback_from_qwen(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    video: np.ndarray,
    exercise_name: str,
    max_new_tokens: int = 1024
) -> str:
    messages: list[dict] = [
        {
            "role": "system", 
            "content": (
                "You are an expert fitness coach that provides feedback. "
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                },
                {
                    "type": "text", 
                    "text": (
                        f"You are observing a person doing the following exercise {exercise_name}."
                        "First check if they are doing the exercise correctly or if they have made a mistake. "
                        "If they are doing the exercise correctly, provide them with positive encouragement. "
                        "If they have made a mistake, provide feedback to help them correct the mistake. "
                        "Return a python dict with two fields: "
                        "{'has_mistake': <bool>, 'feedback': <str>}. "
                        "RETURN JUST THE PYTHON DICT DO NOT RETURN ANYTHING ELSE. "
                    )
                },
            ],
        }
    ]

    # Preparation for inference
    text: str = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    video_metadata = {"fps": 2, "total_num_frames": len(video)}
    inputs = processor(
        text=[text],
        images=None,
        videos=video,
        padding=True,
        video_metadata=[video_metadata],
        return_tensors="pt",
    )
    inputs = inputs.to("cuda").to(torch.bfloat16)

    # Inference: Generation of the output
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated_ids: torch.Tensor = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed: list[torch.Tensor] = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text: list[str] = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return_dict = output_text[0]
    
    # Make sure json is formed correctly
    return_dict = re.sub(r"^```(?:json)?\s*", "", return_dict.strip(), flags=re.IGNORECASE)
    return_dict = re.sub(r"\s*```$", "", return_dict.strip())
    
    # Attempt repair for truncated/malformed output
    repaired = repair_json(return_dict, return_objects=True)
    if isinstance(repaired, dict):
        try:
            return repaired['feedback']
        except ValueError:
            pass
    return ""

def get_predictions(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    dataset: Dataset,
    prediction_interval: float = 5., # prompting interval in sec
    llm_max_new_tokens: int = 128,
    resolution: int = 384, # input video resolution
    width: int = None,
    height: int = None
) -> List[Dict[str, str | List[str] | List[float]]]:
    # init ds
    predictions = []
    
    print("Inference loop ... ")
    for data in tqdm(dataset):
        # the dataset should load the video frame paths 
        # (see flag load_video_frame_paths in data.py)
        assert "mini_episode_video_frame_paths" in data

        video_id = data["video_id"]
        mini_episode_id = data["mini_episode_id"]
        mini_episode_exercise_name = data["mini_episode_exercise_name"]
        mini_episode_video_frame_paths = data["mini_episode_video_frame_paths"]
        mini_episode_video_frame_timestamps = data["mini_episode_video_frame_timestamps"]
        
        # will be available for the train/benchmark splits, not for the competition split
        gt_mini_episode_feedbacks = data.get("mini_episode_feedbacks", None)
        gt_mini_episode_feedback_timestamps = data.get("mini_episode_feedback_timestamps", None)
        
        print("="*40)

        last_pred_timestamp = mini_episode_video_frame_timestamps[0]
        pred_mini_episode_feedbacks, pred_mini_episode_feedback_timestamps = [], []
        for video_frame_idx in range(len(mini_episode_video_frame_paths)):
            curr_timestamp = mini_episode_video_frame_timestamps[video_frame_idx]
            # wait prediction_interval before next utterance
            if (curr_timestamp - last_pred_timestamp) >= prediction_interval:
                # predict feedback
                video = load_frames_into_np_array(
                    mini_episode_video_frame_paths[:video_frame_idx + 1],
                    resolution=resolution,
                    width=width,
                    height=height
                )
                
                pred_feedback = get_feedback_from_qwen(
                    model, processor, video, mini_episode_exercise_name,
                    max_new_tokens=llm_max_new_tokens
                )
                
                if pred_feedback is not None:
                    pred_mini_episode_feedbacks.append(pred_feedback)
                    pred_mini_episode_feedback_timestamps.append(curr_timestamp)

                # update next prediction timestamp
                last_pred_timestamp += prediction_interval

        
        if gt_mini_episode_feedbacks is not None and gt_mini_episode_feedback_timestamps is not None:
            for _gt_feedback, _gt_feedback_timestamp in \
                zip(gt_mini_episode_feedbacks, gt_mini_episode_feedback_timestamps):
                print(f"GT at {_gt_feedback_timestamp}: ", _gt_feedback)
            print("-"*40)
            
        for _pred_feedback, _pred_feedback_timestamp in \
            zip(pred_mini_episode_feedbacks, pred_mini_episode_feedback_timestamps):
            print(f"Pred at {_pred_feedback_timestamp}: ", _pred_feedback)

        predictions.append(
            {
                "mini_episode_id": mini_episode_id,
                "pred_feedbacks": pred_mini_episode_feedbacks,
                "pred_feedback_timestamps": pred_mini_episode_feedback_timestamps
            }
        )

    return predictions

def run_pred(args: argparse.Namespace):
    model, processor = load_modal_and_processor(model_id=args.model_id)
    dataset = load_dataset(args.data_root, args.model_fps)
    predictions = get_predictions(
        model,
        processor,
        dataset,
        prediction_interval=args.prediction_interval,
        llm_max_new_tokens=args.llm_max_new_tokens,
        resolution=args.resolution,
        width=args.width,
        height=args.height
    )
    save_predictions_file(
        predictions_dict=predictions,
        predictions_save_root=args.predictions_save_root,
        predictions_file_name=args.predictions_file_name
    )


if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    run_pred(args)

