from typing import Tuple
import argparse
import ast
import json
import random
import re
from typing import Any, List, Union, Optional

import evaluate
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import FitCoachDatset

PROMPT = (
    "You are an intelligent chatbot designed for evaluating feedbacks provided by a virtual "
    "fitness coach to a person. You always provide your responses as a python dictionary string.\n"
    "Your task is to compare the accuracy of the the predicted feedback with the ground truth "
    "feedback. Here is how you can accomplish this:\n"
    "-The predicted feedback must be factually accurate, relevant and align with the ground truth "
    "feedback.\n"
    "-Consider synonyms or paraphrases as valid matches.\n"
    "-Take into account repetition counts that can expressed both in numeric form or in words."
)

USER_CONTENT = (
    "Please evaluate the following predicted feedback:\n"
    "-Ground truth feedback: <1>\n"
    "-Predicted feedback: <2>\n\n"
    "Provide your evaluation as a python dictionary string with the accuracy score where the "
    "score is an integer value between 1 and 5, with 5 indicating the highest level of accuracy."
    "Generate the response only in the form of a Python dictionary string with keys 'score', "
    "where its value is the accuracy score in INTEGER, not STRING."
    "For example, your response should look like this: {'score': int(score)}."
    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION."
)

def parse_args() -> argparse.ArgumentParser:
    """Parse command line arguments."""
    _parser = argparse.ArgumentParser(
        description="Evaluator"
    )
    _parser.add_argument(
        "--results_file",
        default=None,
        required=True,
        help="Path to results json file. First run scripts/evaluate_baseline.py .",
    )

    _parser.add_argument(
        "--data_root",
        required=True,
        help="Path to results json file. First run scripts/evaluate_baseline.py .",
    )
    
    return _parser.parse_args()

def load_llm_judge(
    model_id: str = "meta-llama/Meta-Llama-3-70B-Instruct"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the LLM judge model and tokenizer.

    :param model_id:
        The model identifier to load from HuggingFace Hub.
    :return:
        A tuple containing the loaded causal language model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True
    )
    return model, tokenizer

def _get_alignment_matrix(
    gt_feedback_timestamps: np.array,
    pred_feedback_timestamps: np.array,
    pred_feedbacks: list[str],
    tolerance: float = 3.0,
) -> tuple[list[int], list[int]]:
    """Perform temporal matching
    :param gt_feedback_timestamps:
        List of timestamps for ground truth feedback.
    :param pred_feedback_timestamps:
        List of timestamps for predicted feedback.
    :param pred_feedbacks:
        List of predicted feedbacks.
    :param tolerance:
        Temporal matching tolerance.

    :return:
        Indices of matching gt and pred feedbacks
    """
    matching_row_idxs, matching_col_idxs = [], []
    last_match_idx = -1
    for idx_x, x in enumerate(gt_feedback_timestamps):
        min_idx = np.argmin((pred_feedback_timestamps - x) ** 2)
        if (
            np.abs(x - pred_feedback_timestamps[min_idx]) < (tolerance / 2.0)
            and min_idx > last_match_idx
            and (min_idx not in matching_col_idxs)
            and pred_feedbacks[min_idx] != ""
        ):
            matching_row_idxs.append(idx_x)
            matching_col_idxs.append(min_idx)
            last_match_idx = min_idx
    return matching_row_idxs, matching_col_idxs

def _get_temporally_aligned_feedbacks(
    gt_feedback_timestamps: list[float],
    pred_feedback_timestamps: list[float],
    gt_feedbacks: list[str],
    pred_feedbacks: list[str],
    tolerance: float = 3.0,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    Returns a list of temporally aligned feedbacks between the ground truth and predictions.

    :param gt_feedback_timestamps:
        List of ground truth feedback timestamps.
    :param pred_feedback_timestamps:
        List of predicted feedback timestamps.
    :param gt_feedbacks:
        List of ground truth feedbacks
    :param pred_feedbacks:
        List of predicted feedbacks
    :param tolerance:
        Temporal window for matching (in seconds).

    :return:
        Number of temporal matches (int) for ground truth and predicted feedbacks,
        along with the matches.
    """
    gt_feedback_timestamps = np.array(gt_feedback_timestamps)
    pred_feedback_timestamps = np.array(pred_feedback_timestamps)

    matched_feedbacks = []
    matched_idxs_gt = []
    matched_idxs_pred = []
    matching_row_idxs, matching_col_idxs = [], []
    if len(pred_feedback_timestamps) > 0:
        matching_row_idxs, matching_col_idxs = _get_alignment_matrix(
            gt_feedback_timestamps,
            pred_feedback_timestamps,
            pred_feedbacks,
            tolerance,
        )

    for match_idx, match_jdx in zip(matching_row_idxs, matching_col_idxs):
        matched_feedbacks.append((gt_feedbacks[match_idx], pred_feedbacks[match_jdx]))
        matched_idxs_gt.append(match_idx)
        matched_idxs_pred.append(match_jdx)

    return len(matched_idxs_gt), len(matched_idxs_pred), matched_feedbacks

def _compute_temporal_fscore(
    gt_feedbacks: list[str],
    pred_feedbacks: list[str],
    gt_feedback_timestamps: list[float],
    pred_feedback_timestamps: list[float],
    t_f_score_running_stats: dict[str, Union[float, int]],
    eps: Optional[float] = 1e-12,
) -> tuple[float, list[tuple[str, str]], dict[str, Union[float, int]]]:
    """
    :param gt_feedback_timestamps:
        List of ground truth feedback timestamps.
    :param pred_feedback_timestamps:
        List of predicted feedback timestamps.
    :param gt_feedbacks:
        List of ground truth feedbacks
    :param pred_feedbacks:
        List of predicted feedbacks
    :param t_f_score_running_stats:
        Stats for computing the temporal F-score over the entire dataset.

    :return:
        Temporal F-score (float).
    """
    # Match ground truth feedbacks to predicted feedbacks (recall)
    num_matched_gt, _, matched_feedbacks = _get_temporally_aligned_feedbacks(
        gt_feedback_timestamps, pred_feedback_timestamps, gt_feedbacks, pred_feedbacks
    )

    # Match predicted feedbacks to ground truth feedbacks (precision)
    _, num_matched_preds, _ = _get_temporally_aligned_feedbacks(
        pred_feedback_timestamps, gt_feedback_timestamps, pred_feedbacks, gt_feedbacks
    )

    # Accumulate running stats
    t_f_score_running_stats["total_matched_gt_feedbacks"] += num_matched_gt
    t_f_score_running_stats["total_matched_pred_feedbacks"] += num_matched_preds
    t_f_score_running_stats["total_num_gt_feedbacks"] += len(gt_feedbacks)
    t_f_score_running_stats["total_num_pred_feedbacks"] += len(pred_feedbacks)

    # Compute temporal precision and recall
    precision = t_f_score_running_stats["total_matched_pred_feedbacks"] / (
        t_f_score_running_stats["total_num_pred_feedbacks"] + eps
    )
    recall = t_f_score_running_stats["total_matched_gt_feedbacks"] / (
        t_f_score_running_stats["total_num_gt_feedbacks"] + eps
    )
    f_score = 2 * ((precision * recall) / (precision + recall + eps))
    return f_score, matched_feedbacks, t_f_score_running_stats

def _compute_bert_scores(bert_score, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
    """
    :param matched_feedbacks:
        List of matched ground truth and predicted feedbacks.

    :return:
        List of METEOR scores.
    """
    _bert_scores = []
    if len(matched_feedbacks) > 0:
        _bert_scores = bert_score.compute(
            references=[x[0] for x in matched_feedbacks],
            predictions=[x[1] for x in matched_feedbacks],
            lang="en",
        )["f1"]
    return _bert_scores

def _compute_meteor_scores(meteor_score, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
    """
    :param matched_feedbacks:
        List of matched ground truth and predicted feedbacks.

    :return:
        List of METEOR scores.
    """
    _meteor_scores = []
    for matched_feedback in matched_feedbacks:
        _meteor_scores.append(
            meteor_score.compute(
                references=[matched_feedback[0]], predictions=[matched_feedback[1]]
            )["meteor"]
        )
    return _meteor_scores

def _compute_rouge_scores(rouge_score, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
    """
    :param matched_feedbacks:
        List of matched ground truth and predicted feedbacks.

    :return:
        List of ROUGE scores.
    """
    _rouge_scores = []
    for matched_feedback in matched_feedbacks:
        _rouge_scores.append(
            rouge_score.compute(
                references=[matched_feedback[0]], predictions=[matched_feedback[1]]
            )["rougeL"]
        )
    return _rouge_scores

def _compute_llm_scores(model, tokenizer, matched_feedbacks: list[tuple[str, str]]) -> list[float]:
    """
    Compute LLM-based accuracy scores for matched feedback pairs.

    Uses a language model judge to evaluate the accuracy of predicted feedbacks
    against ground truth feedbacks by generating a score between 1 and 5.

    :param model:
        The causal language model used as a judge.
    :param tokenizer:
        The tokenizer associated with the judge model.
    :param matched_feedbacks:
        List of tuples containing matched ground truth and predicted feedbacks.

    :return:
        List of LLM accuracy scores as floats.
    """
    _llm_scores = []
    for matched_feedback in tqdm(matched_feedbacks, desc="Computing LLM Acc"):
        messages = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": fill_template(USER_CONTENT, [matched_feedback[0], matched_feedback[1]]),
            },
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            outputs = model.generate(
                input_ids,
                max_new_tokens=64,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )

        response = outputs[0][input_ids.shape[-1]:]
        response = tokenizer.decode(response, skip_special_tokens=True)

        score_dicts = extract_substrings_in_curly_braces(response)
        if len(score_dicts) == 1:
            try:
                score_dict = ast.literal_eval("{" + score_dicts[0] + "}")
                _llm_scores.append(float(score_dict["score"]))
            except (SyntaxError, TypeError, ValueError):
                pass

    return _llm_scores


def _print_eval_summary(
    gt_feedbacks: list[str],
    gt_feedback_timestamps: list[float],
    pred_feedbacks: list[str],
    pred_feedback_timestamps: list[float],
    t_f_score: float,
    meteor_scores: list[float],
    rouge_scores: list[float],
    bert_scores: list[float],
    llm_acc_scores: list[float],
) -> None:
    """
    :param gt_feedbacks:
        List of ground truth feedbacks.
    :param gt_feedback_timestamps:
        List of ground truth feedback timestamps.
    :param pred_feedbacks:
        List of predicted feedbacks.
    :param pred_feedback_timestamps:
        List of predicted feedback timestamps.
    :param t_f_score:
        Temporal F-score.
    :param meteor_scores:
        List of METEOR scores.
    :param rouge_scores:
        List of ROUGE scores.
    :param bert_scores:
        List of BERT scores.
    :param llm_acc_scores:
        List of LLM accuracy scores.
    """
    mean = lambda x: sum(x) / (len(x) + 1e-12)
    tqdm.write("=" * 40)
    tqdm.write("-" * 40)
    tqdm.write("GT Timestamp => GT Feedback")
    for gt_feedback, gt_feedback_timestep in zip(gt_feedbacks, gt_feedback_timestamps):
        tqdm.write(f"{gt_feedback_timestep:.2f} => {gt_feedback}")
    tqdm.write("-" * 40)
    tqdm.write("Pred Timestamp => Pred Feedback")
    for pred_feedback, pred_feedback_timestep in zip(pred_feedbacks, pred_feedback_timestamps):
        tqdm.write(f"{pred_feedback_timestep:.2f} => {pred_feedback}")
    tqdm.write("-" * 40)
    tqdm.write("Running Means ==>")
    tqdm.write(f"METEOR Score: {mean(meteor_scores):.3f}")
    tqdm.write(f"Rouge-L Score: {mean(rouge_scores):.3f}")
    tqdm.write(f"BERT Score: {mean(bert_scores):.3f}")
    tqdm.write(f"LLM Acc Score: {mean(llm_acc_scores):.3f}")
    tqdm.write(f"Temporal F-Score: {t_f_score:.3f}")


def extract_substrings_in_curly_braces(text: str) -> List[str]:
    """
    Extracts substrings enclosed in curly braces using regular expressions.

    :param text:
        The input string to search for substrings enclosed in curly braces.

    :return:
        A list of strings found between curly braces, with the braces themselves excluded.
    """
    pattern = r"\{(.*?)\}"  # Matches content between curly braces (non-greedy)
    matches = re.findall(pattern, text)
    return matches


def fill_template(template: str, fillers: List[str]) -> str:
    """
    Fills a string template (here the LLM prompts) with content.

    :param template:
        The template string containing placeholders in the format <1>, <2>, etc.
    :param fillers:
        A list of strings to replace the placeholders in the template.

    :return:
        The template string with all placeholders replaced by the corresponding fillers.
    """
    for idx, filler in enumerate(fillers):
        placeholder = f"<{idx + 1}>"
        template = template.replace(placeholder, filler)
    return template


def load_json_from_file(file_path: str) -> Any:
    """
    Loads JSON data from a file and returns it as a Python object.

    :param file_path:
        The path to the JSON file to be loaded.

    :return:
        The parsed JSON data as a Python object, or None if the file is not found.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

def run_eval(args) -> None:
    """
    Run the full evaluation pipeline on predicted feedbacks against ground truth.

    Loads the dataset and predictions, computes temporal F-score, METEOR, ROUGE-L,
    BERTScore, and LLM accuracy scores for each evaluation item, and prints a running
    summary after processing each item.

    :param args:
        Parsed command line arguments containing:
        - data_root: Path to the dataset root directory.
        - results_file: Path to the JSON file containing predictions.
    """
    dataset = FitCoachDatset(
        args.data_root,
        load_video_frame_paths=False
    )

    llm_judge_model, llm_judge_tokenizer = load_llm_judge()

    preds = load_json_from_file(args.results_file)

    bert_score = evaluate.load("bertscore")
    rouge_score = evaluate.load("rouge")
    meteor_score = evaluate.load("meteor")
    llm_acc_scores, meteor_scores, rouge_scores, bert_scores = [], [], [], []
    t_f_score_running_stats = {
        "total_num_gt_feedbacks": 0,
        "total_num_pred_feedbacks": 0,
        "total_matched_gt_feedbacks": 0,
        "total_matched_pred_feedbacks": 0,
    }

    for eval_item in tqdm(preds):
        assert 'mini_episode_id' in eval_item and \
            "pred_feedbacks" in eval_item and \
            "pred_feedback_timestamps" in eval_item
        
        pred_feedbacks, pred_feedback_timestamps = eval_item["pred_feedbacks"], eval_item["pred_feedback_timestamps"]
        gt_feedbacks, gt_feedback_timestamps = dataset.get_responses_and_timestamps_of_miniepisode(
            eval_item['mini_episode_id']
        )

        t_f_score, matched_feedbacks, t_f_score_running_stats = _compute_temporal_fscore(
            gt_feedbacks,
            pred_feedbacks,
            gt_feedback_timestamps,
            pred_feedback_timestamps,
            t_f_score_running_stats,
        )

        rouge_scores += _compute_rouge_scores(rouge_score, matched_feedbacks)
        meteor_scores += _compute_meteor_scores(meteor_score, matched_feedbacks)
        bert_scores += _compute_bert_scores(bert_score, matched_feedbacks)
        with torch.no_grad():
            llm_acc_scores += _compute_llm_scores(
                llm_judge_model,
                llm_judge_tokenizer,
                matched_feedbacks
            )

        _print_eval_summary(
            gt_feedbacks,
            gt_feedback_timestamps,
            pred_feedbacks,
            pred_feedback_timestamps,
            t_f_score,
            meteor_scores,
            rouge_scores,
            bert_scores,
            llm_acc_scores,
        )



if __name__ == "__main__":
    # Parse CLI arguments
    args = parse_args()
    run_eval(args)
