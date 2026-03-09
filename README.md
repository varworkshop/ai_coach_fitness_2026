# AI Coach Challenge: Fitness -- VAR Workshop @ CVPR 2026

Here we provide the code to load and preprocess the QEVD-FIT-COACH(-Benchmark) benchmark and dataset for the AI Coach Challenge: Fitness at the VAR Workshop @ CVPR 2026. We also provide code to implement a simple Qwen3-VL-2B-Instruct baseline and evaluate it's predictions.

<video src="./assets/fitness_competition.mp4" controls width="100%"></video>

[**What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction (NeurIPS 2024 D&B Track)
**](https://arxiv.org/pdf/2407.08101)

## Running the Code

First download and extract the QEVD-FIT-COACH(-Benchmark/Competition) [here](https://www.qualcomm.com/developer/software/qevd-dataset/downloads) to `DATA_ROOT`. Links to splits:
- [Train](https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-COACH/QEVD-FIT-COACH.zip).
- [Benchmark](https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-COACH-Benchmark/QEVD-FIT-COACH-Benchmark.zip).
- [VAR Competition](https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/QEVD-FIT-COACH-Competition-CVPR2025/QEVD-FIT-COACH-Competition-CVPR2025.zip)


The `DATA_ROOT` folder should contain the following:
- *feedbacks_long_range.json*: The json file containing the annotations. Details [here](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/QSC-QEVD-FIT-COACH-Dataset-Download-Instructions.pdf).
- *long_range_videos*: Folder containing the videos and the associated timestamps when each frame was recorded.

### Extract Video Frames
You can specifiy the fps, here we choose 2 for the Qwen3-VL-Instruct baseline.
```
python extract_frames.py \
	--dataset_root <DATA_ROOT> \
	--fps 2
```
This extracts the video frames at 2 fps to the folder `long_range_video_frames_2fps` in the `DATA_ROOT` folder.

### Dataloader
We provide an simplified dataloader in `data.py` that is built upon the official dataloader [here](https://github.com/Qualcomm-AI-research/FitCoach/blob/main/src/fitness_datasets/fitcoach.py). You can switch between the train/benchmark/competition splits by specifying the appropriate data folder (`DATA_ROOT`).

### Run Qwen3-VL Baseline
You can run the simple Qwen3-VL-2B-Instruct baseline that provides feedbacks every 5 seconds using,
```
export HF_HOME=<path to your hf cache>
python qwen3_vl_baseline.py \
    --data_root <DATA_ROOT>
```
The `qwen3_vl_baseline.py` script will save a file `predictions.json` to the folder `./predictions` by default. You can control where to save this file using the flags  `--predictions_save_root` and `--predictions_file_name`.

### Run Evaluation
You can evaluate the baseline predictions at `./predictions/predictions.json` using,
```
export HF_HOME=<path to your hf cache>
python eval.py \
    --data_root <DATA_ROOT> \
    --results_file ./predictions/predictions.json
```
The evaluation is based on the official code [here](https://github.com/Qualcomm-AI-research/FitCoach/blob/main/scripts/evaluate_baseline.py).

## Citation

```text
@inproceedings{livefit,
   title = {What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction},
   author = {Sunny Panchal and Apratim Bhattacharyya and Guillaume Berger and Antoine Mercier and Cornelius B{\"{o}}hm and Florian Dietrichkeit and Reza Pourreza and Xuanlin Li and Pulkit Madan and Mingu Lee and Mark Todorovich and Ingo Bax and Roland Memisevic},
   booktitle = {NeurIPS (D&B Track)},
   year = {2024},
}
```
