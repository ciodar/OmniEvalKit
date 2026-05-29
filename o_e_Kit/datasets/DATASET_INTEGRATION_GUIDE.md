# Dataset Integration Full Guide

This document explains how to integrate a new multimodal evaluation dataset into the o_e_Kit framework.

## ⚠️ Files That Must Be Modified (Copy This Checklist)

When integrating a new dataset `xxx_dataset`, **all** of the following files **must** be modified. Missing any will cause errors:

```
□ 1. playground/convert_xxx.py              # Conversion script (generates JSONL)
□ 2. data/omni/raw_hf/xxx/                  # Data directory (JSONL + media files)
□ 3. o_e_Kit/utils/args/dataset_args.py     # DatasetConfig registration
□ 4. o_e_Kit/utils/dataset_loader.py        # Add elif branch in load_dataset()
□ 5. o_e_Kit/utils/evaluation_runner.py     # Add evaluation logic in evaluate_omni_datasets()
□ 6. o_e_Kit/utils/eval.py                  # ⚠️ Two places:
     □ 6a. Add "xxx_dataset" to the dataset_name in [...] list
     □ 6b. Add group field configuration to the group_fields dict
□ 7. All generation config files under o_e_Kit/configs/ (6 files):
     □ omni_generation_configs.json
     □ omni_generation_configs_nosys.json
     □ omni_generation_configs_nosys_interleave.json      # ⚠️ Default for test scripts
     □ omni_generation_configs_nosys_interleave_sys.json
     □ omni_generation_configs_nosys_interleave_96.json
     □ omni_generation_configs_fullprompt.json
□ 8. Run validation tests (see "Verifying Integration" section below)
```

### Commonly Missed Items 🔥

| Missed Location | Error Message |
|---------|---------|
| `eval.py` dataset list | `Unsupported dataset 'xxx'. Please add an evaluation flow...` |
| `configs/*.json` | `No config found for dataset 'xxx'` |
| `dataset_loader.py` | `Unsupported dataset: xxx` |
| `evaluation_runner.py` | Dataset won't be evaluated (silently skipped) |

## Overview

Integrating a new dataset requires the following steps:

1. **Analyze Data Format** - Understand the raw data structure
2. **Conversion Script** - `playground/convert_xxx.py`
3. **Extract Audio** - Extract `.wav` files from video (⚠️ required for `load_av`)
4. **Place Data Files** - `data/omni/raw_hf/`
5. **Dataset Configuration** - `o_e_Kit/utils/args/dataset_args.py`
6. **Dataset Loading** - `o_e_Kit/utils/dataset_loader.py`
7. **Evaluation Execution** - `o_e_Kit/utils/evaluation_runner.py`
8. **Generation Configuration** - All JSON config files under `o_e_Kit/configs/` (6 files)
9. **Evaluation Functions** - `o_e_Kit/utils/eval.py` (dataset list + group fields)
10. **Validation Tests** - Run eval_main.py to verify dataset loading and evaluation

## Quick Checklist (Review Before Integrating a New Benchmark)

Before writing any code, think through / check the following to significantly reduce rework:

1. **Task & Modality**
   - Is it **MCQ / Open QA / Caption / Multi-task mixed**?
   - Input modalities: **Audio / Video / Image / Multimodal combination**? How many media streams? (e.g., multiple audio, multiple images, multiple videos)
   - Are there **sequential/multi-turn QA** (e.g., SQA) or **streaming/proactive output** special interaction patterns?

2. **Time-Related Information**
   - Does the data have **timestamps / time ranges** (e.g., `time_stamp`, `time`, `realtime`)?
   - Does the official evaluation define a **fixed context window** (e.g., "60 seconds before the question")?
   - Do you plan to use **offline chunking** or online trimming? (This repo recommends offline chunking.)

3. **Paths & Naming**
   - Do the video/audio paths in the annotations **correspond one-to-one** with actual files on disk? Are there inconsistencies like `*_1-25.mp4` vs `*.mp4`?
   - Is a "**filename normalization**" mapping needed in the conversion script, or should naming be cleaned up during data preparation?

4. **Audio Source & Extraction Strategy**
   - Does this benchmark require audio?
   - Is audio in **separate files** (WAV/MP3) or **embedded in video**?
   - When to extract audio:
     - At conversion time: `convert_xxx.py --extract-audio` (recommended, extracts from chunked clips);
     - Or online during loading via `auto_extract_audio` (only suitable for small datasets or debugging).

5. **Unified JSONL Schema**
   - Have you designed which fields each sample needs?
   - Required fields: `VideoPath` / `WavPath` / `dataset_type` / `dataset_name` / `question` / `choices` / `gt_answer`.
   - What metadata needs to be included for **grouped statistics / analysis**? For example:
     - `task` / `task_group` (OVOBench)
     - `task_type` / `required_ability` / `video_categories` (StreamingBench)
     - `sqa_context` (sequential QA text context), etc.

6. **Generation Config & Prompt Design**
   - What **prompt template** should the model see? Does it vary by benchmark subtask?
   - Are additional placeholders needed: `{media}` / `{sqa_context}` / `<audio_1>` / `<image_1>`?
   - Estimate the **max_tokens / max_frames / max_fps** needed per sample. Is fps reduction necessary for long videos?

7. **Evaluation Statistics & Grouping**
   - Which fields do you want **grouped statistics** for in the final report? For example:
     - By task type, question type, ability label, video category, high-level category, etc.
   - Are these fields already written into the JSONL and configured in `eval.py`'s `group_fields`?

8. **Pre-Run Self-Check**
   - Did the conversion script output **missing_video / missing_clip** statistics? Are these numbers 0?
   - Is the directory structure correct: does `data_prefix_dir` + `VideoPath` form a valid file path?
   - Run with `python3 eval_main.py --eval_xxx_dataset --max_sample_num 5` and check:
     - Whether media can be loaded correctly;
     - Whether there are "audio not found" warnings in the logs;
     - Whether the prompt preview matches expectations.

With this checklist in hand, the "Detailed Steps" below will be clearer.

## Detailed Steps

### Step 1: Analyze Raw Data Format

First, analyze the raw format of the new dataset:

```bash
# View directory structure
ls -la /path/to/dataset/

# View JSON/JSONL file structure
head -n 50 /path/to/dataset/data.json
python3 -c "import json; print(json.dumps(json.load(open('data.json'))[0], indent=2))"
```

Understand:
- Video/audio/image path format
- Question and answer format
- Options format (MCQ)
- Metadata fields

### Step 2: Create Conversion Script (convert_xxx.py)

Create `convert_xxx.py` under the `playground/` directory:

```python
#!/usr/bin/env python3
"""
XXX Data Conversion Script
Converts raw data to the framework's unified standard JSONL format
"""

import os
import json
import argparse
from tqdm import tqdm


def convert_xxx_to_jsonl(input_path, output_path, data_root, verify_paths=True):
    """
    Unified output format:
    {
        "VideoPath": "path/to/video.mp4",    # Video path
        "WavPath": "",                        # Audio path (empty = auto-extract)
        "ImagePath": "path/to/image.jpg",    # Image path (if any)
        "dataset_type": "mcq",               # Task type: mcq, open_qa, caption
        "dataset_name": "xxx",               # Dataset name
        "question": "question text",         # Question
        "choices": ["Option A", "Option B", ...],  # MCQ options (without letter prefix)
        "gt_answer": "A",                    # Answer (letter for MCQ)
        # Other metadata...
    }
    """
    # 1. Load raw data
    # 2. Convert format (apply trimming, renaming normalization, etc. if needed)
    # 3. Verify paths (optional but strongly recommended)
    # 4. Write JSONL files (single-line + pretty versions)
    # 5. Output statistics (total samples / missing videos / per-task distribution, etc.)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert XXX to standard JSONL format')
    parser.add_argument('--input', type=str, default='/path/to/input')
    parser.add_argument('--output', type=str, default='/path/to/output.jsonl')
    parser.add_argument('--data-root', type=str, default='/path/to/data/')
    parser.add_argument('--no-verify', action='store_true')
    
    args = parser.parse_args()
    convert_xxx_to_jsonl(args.input, args.output, args.data_root, not args.no_verify)
```

Run the conversion:

```bash
cd playground
mkdir -p /path/to/dataset/annotation
python3 convert_xxx.py --no-verify
```

### Step 2.1: Trimming Strategy for Long Videos / Timestamped Tasks (OVOBench / StreamingBench, etc.)

For benchmarks with "long videos + explicit timestamps per question" (e.g., **OVOBench**, **StreamingBench**),
**do not feed the entire long video to the model**. Instead, explicitly handle the time window in the conversion script:

- **Typical Pattern A: Offline Chunking (Recommended)**
  - One sample per question, each corresponding to a pre-trimmed video clip:
    - OVOBench: `chunked_videos/{id}.mp4` / `{id}_{j}.mp4`, based on `realtime` or `test_info.realtime`;
    - StreamingBench-Real/Omni/SQA: `chunked_*_60/<original_video_name>_start_end.mp4`, based on `time_stamp` and `context_time` (e.g., 60 seconds).
  - In JSONL:
    - `VideoPath` directly references the **clip path** (relative to `data_root`), e.g.:
      - `chunked_videos/123.mp4`
      - `chunked_omni_60/sample_2_Misleading_Context_Understanding_240_300.mp4`
  - Advantages:
    - Strict alignment with the official implementation (consistent context window);
    - No online ffmpeg trimming during evaluation, yielding more stable reproduction.

- **Typical Pattern B: Online Trimming by Time Field (Advanced)**
  - No clip generation at conversion time; only record in JSONL:
    - `VideoPath`: path to the original long video;
    - `time_stamp` / `time_range` / `start_sec` / `end_sec` fields;
  - Use these time ranges in the video loading logic (e.g., `load_video` / `encode_video`) to sample only local frames.
  - This repo's main branch primarily uses **Pattern A (offline chunking)**; Pattern B is only an extension concept.

**Practical Experience (Pitfalls):**

- **Filename consistency is critical**:
  - Official annotations may list `sample_18_Scene_Understanding_1-25.mp4` while the actual disk only has `sample_18_Scene_Understanding.mp4`.
  - Add a "normalization" step in `convert_xxx.py` (e.g., `normalize_video_name`) to map to the real file:
    - First check if the original name exists;
    - If not, try stripping `_1-25` / `_26-50` suffixes and recheck;
    - Print the mapping process for easy debugging.

- **Long video time window selection**:
  - OVOBench: typically from 0 to `realtime`;
  - StreamingBench: strictly follows the official spec, using `[timestamp - context_time, timestamp]` (default 60 seconds).

### Step 3: Extract Audio from Video ⚠️ Important

**`load_av` requires audio files to exist!** The framework automatically looks for audio files with the same name in the video directory:
- `xxx.mp4` → looks for `xxx.wav` / `xxx.mp3` / `xxx.m4a` / `xxx.flac`

If no audio files exist in the video directory, extract them in advance:

```bash
# Method 1: Extract during conversion (recommended)
#   - If using "offline chunking", extract audio from clips, not the original long video
#   - i.e., trim first then extract: full.mp4 → chunk_x_y.mp4 → chunk_x_y.wav
python3 convert_xxx.py --extract-audio --workers 8

# Method 2: Batch extraction tool
python3 playground/extract_audio_batch.py \
    --data-dir /path/to/dataset \
    --dry-run  # Scan first for statistics

python3 playground/extract_audio_batch.py \
    --data-dir /path/to/dataset \
    --workers 8  # Execute extraction

# Method 3: Extract from JSONL video paths
python3 playground/extract_audio_batch.py \
    --jsonl /path/to/annotation.jsonl \
    --data-dir /path/to/dataset \
    --workers 8
```

Extraction parameters:
- `--workers`: Parallelism (default 4, recommend 8-16 for large datasets)
- `--sample-rate`: Sample rate (default 16000)
- `--dry-run`: Statistics only, no execution

### Step 4: Place Data Files

Place data in the corresponding subdirectory under `data/omni/raw_hf/` (data downloaded via `scripts/hf_download.py` is automatically placed correctly):

```bash
# If the dataset is already on HF, users can auto-download via the download script
python scripts/hf_download.py --datasets xxx_dataset --output_dir ./data

# During development, can also be placed manually
cp -r /path/to/your/DatasetName ./data/omni/raw_hf/dataset-name
```

### Step 5: Add Dataset Configuration (DatasetConfig)

Add a configuration in `o_e_Kit/utils/args/dataset_args.py`:

```python
# Add to the DATASET_REGISTRY list
DatasetConfig(
    name="xxx_dataset",                    # Internal name (used for --eval_xxx_dataset)
    display_name="XXX Dataset",            # Display name
    category="omni",                       # Category: audio, video, omni
    subcategory="qa",                      # Subcategory: qa, asr, caption, cls
    paths={
        "data_prefix_dir": "./data/omni/raw_hf/xxx-dataset/",
        "annotation_path": "./data/omni/raw_hf/xxx-dataset/annotation/xxx.jsonl"
    },
    description="XXX Dataset: N samples for task description"
),
```

> **Multi-split / Multi-sub-benchmark Scenarios (e.g., StreamingBench-Real/Omni/SQA)**
>
> Why split into multiple datasets?
> - **Different sub-benchmarks need different prompt templates / generation configs**:
>   - For example, StreamingBench Real is just video MCQ, Omni needs to emphasize audio, and SQA needs an extra `{sqa_context}` text context.
>   - In this framework, one `dataset_name` corresponds to one unique `user_prompt` in `omni_generation_configs.json`, so semantically different subtasks cannot be lumped under the same dataset.
> - Evaluation and statistics dimensions may differ:
>   - Different sub-benchmarks may want separate overall scores or separate grouping.
>
> In practice, register an independent `DatasetConfig` for each sub-benchmark, sharing `data_prefix_dir` but using different `annotation_path` and generation configs.
> - For example:
>
> ```python
> DatasetConfig(
>     name="streamingbench_real",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_real.jsonl"
>     },
>     ...
> ),
> DatasetConfig(
>     name="streamingbench_omni",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_omni.jsonl"
>     },
>     ...
> ),
> DatasetConfig(
>     name="streamingbench_sqa",
>     paths={
>         "data_prefix_dir": "./data/omni/raw_hf/streamingbench/",
>         "annotation_path": "./data/omni/raw_hf/streamingbench/streaming_sqa.jsonl"
>     },
>     ...
> ),
> ```

### Step 6: Add Dataset Loading Logic

Add to the `load_dataset` function in `o_e_Kit/utils/dataset_loader.py`:

```python
elif dataset_name == "xxx_dataset":
    dataset = OmniEvalDataset(
        annotation_path=args.xxx_dataset_annotation_path,
        data_prefix_dir=args.xxx_dataset_data_prefix_dir,
        dataset_name=dataset_name
    )
```

### Step 7: Add Evaluation Logic (evaluate_omni_datasets / eval.py)

Add to `evaluate_omni_datasets` in `o_e_Kit/utils/evaluation_runner.py`:

```python
# XXX Dataset Evaluation
if getattr(args, 'eval_xxx_dataset', False):
    dataset = load_dataset(args, "xxx_dataset")
    result['xxx_dataset'] = infer_and_evaluate(
        model, dataset, args.model_name, "xxx_dataset", time,
        answer_path=args.answer_path, batch_size=args.batchsize,
        generate_method=args.generate_method
    )
```

> **Grouped Statistics by Task Category (eval.py)**
>
> - If the new dataset needs grouped statistics by certain metadata fields (e.g., `task` / `task_group` / `task_type` / `required_ability`),
>   add them to `group_fields` in `o_e_Kit/utils/eval.py`:
>
> ```python
> group_fields = {
>     ...
>     'ovobench': ['task', 'task_group'],  # Subtask & Category
>     'streamingbench_real': ['task_type', 'required_ability', 'video_categories'],
>     'streamingbench_omni': ['task_type', 'required_ability', 'video_categories'],
>     'streamingbench_sqa': ['task_type', 'required_ability', 'video_categories'],
> }
> ```

### Step 8: Add Generation Configuration (omni_generation_configs.json)

Add to `o_e_Kit/configs/omni_generation_configs.json`:

```json
{
    "mcq": {
        "xxx_dataset": {
            "user_prompt": "{media}\n{question}\n{options}\nPlease select the correct answer from the options above. Only respond with the letter.",
            "system_prompt": "",
            "max_tokens": 128,
            "max_frames": 64,      // Max frames (increase for long videos)
            "max_fps": 1.0,        // Sampling fps (decrease for long videos)
            "load_av": true,       // Whether to load audio+video
            "keep_placeholder": false,
            "interleave_fps": 1.0  // Audio-frame interleave frequency (segments/second), 0 = no interleave
        }
    }
}
```

Configuration notes:
- `max_frames`: Maximum number of frames to sample; can be set to 128 for long videos
- `max_fps`: Sampling frame rate; can be set to 0.5 for long videos (recommended downsampling for StreamingBench / LVBench)
- `load_av`: Whether to extract audio from video. When `true`, the framework looks for an audio file with the same name in the `video_path` directory.
- `keep_placeholder`: Whether to preserve media placeholders (e.g., `<audio_1>`). Used for multi-audio tasks with placeholders (UNO-Bench, AV-Odyssey, etc.).
- `interleave_fps`: Audio-frame interleave frequency (segments/second). e.g., `1.0` means cut one audio segment per second and interleave it into the frame sequence; `0` means no interleave (full audio after all frames).

## Verifying Integration

```bash
# 1. Verify config loading
python3 -c "
from o_e_Kit.utils.args.dataset_args import DATASET_REGISTRY
for c in DATASET_REGISTRY:
    if 'xxx' in c.name:
        print(f'{c.name}: {c.paths}')
"

# 2. Quick verification (small sample)
python3 eval_main.py --eval_xxx_dataset --max_sample_num 5

# 3. Full evaluation
python3 eval_main.py --eval_xxx_dataset --max_sample_num 100
```

## Integrated Dataset Examples

| Dataset | Samples | Type | Conversion Script |
|---------|---------|------|----------|
| Daily-Omni | 1,197 | MCQ | convert_daily_omni.py |
| OmniBench | 1,142 | MCQ | convert_omnibench.py |
| WorldSense | 3,172 | MCQ | (jsonl already exists) |
| AV-Odyssey | 4,555 | MCQ | convert_av_odyssey.py |
| UNO-Bench MC | 1,000 | MCQ | convert_unobench_mc.py |
| Video-Holmes | 1,837 | MCQ | (jsonl already exists) |
| AVUT-Benchmark Human | 1,734 | MCQ | convert_avut_benchmark.py |
| AVUT-Benchmark Gemini | 9,874 | MCQ | convert_avut_benchmark.py |
| LVBench | 1,549 | MCQ | convert_lvbench.py |
| OVO-Bench (Omni) | - | MCQ/QA | convert_ovobench_omni.py |
| StreamingBench-Real (Offline) | - | MCQ | convert_streamingbench_real.py |
| StreamingBench-Omni (Offline) | - | MCQ | convert_streamingbench_omni.py |
| StreamingBench-SQA (Offline) | - | MCQ | convert_streamingbench_sqa.py |

## Unified JSONL Format Specification

### MCQ (Multiple Choice)

```json
{
  "VideoPath": "path/to/video.mp4",
  "WavPath": "",
  "ImagePath": "",
  "dataset_type": "mcq",
  "dataset_name": "dataset_name",
  "question": "question text",
  "choices": ["Option A content", "Option B content", "Option C content", "Option D content"],
  "gt_answer": "A"
}
```

### Open QA (Open-ended Question Answering)

```json
{
  "VideoPath": "path/to/video.mp4",
  "WavPath": "",
  "dataset_type": "open_qa",
  "dataset_name": "dataset_name",
  "question": "question text",
  "gt_answer": "full answer text"
}
```

### Path Dictionary Format (Multimedia)

Used for scenarios with multiple audio/images/videos (e.g., UNO-Bench, AV-Odyssey):

```json
{
  "audio_paths_dict": {"<audio_1>": "path/to/audio1.wav", "<audio_2>": "path/to/audio2.wav"},
  "image_paths_dict": {"<image_1>": "path/to/image1.jpg"},
  "video_paths_dict": {"<video_1>": "path/to/video1.mp4"},
  "question": "Listen to <audio_1> and look at <image_1>, ...",
  "choices": [...],
  "gt_answer": "B"
}
```

### Recommended Fields for Long Video / Timestamp Tasks (Optional but Strongly Recommended)

- **OVOBench**:
  - `task`: Fine-grained task type (EPM / ASI / HLD / STU / OJR / ATR / ACR / OCR / FPD / REC / SSR / CRR)
  - `task_group`: High-level task category (e.g., `Backward Tracing`, `Real-Time Visual Perception`, etc.)
  - `realtime`: Official time field, used for video trimming or statistics.

- **StreamingBench-Real/Omni**:
  - `task_type`: Official task type (Clips Summarize / Object Recognition / ...)
  - `required_ability`: Ability label (episodic memory / working memory / ...)
  - `video_categories`: Video scene category (preparation_of_meals / playing_card / ...)
  - `time_range`: Original annotation time range string ("[0:00:00 - 0:01:00]")
  - `time_stamp`: Current question timestamp ("HH:MM:SS")

- **StreamingBench-SQA (Sequential QA)**:
  - In addition to the above fields, it is recommended to add:
    - `sqa_context`: Textual history QA context, used to replicate the official SQA logic in the prompt, e.g.:
      - `"Here are the contextual information ... At timestamp 00:00:36, the following question and answer occurred: Question: ...; Options: ...; Answer: A; ..."`

## Notes

1. **Option format**: `choices` list does not include letter prefixes; letters are dynamically added during model inference.
2. **Answer format**: MCQ `gt_answer` only stores the letter (A/B/C/D).
3. **Path verification**: Optionally verify whether media files exist during conversion.
4. **Dual output**: Output both `.jsonl` (single-line) and `.pretty.jsonl` (formatted) versions.
5. **Data directory**: Evaluation data is downloaded from HuggingFace via `scripts/hf_download.py`; can also be placed manually during development.
6. **Audio extraction** ⚠️:
   - When `load_av=true`, the framework automatically looks for `.wav` files in the same directory as the video.
   - Audio lookup rule: `xxx.mp4` → `xxx.wav` / `xxx.mp3` / `xxx.m4a` / `xxx.flac`
   - If audio does not exist, **it must be extracted in advance**, otherwise `load_av` will have no audio to load.
   - Use `playground/extract_audio_batch.py` for batch extraction.
   - Or add the `--extract-audio` parameter to the conversion script.
7. **Filename consistency** ⚠️:
   - Ensure that `VideoPath` in annotations matches the actual filename on disk (including case and extension).
   - For official annotations with segment suffixes (e.g., `_1-25` / `_26-50`) but only a single full video on disk:
     - Either generate the corresponding segment files during download/extraction as per official instructions;
     - Or add a "normalization" logic in `convert_xxx.py` (see `normalize_video_name` practice in StreamingBench integration).
   - After conversion, carefully check `missing_video` / `missing_clip` in the statistics. If they are not 0, prioritize investigating the issue.
