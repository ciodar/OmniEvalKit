import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


RTD_SPLITS = [
    "RTD_world_knowledge",
    "RTD_counting",
    "RTD_fine_grained_movement",
    "RTD_interaction_relation",
    "RTD_OCR",
    "RTD_Omni",
]

PR_SPLITS = [
    "PR_correction",
    "PR_event_reminder",
    "PR_post_event_reminder",
]

ALL_SPLITS = RTD_SPLITS + PR_SPLITS


def task_type_from_split(split: str) -> str:
    split_lower = split.lower()
    if "correction" in split_lower:
        return "correction"
    if "post_event" in split_lower:
        return "post_event_reminder"
    if "rtd" in split_lower:
        return "real_time_description"
    return "proactive_reminder"


def is_rtd_split(split: str) -> bool:
    return split in RTD_SPLITS


def is_pr_split(split: str) -> bool:
    return split in PR_SPLITS


class OmniDuplexEvalDataset(Dataset):
    """Streaming dataset for Omni-DuplexEval benchmark.

    Loads data from HuggingFace ``Hothan/Omni-DuplexEval``, materializes
    video and audio files to disk, and returns standard ``(idx, paths, annotation)`` triplets.
    """

    def __init__(
        self,
        hf_dataset_name: str = "Hothan/Omni-DuplexEval",
        split: str = "RTD_OCR",
        media_dir: str = "/tmp/omniduplexeval_media",
        max_samples: Optional[int] = None,
        hf_token: Optional[str] = None,
    ):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install 'datasets' to use OmniDuplexEvalDataset: pip install datasets")

        if split not in ALL_SPLITS:
            raise ValueError(f"Unknown Omni-DuplexEval split '{split}'. Available: {ALL_SPLITS}")

        self.split = split
        self.media_dir = media_dir
        Path(media_dir).mkdir(parents=True, exist_ok=True)

        load_kwargs = {"path": hf_dataset_name, "split": split}
        if hf_token:
            load_kwargs["token"] = hf_token

        self.ds = load_dataset(**load_kwargs)

        if max_samples is not None and max_samples > 0 and max_samples < len(self.ds):
            self.ds = self.ds.select(range(max_samples))

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[int, Dict[str, str], Dict[str, Any]]:
        row = self.ds[idx]
        sample_id = str(row["id"])

        video_path = self._materialize_video(row.get("video"), sample_id)
        audio_path = self._materialize_audio(row.get("question_audio"), sample_id)

        paths: Dict[str, str] = {}
        if video_path:
            paths["video_path"] = video_path
        if audio_path:
            paths["audio_path"] = audio_path

        annotation = {
            "dataset_name": f"omniduplexeval_{self.split.lower()}",
            "question_text": row.get("question_text", ""),
            "answer1": row.get("answer1", ""),
            "answer2": row.get("answer2", ""),
            "reminder1": _parse_float(row.get("reminder1")),
            "reminder2": _parse_float(row.get("reminder2")),
            "video_type": row.get("video_type", ""),
            "video_duration": _parse_float(row.get("video_duration", 0)),
            "split": self.split,
            "id": sample_id,
            "task_type": task_type_from_split(self.split),
        }

        return idx, paths, annotation

    def _materialize_video(self, video_value: Any, sample_id: str) -> Optional[str]:
        if video_value is None:
            return None

        output_path = os.path.join(self.media_dir, f"{sample_id}.mp4")

        if isinstance(video_value, str):
            if os.path.exists(video_value):
                return video_value
            return None

        if isinstance(video_value, dict):
            path = video_value.get("path")
            if path and os.path.exists(path):
                return str(path)
            bytes_value = video_value.get("bytes")
            if bytes_value:
                with open(output_path, "wb") as f:
                    f.write(bytes_value)
                return output_path

        path_attr = getattr(video_value, "path", None)
        if path_attr and os.path.exists(path_attr):
            return str(path_attr)

        if isinstance(video_value, (bytes, bytearray)):
            with open(output_path, "wb") as f:
                f.write(video_value)
            return output_path

        return None

    def _materialize_audio(self, audio_value: Any, sample_id: str) -> Optional[str]:
        if audio_value is None:
            return None

        output_path = os.path.join(self.media_dir, f"{sample_id}.wav")

        if isinstance(audio_value, str):
            if os.path.exists(audio_value):
                return audio_value
            return None

        if isinstance(audio_value, dict):
            path = audio_value.get("path")
            if path and os.path.exists(path):
                return str(path)
            bytes_value = audio_value.get("bytes")
            if bytes_value:
                with open(output_path, "wb") as f:
                    f.write(bytes_value)
                return output_path

        path_attr = getattr(audio_value, "path", None)
        if path_attr and os.path.exists(path_attr):
            return str(path_attr)

        if isinstance(audio_value, (bytes, bytearray)):
            with open(output_path, "wb") as f:
                f.write(audio_value)
            return output_path

        return None

    def cleanup_media(self, keep: bool = False) -> None:
        if keep:
            return
        import shutil
        if os.path.isdir(self.media_dir):
            shutil.rmtree(self.media_dir, ignore_errors=True)

    @staticmethod
    def get_split_names(split_filter: Optional[str] = None) -> List[str]:
        if split_filter == "rtd":
            return list(RTD_SPLITS)
        if split_filter == "pr":
            return list(PR_SPLITS)
        if split_filter:
            return [s for s in ALL_SPLITS if split_filter.lower() in s.lower()]
        return list(ALL_SPLITS)


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except (ValueError, AttributeError):
            return None
    return None
