import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from o_e_Kit.datasets.omniduplexeval_dataset import OmniDuplexEvalDataset, task_type_from_split


def infer_omniduplexeval(
    model,
    dataset: OmniDuplexEvalDataset,
    response_root: str,
    dataset_name: str,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    """Run streaming inference on Omni-DuplexEval samples and save responses.

    For each sample:
      1. Materialize video (and optionally audio).
      2. Run duplex streaming generation via ``model.generate()``.
      3. Save the model response to ``{response_root}/{split}/{id}.json``.

    Returns a list of result dicts with status per sample.
    """
    split = dataset.split
    split_dir = Path(response_root) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    total = len(dataset)

    for idx in tqdm(range(total), desc=f"Omni-DuplexEval [{split}]"):
        _, paths, annotation = dataset[idx]
        sample_id = annotation["id"]
        out_path = split_dir / f"{sample_id}.json"

        if skip_existing and out_path.exists():
            results.append({"split": split, "id": sample_id, "status": "skipped"})
            continue

        try:
            response_data = _run_single_sample(model, paths, annotation)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            results.append({"split": split, "id": sample_id, "status": "success"})
        except Exception as exc:
            _write_error_response(out_path, annotation, str(exc))
            results.append({"split": split, "id": sample_id, "status": "failed", "error": str(exc)})

    return results


def _run_single_sample(model, paths: Dict[str, str], annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run duplex generation on one sample and return the response list.

    The returned list matches Omni-DuplexEval's expected format:
      ``[{"sentence": "...", "start": t1, "end": t2}, ...]``
    """
    dataset_name = annotation["dataset_name"]

    generation_kwargs = {
        "dataset_name": dataset_name,
        "paths": [paths],
        "items": [annotation],
    }

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    if not outputs or not isinstance(outputs, list):
        return []

    output = outputs[0]
    if not isinstance(output, dict):
        return []

    # Prefer structured CTC data when available (RTD tasks)
    ctc_data = output.get("prediction_ctc_data")
    if ctc_data and isinstance(ctc_data, list) and len(ctc_data) > 0:
        return ctc_data

    # Fall back to plain text response (PR tasks) as a single segment
    response_text = output.get("response", "")
    if response_text:
        return [{"sentence": response_text.strip(), "start": 0.0, "end": float(annotation.get("video_duration", 0) or 0)}]

    return []


def _write_error_response(out_path: Path, annotation: Dict[str, Any], error_msg: str) -> None:
    fallback = [{"sentence": "", "start": 0.0, "end": 0.0}]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fallback, f, ensure_ascii=False, indent=2)
