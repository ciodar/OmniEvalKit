#!/usr/bin/env python3
"""
Convert Full-Duplex-Bench v1/v1.5 directory structure to OmniEvalKit JSONL format.

Usage:
    python convert_to_jsonl.py --data-dir /path/to/downloaded/data \
        --output-dir /path/to/output/jsonl

The data directory is expected to contain:
    v1_0/{task}/{sample_id}/input.wav
    v1_5/{task}/{sample_id}/input.wav  (+ clean_input.wav, metadata.json)
"""

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path


def find_v1_0_samples(data_dir: str):
    """Discover all v1.0 samples."""
    samples = []
    v1_0_dir = os.path.join(data_dir, "v1.0")
    if not os.path.isdir(v1_0_dir):
        print(f"[WARN] v1_0 directory not found: {v1_0_dir}")
        return samples

    # Map directory names to canonical task names used in evaluation
    task_map = {
        "candor_pause_handling": "pause_handling",
        "synthetic_pause_handling": "pause_handling",
        "candor_turn_taking": "smooth_turn_taking",
        "icc_backchannel": "backchannel",
        "synthetic_user_interruption": "user_interruption",
    }

    for dir_name, task_name in task_map.items():
        task_dir = os.path.join(v1_0_dir, dir_name)
        if not os.path.isdir(task_dir):
            continue
        for sample_dir in sorted(glob(os.path.join(task_dir, "*"))):
            if not os.path.isdir(sample_dir):
                continue
            sample_id = os.path.basename(sample_dir)
            input_wav = os.path.join(sample_dir, "input.wav")
            if not os.path.isfile(input_wav):
                continue

            rel_path = os.path.relpath(input_wav, data_dir)
            samples.append({
                "WavPath": rel_path,
                "task": task_name,
                "fdb_version": "v1_0",
                "sample_id": sample_id,
                "source_dir": dir_name,
            })
    return samples


def find_v1_5_samples(data_dir: str):
    """Discover all v1.5 samples, including metadata."""
    samples = []
    v1_5_dir = os.path.join(data_dir, "v1.5")
    if not os.path.isdir(v1_5_dir):
        print(f"[WARN] v1_5 directory not found: {v1_5_dir}")
        return samples

    for task_dir in sorted(glob(os.path.join(v1_5_dir, "*"))):
        if not os.path.isdir(task_dir):
            continue
        task_name = os.path.basename(task_dir)
        for sample_dir in sorted(glob(os.path.join(task_dir, "*"))):
            if not os.path.isdir(sample_dir):
                continue
            sample_id = os.path.basename(sample_dir)
            input_wav = os.path.join(sample_dir, "input.wav")
            if not os.path.isfile(input_wav):
                continue

            rel_path = os.path.relpath(input_wav, data_dir)

            entry = {
                "WavPath": rel_path,
                "task": task_name,
                "fdb_version": "v1_5",
                "sample_id": sample_id,
            }

            # Add clean_input.wav if exists
            clean_input = os.path.join(sample_dir, "clean_input.wav")
            if os.path.isfile(clean_input):
                entry["clean_input_wav"] = os.path.relpath(clean_input, data_dir)

            # Load metadata.json if exists
            metadata_path = os.path.join(sample_dir, "metadata.json")
            if os.path.isfile(metadata_path):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    entry["context_text"] = metadata.get("context_text", "")
                    entry["current_turn_text"] = metadata.get("current_turn_text", "")
                    entry["timestamps"] = metadata.get("timestamps", [])
                except (json.JSONDecodeError, IOError) as e:
                    print(f"[WARN] Failed to read metadata: {metadata_path} - {e}")

            samples.append(entry)
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Convert Full-Duplex-Bench data to OmniEvalKit JSONL"
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Root directory containing v1_0/ and v1_5/ subdirectories"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for JSONL files (default: <data-dir>/jsonl)"
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_dir = args.output_dir or os.path.join(data_dir, "jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Discover samples
    v1_0_samples = find_v1_0_samples(data_dir)
    v1_5_samples = find_v1_5_samples(data_dir)
    all_samples = v1_0_samples + v1_5_samples

    print(f"Found {len(v1_0_samples)} v1.0 samples, {len(v1_5_samples)} v1.5 samples")

    if not all_samples:
        print("[ERROR] No samples found. Check --data-dir path.")
        sys.exit(1)

    # Group by dataset name (fdb_v1_<task>, fdb_v15_<task>)
    dataset_map = {}
    for s in all_samples:
        version_prefix = s["fdb_version"].replace(".", "_").replace("v", "fdb_v")
        task = s["task"]
        dataset_name = f"{version_prefix}_{task}"
        dataset_map.setdefault(dataset_name, []).append(s)

    # Write JSONL files
    for dataset_name, samples in sorted(dataset_map.items()):
        output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
        with open(output_path, "w") as f:
            for s in samples:
                # Remove internal fields not needed in JSONL
                entry = {k: v for k, v in s.items()
                         if k not in ("fdb_version", "sample_id", "source_dir")}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {dataset_name}: {len(samples)} samples -> {output_path}")

    print(f"\nDone. Generated {len(dataset_map)} JSONL files in {output_dir}")


if __name__ == "__main__":
    main()
