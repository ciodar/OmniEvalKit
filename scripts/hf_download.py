#!/usr/bin/env python3
"""
OmniEvalKit HuggingFace dataset download & restore script

Download Parquet datasets from HuggingFace repos and restore them to the local data/ directory structure
so the framework can use them directly.

Usage:
    # Download all datasets (audio + image, excluding video)
    python scripts/hf_download.py --output_dir ./data

    # Download only specific datasets
    python scripts/hf_download.py --datasets omnibench,daily_omni --output_dir ./data

    # Download with videos
    python scripts/hf_download.py --output_dir ./data --download_videos

    # List available datasets
    python scripts/hf_download.py --list
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_REPO_ID = "OmniEvalKit/omnievalkit-dataset"

# Override mapping for video source repos that have moved
VIDEO_REPO_OVERRIDES = {
    "DailyOmni/Daily-Omni": "liarliar/Daily-Omni",
}


def get_hf_api():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        return HfApi(), hf_hub_download, snapshot_download
    except ImportError:
        print("Error: please install huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def download_dataset_info(repo_id: str) -> Dict[str, Any]:
    """Download and parse dataset_info.json"""
    _, hf_hub_download, _ = get_hf_api()
    
    info_path = hf_hub_download(
        repo_id=repo_id,
        filename="dataset_info.json",
        repo_type="dataset",
    )
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_datasets(info: Dict[str, Any]):
    """List all available datasets"""
    datasets = info.get("datasets", {})
    
    print(f"Repo: {info.get('repo_id', '?')}")
    print(f"Version: {info.get('version', '?')}")
    print(f"Total datasets: {info.get('total_datasets', len(datasets))}")
    print()
    
    by_category = {}
    for name, ds in datasets.items():
        cat = ds.get("category", "other")
        by_category.setdefault(cat, []).append(ds)
    
    for cat in sorted(by_category.keys()):
        ds_list = by_category[cat]
        print(f"[{cat}] ({len(ds_list)} datasets)")
        for ds in sorted(ds_list, key=lambda x: x.get("name", "")):
            video_tag = " [video needs download]" if ds.get("has_video") else ""
            size = ds.get("size_mb", 0)
            size_str = f"{size:.0f}MB" if size < 1024 else f"{size/1024:.1f}GB"
            print(f"  {ds['name']:<35s} {ds.get('num_samples', '?'):>6} samples  {size_str:>8}{video_tag}")
        print()


def download_and_restore_dataset(
    ds_info: Dict[str, Any],
    repo_id: str,
    output_dir: str,
    cache_dir: Optional[str] = None,
):
    """Download a single dataset's Parquet and restore"""
    from parquet_to_jsonl import parquet_to_jsonl
    _, hf_hub_download, _ = get_hf_api()
    
    name = ds_info["name"]
    hf_path = ds_info["hf_path"]
    num_shards = ds_info.get("num_shards", 1)
    restore_to = ds_info.get("restore_to", {})
    
    annotation_path = restore_to.get("annotation_path", "")
    data_prefix_dir = restore_to.get("data_prefix_dir", "")
    
    project_root = os.path.abspath(os.path.join(output_dir, ".."))
    
    def _resolve_restore_path(path: str, hf_path: str, is_annotation: bool) -> str:
        """Resolve restore_to path to an absolute path, compatible with legacy absolute paths"""
        if not path:
            return ""
        if path.startswith("./"):
            path = path[2:]
        if os.path.isabs(path):
            # Legacy dataset_info.json may contain absolute paths; infer the correct relative path from hf_path
            basename = os.path.basename(path)
            if is_annotation:
                path = os.path.join("data", hf_path, basename)
            else:
                path = os.path.join("data", hf_path)
            print(f"  Warning: restore_to contains absolute path, auto-corrected to {path}")
        return os.path.normpath(os.path.join(project_root, path))
    
    abs_annotation = _resolve_restore_path(annotation_path, hf_path, is_annotation=True)
    abs_data_prefix = _resolve_restore_path(data_prefix_dir, hf_path, is_annotation=False)
    if not abs_data_prefix:
        abs_data_prefix = os.path.abspath(output_dir)
    
    if os.path.exists(abs_annotation):
        print(f"  Skipped (already exists): {abs_annotation}")
        return {"name": name, "status": "skipped"}
    
    tmp_dir = tempfile.mkdtemp(prefix=f"hf_{name}_")
    
    try:
        split = ds_info.get("split", "test")
        downloaded_files = []
        # Auto-detect shards from the repo, falling back to num_shards from dataset_info
        try:
            api, _, _ = get_hf_api()
            all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
            prefix = f"{hf_path}/{split}-"
            shard_files = sorted(
                f for f in all_files
                if f.startswith(prefix) and f.endswith(".parquet")
            )
        except Exception:
            shard_files = []
        if shard_files:
            detected_shards = len(shard_files)
            if detected_shards != num_shards:
                print(f"  Auto-detected {detected_shards} shards (dataset_info says {num_shards})")
            for shard_name in shard_files:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_name,
                    cache_dir=cache_dir,
                    repo_type="dataset",
                )
                downloaded_files.append(local)
        else:
            for i in range(num_shards):
                shard_name = f"{hf_path}/{split}-{i:05d}-of-{num_shards:05d}.parquet"
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=shard_name,
                    cache_dir=cache_dir,
                    repo_type="dataset",
                )
                downloaded_files.append(local)
        
        if len(downloaded_files) > 1:
            merged_dir = os.path.join(tmp_dir, "shards")
            os.makedirs(merged_dir, exist_ok=True)
            for f in downloaded_files:
                shutil.copy2(f, merged_dir)
            parquet_input = merged_dir
        else:
            parquet_input = downloaded_files[0]
        
        os.makedirs(os.path.dirname(abs_annotation) or '.', exist_ok=True)
        os.makedirs(abs_data_prefix, exist_ok=True)
        
        result = parquet_to_jsonl(
            input_path=parquet_input,
            output_path=abs_annotation,
            media_output_dir=abs_data_prefix,
            restore_media=True,
            data_prefix_dir=abs_data_prefix,
        )
        
        return {
            "name": name,
            "status": "success",
            "annotation_path": abs_annotation,
            "data_prefix_dir": abs_data_prefix,
            "count": result.get("count", 0),
            "stats": result.get("stats", {}),
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": name, "status": "error", "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_videos_for_dataset(ds_info: Dict[str, Any], output_dir: str):
    """Attempt to download videos"""
    video_source = ds_info.get("video_source")
    if not video_source:
        print(f"  No video source info, please download manually")
        return
    
    src_type = video_source.get("type", "")
    
    if src_type == "hf_dataset":
        repo_id = video_source.get("repo_id", "")
        repo_id = VIDEO_REPO_OVERRIDES.get(repo_id, repo_id)
        instructions = video_source.get("instructions", "")
        print(f"  Video source: {repo_id}")
        print(f"  Instructions: {instructions}")
        
        try:
            _, _, snapshot_download = get_hf_api()
            
            restore_to = ds_info.get("restore_to", {})
            data_prefix_dir = restore_to.get("data_prefix_dir", "")
            if data_prefix_dir.startswith("./"):
                data_prefix_dir = data_prefix_dir[2:]
            project_root = os.path.abspath(os.path.join(output_dir, ".."))
            if os.path.isabs(data_prefix_dir):
                hf_path = ds_info.get("hf_path", "")
                data_prefix_dir = os.path.join("data", hf_path)
            target_dir = os.path.normpath(os.path.join(project_root, data_prefix_dir))
            
            print(f"  Downloading videos from {repo_id} to {target_dir} ...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                ignore_patterns=["*.parquet", "*.jsonl", "*.json", "*.md"],
            )
            print(f"  Video download complete")
        except Exception as e:
            print(f"  Auto-download failed: {e}")
            print(f"  Please run manually: {instructions}")
    else:
        instructions = video_source.get("instructions", "Please refer to the dataset documentation to download manually")
        print(f"  {instructions}")


def main():
    parser = argparse.ArgumentParser(description="OmniEvalKit dataset download & restore")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HuggingFace repo ID (default {DEFAULT_REPO_ID})")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output root directory (default ./data)")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Cache directory (default: system default)")
    parser.add_argument("--download_videos", action="store_true",
                        help="Attempt to download video files from original sources")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="Only list available datasets, do not download")
    args = parser.parse_args()
    
    print(f"Fetching dataset info: {args.repo_id}")
    info = download_dataset_info(args.repo_id)
    
    if args.list_only:
        list_datasets(info)
        return
    
    all_datasets = info.get("datasets", {})
    
    if args.datasets:
        names = set(args.datasets.split(","))
        selected = {k: v for k, v in all_datasets.items() if k in names}
        missing = names - set(selected.keys())
        if missing:
            print(f"Warning: the following datasets were not found: {', '.join(missing)}")
    else:
        selected = all_datasets
    
    if not selected:
        print("No matching datasets")
        return
    
    print("=" * 60)
    print(f"OmniEvalKit Dataset Download")
    print(f"Repo: {args.repo_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Selected datasets: {len(selected)}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    video_needed = []
    
    for i, (name, ds_info) in enumerate(sorted(selected.items())):
        print(f"\n[{i+1}/{len(selected)}] Downloading {ds_info.get('display_name', name)} ({name})")
        
        result = download_and_restore_dataset(ds_info, args.repo_id, args.output_dir, args.cache_dir)
        status = result.get("status")
        
        if status == "success":
            success_count += 1
            print(f"  Done: {result.get('count', '?')} samples -> {result.get('annotation_path', '')}")
        elif status == "skipped":
            skip_count += 1
        else:
            fail_count += 1
            print(f"  Failed: {result.get('error', 'unknown')}")
        
        if ds_info.get("has_video"):
            video_needed.append(ds_info)
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Success: {success_count}")
    print(f"Skipped (already exists): {skip_count}")
    print(f"Failed: {fail_count}")
    
    if video_needed:
        print(f"\nDatasets that need additional video downloads ({len(video_needed)}):")
        for ds in video_needed:
            src = ds.get("video_source")
            hint = f" → {src['repo_id']}" if src and src.get("repo_id") else ""
            print(f"  - {ds['name']}{hint}")
        
        if args.download_videos:
            print("\nDownloading videos...")
            for ds in video_needed:
                print(f"\nDownloading video: {ds['name']}")
                download_videos_for_dataset(ds, args.output_dir)
        else:
            print(f"\nTip: use --download_videos to auto-download videos")
    
    print(f"\nData restored to: {os.path.abspath(args.output_dir)}")
    print("You can now run evaluations!")


if __name__ == "__main__":
    main()
