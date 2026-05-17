import os
from datasets import load_dataset

# Set your target path explicitly to your scratch space
cache_dir = "/gpfs/scratch/qp252970/.cache/huggingface/datasets"
output_dir = "/gpfs/scratch/qp252970/data/audio/asr/gigaspeech/test_files"
os.makedirs(output_dir, exist_ok=True)

print("Downloading GigaSpeech test split...")
# 'xs' configuration is perfect if you only want validation and test sets
dataset = load_dataset(
    "speechcolab/gigaspeech", 
    "xs", 
    split="test", 
    cache_dir=cache_dir,
    token=True # Uses your saved 'huggingface-cli login' token
)

print(f"Dataset downloaded. Converting and saving metadata structure to {output_dir}...")
# Loop over the samples to extract wav files and write out your test.jsonl file
# matching OmniEvalKit's target schema.