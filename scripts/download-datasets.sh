#!/bin/bash
## Request 1 task
#SBATCH -n 1                 # (or --ntasks=1) 
#SBATCH -c 4                 # (or --cpus-per-task=4)
## Request the "compute" partition
## (optional, as this is the default partition)
#SBATCH -p compute         # (or --partition=compute)

## Request 1 hour runtime
#SBATCH -t 3:0:0             # (or --time=1:0:0)

## Request 1GB RAM per task
#SBATCH --mem-per-cpu=64G

# Clean up any residual environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load python

source .venv/bin/activate

python scripts/hf_download.py --output_dir ./data --cache_dir /gpfs/scratch/qp252970/.cache/huggingface/datasets --download_videos