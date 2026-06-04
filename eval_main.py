"""
OmniEvalKit main evaluation program

Supports evaluation of multiple datasets, including:
- Audio datasets: GigaSpeech, WeNetSpeech, AudioQA1M, etc.
- Video datasets: OVOBench, StreamingBench
- Multimodal datasets: VisionCap, OmniCap, LiveCC, AVEvent, etc.

StreamingBench supports four task types:
1. real: Real-time visual understanding - visual QA based on current moment
2. omni: Omni-modal understanding - multimodal QA combining video and audio
3. sqa: Sequential QA - continuous dialogue based on video history
4. proactive: Proactive output - special loop reasoning logic where model actively decides when to output

Full proactive task adaptation:
- Inference: Uses generate_proactive method, implementing cyclic timing judgment + two-stage QA
- Evaluation: Uses StreamingProactiveEval evaluator, considering both timing and content accuracy

Usage examples:
python eval_main.py --model_type minicpmo_chat --eval_streamingbench --streamingbench_tasks proactive
python eval_main.py --eval_streamingbench --streamingbench_tasks real omni sqa proactive
"""

import torch
import numpy as np
import os
import datetime

from o_e_Kit.utils.get_args import parse_args
from o_e_Kit.utils.model_loader import load_model
from o_e_Kit.utils.evaluation_runner import run_all_evaluations, save_evaluation_results


def main(args):
    """Main: initialize environment, load model, run evaluations"""
    
    # Set timestamp
    if args.prefix == "":
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    else:
        time = args.prefix
    
    # Initialize distributed training
    # Increase NCCL timeout to avoid hangs due to uneven inference speed
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
        timeout=datetime.timedelta(hours=2),  # Set 2-hour timeout
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    print(f'Init Rank-{torch.distributed.get_rank()}')
    
    if torch.distributed.is_initialized():
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    
    # Load model
    model = load_model(args, args.device)
    
    # Run all evaluations
    result = run_all_evaluations(args, model, args.device, time)
    
    # Sync all processes
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Only save results on main process
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    # Save evaluation results
    save_evaluation_results(result, args, time)
    
    # Clean up distributed training
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    main(args)