"""
Evaluation runner module
Runs evaluations for all datasets
Each dataset is handled explicitly for clarity and maintainability

Supports async evaluation mode controlled by the async_evaluate parameter:
- async_evaluate=False: Sync evaluation (default, backward compatible)
- async_evaluate=True: Async evaluation, returns immediately after inference, evaluation runs in background thread
"""

import torch
import json
import os
from o_e_Kit.utils.infer import infer_and_evaluate, wait_all_evaluations, get_pending_eval_count
from o_e_Kit.utils.dataset_loader import load_dataset
from o_e_Kit.utils.model_loader import load_model
from o_e_Kit.utils.evaluation_runner_audio import evaluate_all_audio_datasets


def evaluate_video_datasets(args, model, device, time, async_evaluate: bool = False):
    """Evaluate video datasets
    
    Args:
        async_evaluate: Whether to use async evaluation mode
    """
    result = {}
    
    # StreamingBench evaluation
    if args.eval_streamingbench:
        for task in args.streamingbench_tasks:
            dataset = load_dataset(args, "StreamingBench", task)
            result_key = f'StreamingBench-{task.upper()}'
            
            # Select inference method based on task type
            if task == "proactive":
                generate_method = "proactive"
                print(f"  -> Using proactive inference mode (cyclic timing judgment + proactive output)")
            else:
                generate_method = "chat"
                print(f"  -> Using chat inference mode (standard QA evaluation)")
            
            result[result_key] = infer_and_evaluate(
                model, dataset, args.model_name, result_key, time, 
                answer_path=args.answer_path, batch_size=args.batchsize, generate_method=generate_method,
                async_evaluate=async_evaluate
            )
            print(f"StreamingBench-{task.upper()} {'inference' if async_evaluate else 'evaluation'} completed")
    
    return result


def evaluate_duplex_datasets(args, device, time, async_evaluate: bool = False):
    """Evaluate Duplex datasets
    
    Args:
        async_evaluate: Whether to use async evaluation mode
    """
    result = {}
    
    # LiveSports-3K CC duplex evaluation (unified format)
    if getattr(args, 'eval_livesports3k_cc', False):
        dataset = load_dataset(args, "livesports3k_cc")
        duplex_model = load_model(args, device, duplex_type="video")
        result['livesports3k_cc'] = infer_and_evaluate(
            duplex_model, dataset, args.model_name, "livesports3k_cc", time,
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OV-AVEL: Open-Vocabulary Audio-Visual Event Localization
    if getattr(args, 'eval_ovavel', False):
        dataset = load_dataset(args, "ovavel")
        duplex_model = load_model(args, device, duplex_type="omni")
        result['ovavel'] = infer_and_evaluate(
            duplex_model, dataset, args.model_name, "ovavel", time,
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    return result


# Full-Duplex-Bench v1/v1.5 dataset names (8 tasks)
FDB_TASKS = [
    "fdb_v1_pause_handling",
    "fdb_v1_backchannel",
    "fdb_v1_smooth_turn_taking",
    "fdb_v1_user_interruption",
    "fdb_v15_user_interruption",
    "fdb_v15_user_backchannel",
    "fdb_v15_talking_to_other",
    "fdb_v15_background_speech",
]


def evaluate_fdb_datasets(args, model, time, async_evaluate: bool = False):
    """
    Evaluate Full-Duplex-Bench v1/v1.5 datasets.
    Inference produces output.wav alongside input.wav via the model's TTS module.
    Actual FDB metrics evaluation is done separately via the FDB evaluation scripts
    (Full-Duplex-Bench/v1_v1.5/evaluation/evaluate.py).
    
    Uses evaluate=False to skip built-in evaluation (FDB has its own metrics pipeline).
    """
    result = {}
    for task in FDB_TASKS:
        eval_flag = f"eval_{task}"
        if getattr(args, eval_flag, False):
            dataset = load_dataset(args, task)
            result[task] = infer_and_evaluate(
                model, dataset, args.model_name, task, time,
                answer_path=args.answer_path, batch_size=args.batchsize,
                generate_method=args.generate_method,
                evaluate=False,
                async_evaluate=async_evaluate
            )
    return result


def evaluate_omni_datasets(args, model, time, async_evaluate: bool = False):
    """Evaluate Omni datasets
    
    Args:
        async_evaluate: Whether to use async evaluation mode
    """
    result = {}
    
    if args.eval_livesports3k_cc:
        dataset = load_dataset(args, "livesports3k_cc")
        result['livesports3k_cc'] = infer_and_evaluate(
            model, dataset, args.model_name, "livesports3k_cc", time, 
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method, evaluate=False,
            async_evaluate=async_evaluate
        )
    
    # Daily-Omni evaluation
    if getattr(args, 'eval_daily_omni', False):
        dataset = load_dataset(args, "daily_omni")
        result['daily_omni'] = infer_and_evaluate(
            model, dataset, args.model_name, "daily_omni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OmniBench evaluation
    if getattr(args, 'eval_omnibench', False):
        dataset = load_dataset(args, "omnibench")
        result['omnibench'] = infer_and_evaluate(
            model, dataset, args.model_name, "omnibench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # UNO-Bench evaluation
    if getattr(args, 'eval_unobench', False):
        dataset = load_dataset(args, "unobench")
        result['unobench'] = infer_and_evaluate(
            model, dataset, args.model_name, "unobench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )

    # UNO-Bench MCQ evaluation
    if getattr(args, 'eval_unobench_mc', False):
        dataset = load_dataset(args, "unobench_mc")
        result['unobench_mc'] = infer_and_evaluate(
            model, dataset, args.model_name, "unobench_mc", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # WorldSense evaluation
    if getattr(args, 'eval_worldsense', False):
        dataset = load_dataset(args, "worldsense")
        result['worldsense'] = infer_and_evaluate(
            model, dataset, args.model_name, "worldsense", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AV-Odyssey evaluation
    if getattr(args, 'eval_av_odyssey', False):
        dataset = load_dataset(args, "av_odyssey")
        result['av_odyssey'] = infer_and_evaluate(
            model, dataset, args.model_name, "av_odyssey", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-MME evaluation
    if getattr(args, 'eval_videomme', False):
        dataset = load_dataset(args, "videomme")
        result['videomme'] = infer_and_evaluate(
            model, dataset, args.model_name, "videomme", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-MME Short evaluation (only duration=short subset)
    if getattr(args, 'eval_videomme_short', False):
        dataset = load_dataset(args, "videomme_short")
        result['videomme_short'] = infer_and_evaluate(
            model, dataset, args.model_name, "videomme_short", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # JointAVBench (audio-visual joint understanding MCQ) evaluation
    if getattr(args, 'eval_jointavbench', False):
        dataset = load_dataset(args, "jointavbench")
        result['jointavbench'] = infer_and_evaluate(
            model, dataset, args.model_name, "jointavbench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OVO-Bench (unified Omni JSONL, offline MCQ/QA) evaluation
    if getattr(args, 'eval_ovobench', False):
        dataset = load_dataset(args, "ovobench")
        result['ovobench'] = infer_and_evaluate(
            model, dataset, args.model_name, "ovobench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Real (unified Omni JSONL, offline MCQ/QA) evaluation
    if getattr(args, 'eval_streamingbench_real', False):
        dataset = load_dataset(args, "streamingbench_real")
        result['streamingbench_real'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_real", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Omni (unified Omni JSONL, offline MCQ/QA) evaluation
    if getattr(args, 'eval_streamingbench_omni', False):
        dataset = load_dataset(args, "streamingbench_omni")
        result['streamingbench_omni'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_omni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Omni-Fix (filtered Omni JSONL, only 4 task_types) evaluation
    if getattr(args, 'eval_streamingbench_omni_fix', False):
        dataset = load_dataset(args, "streamingbench_omni_fix")
        result['streamingbench_omni_fix'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_omni_fix", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-SQA (unified Omni JSONL, offline MCQ/QA) evaluation
    if getattr(args, 'eval_streamingbench_sqa', False):
        dataset = load_dataset(args, "streamingbench_sqa")
        result['streamingbench_sqa'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_sqa", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-Holmes evaluation
    if getattr(args, 'eval_video_holmes', False):
        dataset = load_dataset(args, "video_holmes")
        result['video_holmes'] = infer_and_evaluate(
            model, dataset, args.model_name, "video_holmes", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVUT-Benchmark Human evaluation
    if getattr(args, 'eval_avut_benchmark_human', False):
        dataset = load_dataset(args, "avut_benchmark_human")
        result['avut_benchmark_human'] = infer_and_evaluate(
            model, dataset, args.model_name, "avut_benchmark_human", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVUT-Benchmark Gemini evaluation
    if getattr(args, 'eval_avut_benchmark_gemini', False):
        dataset = load_dataset(args, "avut_benchmark_gemini")
        result['avut_benchmark_gemini'] = infer_and_evaluate(
            model, dataset, args.model_name, "avut_benchmark_gemini", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # FutureOmni evaluation (future prediction benchmark)
    if getattr(args, 'eval_futureomni', False):
        dataset = load_dataset(args, "futureomni")
        result['futureomni'] = infer_and_evaluate(
            model, dataset, args.model_name, "futureomni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVMeme-Exam Full evaluation (audio-visual meme understanding)
    if getattr(args, 'eval_avmeme_full', False):
        dataset = load_dataset(args, "avmeme_full")
        result['avmeme_full'] = infer_and_evaluate(
            model, dataset, args.model_name, "avmeme_full", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVMeme-Exam Main evaluation (dataset with text_cheat removed)
    if getattr(args, 'eval_avmeme_main', False):
        dataset = load_dataset(args, "avmeme_main")
        result['avmeme_main'] = infer_and_evaluate(
            model, dataset, args.model_name, "avmeme_main", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Omni-DuplexEval Real-time Description evaluation (inference + save raw output, evaluated by external OmniDuplexEval script)
    if getattr(args, 'eval_omniduplexeval_rtd', False):
        dataset = load_dataset(args, "omniduplexeval_rtd")
        result['omniduplexeval_rtd'] = infer_and_evaluate(
            model, dataset, args.model_name, "omniduplexeval_rtd", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            evaluate=False,
            async_evaluate=async_evaluate
        )
    
    # Omni-DuplexEval Proactive Reminder evaluation (inference + save raw output, evaluated by external OmniDuplexEval script)
    if getattr(args, 'eval_omniduplexeval_pr', False):
        dataset = load_dataset(args, "omniduplexeval_pr")
        result['omniduplexeval_pr'] = infer_and_evaluate(
            model, dataset, args.model_name, "omniduplexeval_pr", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            evaluate=False,
            async_evaluate=async_evaluate
        )
    
    return result


def run_all_evaluations(args, model, device, time, async_evaluate: bool = True):
    """Run all evaluation tasks
    
    Args:
        args: Command line arguments
        model: Model instance
        device: Device
        time: Timestamp
        async_evaluate: Whether to use async evaluation mode (default True)
            - True: Returns immediately after inference, evaluation runs in background thread, all evaluations awaited after all inference completes
            - False: Sync mode, evaluates immediately after each dataset inference (may cause multi-GPU timeout)
    
    Returns:
        Dictionary of evaluation results
    """
    result = {}
    
    if async_evaluate:
        print("\nUsing async evaluation mode: inference and evaluation run in parallel to avoid multi-GPU timeout")
    
    # Evaluate audio datasets (ASR and QA)
    print("\n" + "="*60)
    print("[Starting audio dataset evaluation]")
    print("="*60)
    audio_results = evaluate_all_audio_datasets(args, model, time, async_evaluate=async_evaluate)
    result.update(audio_results)
    if audio_results:
        print(f"Completed {len(audio_results)} audio dataset(s) {'inference' if async_evaluate else 'evaluation'}")
    
    # Evaluate video datasets
    print("\n" + "="*60)
    print("[Starting video dataset evaluation]")
    print("="*60)
    video_results = evaluate_video_datasets(args, model, device, time, async_evaluate=async_evaluate)
    result.update(video_results)
    if video_results:
        print(f"Completed {len(video_results)} video dataset(s) {'inference' if async_evaluate else 'evaluation'}")
    
    # Evaluate Duplex datasets
    print("\n" + "="*60)
    print("[Starting Duplex dataset evaluation]")
    print("="*60)
    duplex_results = evaluate_duplex_datasets(args, device, time, async_evaluate=async_evaluate)
    result.update(duplex_results)
    if duplex_results:
        print(f"Completed {len(duplex_results)} Duplex dataset(s) {'inference' if async_evaluate else 'evaluation'}")
    
    # Evaluate Full-Duplex-Bench v1/v1.5 datasets
    print("\n" + "="*60)
    print("[Starting Full-Duplex-Bench dataset evaluation]")
    print("="*60)
    fdb_results = evaluate_fdb_datasets(args, model, time, async_evaluate=async_evaluate)
    result.update(fdb_results)
    if fdb_results:
        print(f"Completed {len(fdb_results)} FDB dataset(s) {'inference' if async_evaluate else 'evaluation'}")
    
    # Evaluate Omni datasets
    print("\n" + "="*60)
    print("[Starting Omni dataset evaluation]")
    print("="*60)
    omni_results = evaluate_omni_datasets(args, model, time, async_evaluate=async_evaluate)
    result.update(omni_results)
    if omni_results:
        print(f"Completed {len(omni_results)} Omni dataset(s) {'inference' if async_evaluate else 'evaluation'}")
    
    # In async mode, wait for all evaluation tasks to complete
    if async_evaluate:
        # Only rank 0 needs to wait for evaluation results
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            pending_count = get_pending_eval_count()
            if pending_count > 0:
                print("\n" + "="*60)
                print(f"[Waiting for async evaluation tasks to complete] Total {pending_count} task(s)")
                print("="*60)
                async_results = wait_all_evaluations()
                result.update(async_results)
    
    print("\n" + "="*60)
    print(f"[All evaluation tasks completed] Evaluated {len(result)} dataset(s) in total")
    print("="*60)
    
    return result

def save_evaluation_results(result, args, time):
    """Save evaluation results, one file per dataset"""
    print(f"\nFinal evaluation results summary: {result}")
    
    # Build result save path
    result_dir = os.path.join(args.answer_path, args.model_name, time)
    os.makedirs(result_dir, exist_ok=True)
    
    # Check for valid results
    output_flag = False
    saved_files = []
    
    for dataset_name, score in result.items():
        if score is not None and score >= 0.0:
            output_flag = True
            
            # Create a separate result file for each dataset
            result_filename = f"result_{dataset_name}.json"
            result_path = os.path.join(result_dir, result_filename)
            job_id = os.getenv('JOB_ID', -1)
            # Save individual dataset result
            dataset_result = {
                "dataset_name": dataset_name,
                "score": score,
                "model_name": args.model_name,
                "evaluation_time": time,
                "job_id": job_id,
            }
            
            with open(result_path, "w", encoding='utf-8') as f:
                json.dump(dataset_result, f, indent=4, ensure_ascii=False)
            
            saved_files.append(result_filename)
            print(f"{dataset_name} result saved to: {result_filename}")
    
    if output_flag:
        print(f"Saved {len(saved_files)} dataset result file(s)")
        print(f"Results saved in directory: {result_dir}")
    else:
        print("No valid evaluation results to save")