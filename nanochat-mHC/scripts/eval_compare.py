"""
compare evaluations across multiple nanochat models.

runs CORE and/or ChatCORE evals, collects timing/throughput metrics,
and outputs a JSON results file for the dashboard.

usage:
    # compare two local models on all evals
    python -m scripts.eval_compare --models base:d20 base:d20-mhc --evals core chat

    # single HF model, CORE only
    python -m scripts.eval_compare --models hf:openai-community/gpt2 --evals core

    # multi-GPU
    torchrun --nproc_per_node=8 -m scripts.eval_compare -- --models base:d20 base:d20-mhc --evals core

    # smoke test
    python -m scripts.eval_compare --models base:d20 --evals core --max-per-task 5

    # auto-generate dashboard
    python -m scripts.eval_compare --models base:d20 base:d20-mhc --evals core chat --dashboard
"""

import os
import gc
import json
import time
import argparse
from datetime import datetime
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from scripts.base_eval import evaluate_model, load_hf_model
from scripts.chat_eval import run_chat_eval


def parse_model_spec(spec):
    """parse a model spec string into (source, tag, step).

    formats:
        hf:openai-community/gpt2     -> ("hf", "openai-community/gpt2", None)
        base:d20                       -> ("base", "d20", None)
        base:d20:5000                  -> ("base", "d20", 5000)
        mid:d20-mhc                    -> ("mid", "d20-mhc", None)
    """
    parts = spec.split(":", maxsplit=2)
    source = parts[0]
    tag = parts[1] if len(parts) > 1 else None
    step = int(parts[2]) if len(parts) > 2 else None
    return source, tag, step


def get_gpu_info():
    """return gpu name and vram if cuda is available."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": props.name,
            "gpu_vram_gb": round(props.total_mem / (1024**3), 1),
        }
    return {"gpu_name": "cpu", "gpu_vram_gb": 0}


def count_params(model):
    """count total parameters. works for both nanochat GPT and HF ModelWrapper."""
    if hasattr(model, "parameters"):
        return sum(p.numel() for p in model.parameters())
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        return sum(p.numel() for p in model.model.parameters())
    return 0


def get_model_config_dict(model):
    """extract model config as a dict for metadata."""
    # nanochat GPT model
    if hasattr(model, "config"):
        cfg = model.config
        return {
            "n_layer": cfg.n_layer,
            "n_embd": cfg.n_embd,
            "n_head": cfg.n_head,
            "mhc_enabled": cfg.mhc_enabled,
            "mhc_static": cfg.mhc_static if cfg.mhc_enabled else None,
            "mhc_num_streams": cfg.mhc_num_streams if cfg.mhc_enabled else None,
        }
    # HF ModelWrapper
    if hasattr(model, "model") and hasattr(model.model, "config"):
        hf_cfg = model.model.config
        return {
            "n_layer": getattr(hf_cfg, "n_layer", getattr(hf_cfg, "num_hidden_layers", None)),
            "n_embd": getattr(hf_cfg, "n_embd", getattr(hf_cfg, "hidden_size", None)),
            "n_head": getattr(hf_cfg, "n_head", getattr(hf_cfg, "num_attention_heads", None)),
            "mhc_enabled": False,
        }
    return {}


def load_model_from_spec(spec, device):
    """load model and tokenizer from a model spec string.

    returns (model, tokenizer, model_name, is_hf)
    """
    source, tag, step = parse_model_spec(spec)

    if source == "hf":
        hf_path = tag
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path
        return model, tokenizer, model_name, True

    # local checkpoint
    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=tag, step=step)
    step_loaded = meta.get("step", "?")
    model_name = f"{source}:{tag} (step {step_loaded})"
    return model, tokenizer, model_name, False


CHAT_TASKS = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
CHAT_BASELINES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
    "GSM8K": 0.0,
    "HumanEval": 0.0,
    "SpellingBee": 0.0,
}


def run_core_eval(model, tokenizer, device, max_per_task):
    """run CORE eval with timing. returns (results_dict, wall_time_seconds)."""
    t0 = time.time()
    out = evaluate_model(model, tokenizer, device, max_per_task=max_per_task)
    wall_time = time.time() - t0
    return out, wall_time


def run_all_chat_evals(model, tokenizer, device, is_hf, max_problems, batch_size=8):
    """run all ChatCORE tasks with timing. returns (results_dict, wall_time_seconds).

    generative tasks (GSM8K, HumanEval, SpellingBee) require Engine and only work
    with nanochat models, not HF wrappers. they are skipped for HF models.
    """
    from nanochat.engine import Engine

    generative_tasks = {"GSM8K", "HumanEval", "SpellingBee"}
    engine = None
    if not is_hf:
        engine = Engine(model, tokenizer)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    results = {}
    t0 = time.time()
    for task_name in CHAT_TASKS:
        if task_name in generative_tasks and is_hf:
            print0(f"  skipping {task_name} (generative task, requires nanochat Engine)")
            continue

        print0(f"  evaluating {task_name}...")
        with autocast_ctx:
            acc = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=batch_size,
                num_samples=1,
                max_new_tokens=512,
                temperature=0.0,
                top_k=50,
                max_problems=max_problems,
            )
        results[task_name] = acc
        print0(f"  {task_name}: {100 * acc:.2f}%")

    wall_time = time.time() - t0

    # compute chatcore metric if all tasks evaluated
    centered_results = {}
    chatcore_metric = None
    for task_name, acc in results.items():
        baseline = CHAT_BASELINES.get(task_name, 0.0)
        centered_results[task_name] = (acc - baseline) / (1.0 - baseline)
    if len(results) == len(CHAT_TASKS):
        chatcore_metric = sum(centered_results.values()) / len(centered_results)

    out = {
        "results": results,
        "centered_results": centered_results,
        "chatcore_metric": chatcore_metric,
    }
    return out, wall_time


def print_summary(all_results):
    """print a summary table to the terminal."""
    print0()
    print0("=" * 80)
    print0("EVALUATION SUMMARY")
    print0("=" * 80)

    for entry in all_results:
        name = entry["model_name"]
        params = entry["param_count"]
        print0(f"\n  {name} ({params:,} params)")

        if "core" in entry:
            core = entry["core"]
            print0(f"    CORE metric: {core['core_metric']:.4f} ({core['wall_time']:.1f}s)")

        if "chat" in entry:
            chat = entry["chat"]
            metric = chat.get("chatcore_metric")
            metric_str = f"{metric:.4f}" if metric is not None else "n/a (incomplete)"
            print0(f"    ChatCORE metric: {metric_str} ({chat['wall_time']:.1f}s)")

    print0()
    print0("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="compare evals across nanochat models")
    parser.add_argument("--models", nargs="+", required=True,
                        help="model specs: base:d20, hf:openai-community/gpt2, etc.")
    parser.add_argument("--evals", nargs="+", default=["core"],
                        help="eval types to run: core, chat")
    parser.add_argument("--max-per-task", type=int, default=-1,
                        help="max examples per CORE task (-1 = all)")
    parser.add_argument("--max-problems", type=int, default=None,
                        help="max problems per ChatCORE task (None = all)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="batch size for categorical chat evals")
    parser.add_argument("--dashboard", action="store_true",
                        help="generate HTML dashboard after eval")
    parser.add_argument("--output", type=str, default=None,
                        help="custom output path for results JSON")
    args = parser.parse_args()

    # normalize eval names
    evals = [e.lower().strip() for e in args.evals]

    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    gpu_info = get_gpu_info()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    for spec in args.models:
        print0(f"\n{'=' * 60}")
        print0(f"loading model: {spec}")
        print0(f"{'=' * 60}")

        model, tokenizer, model_name, is_hf = load_model_from_spec(spec, device)
        param_count = count_params(model)
        config_dict = get_model_config_dict(model)

        entry = {
            "model_spec": spec,
            "model_name": model_name,
            "param_count": param_count,
            "config": config_dict,
        }

        if "core" in evals:
            print0(f"\nrunning CORE eval for {model_name}...")
            with autocast_ctx:
                core_out, core_time = run_core_eval(model, tokenizer, device, args.max_per_task)
            entry["core"] = {
                "results": core_out["results"],
                "centered_results": core_out["centered_results"],
                "core_metric": core_out["core_metric"],
                "wall_time": round(core_time, 2),
            }

        if "chat" in evals:
            print0(f"\nrunning ChatCORE eval for {model_name}...")
            chat_out, chat_time = run_all_chat_evals(
                model, tokenizer, device, is_hf,
                max_problems=args.max_problems,
                batch_size=args.batch_size,
            )
            entry["chat"] = {
                "results": chat_out["results"],
                "centered_results": chat_out["centered_results"],
                "chatcore_metric": chat_out["chatcore_metric"],
                "wall_time": round(chat_time, 2),
            }

        all_results.append(entry)

        # free GPU memory before loading next model
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # build final output
    output = {
        "timestamp": timestamp,
        "gpu_info": gpu_info,
        "eval_config": {
            "evals": evals,
            "max_per_task": args.max_per_task,
            "max_problems": args.max_problems,
        },
        "models": all_results,
    }

    # write results JSON
    if ddp_rank == 0:
        if args.output:
            output_path = args.output
        else:
            base_dir = get_base_dir()
            results_dir = os.path.join(base_dir, "eval_results")
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, f"{timestamp}.json")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print0(f"\nresults written to {output_path}")

        print_summary(all_results)

        # generate dashboard if requested
        if args.dashboard:
            print0("\ngenerating dashboard...")
            from scripts.eval_dashboard import generate_dashboard
            html_path = output_path.replace(".json", ".html")
            generate_dashboard(output_path, html_path)
            print0(f"dashboard written to {html_path}")

    compute_cleanup()


if __name__ == "__main__":
    main()
