"""
modal app for serverless GPU evaluation of nanochat models.

usage:
    # eval HF model (CORE only, chat evals not supported for HF models)
    modal run modal_eval.py --models "hf:openai-community/gpt2" --evals "core"

    # eval checkpoint from HF repo (tag defaults to repo basename)
    modal run modal_eval.py --models "base:nanochat-mhc-d20-static" --evals "core,chat" \
        --hf-ckpt-repo "tomzhengy/nanochat-mhc-d20-static"

    # eval with explicit tag override
    modal run modal_eval.py --models "base:d20-mhc" --evals "core" \
        --hf-ckpt-repo "tomzhengy/nanochat-mhc-d20-static" --ckpt-model-tag "d20-mhc"

    # override GPU
    modal run modal_eval.py --models "base:d20" --evals "core" --gpu a100

setup:
    modal secret create huggingface HF_TOKEN=hf_xxx
"""

import modal

# build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "curl", "build-essential")
    # install rust for rustbpe tokenizer
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> /root/.bashrc",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    # install uv
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"PATH": "/root/.local/bin:/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    # copy the repo
    .copy_local_dir(".", "/app")
    # install python deps and build tokenizer
    .run_commands(
        "cd /app && uv venv && uv sync --extra gpu",
        "cd /app && uv run maturin develop --release --manifest-path rustbpe/Cargo.toml",
    )
    # pre-download tokenizer into the image
    .run_commands(
        "cd /app && uv run python -c \""
        "from huggingface_hub import snapshot_download; "
        "import os; "
        "os.makedirs(os.path.expanduser('~/.cache/nanochat/tokenizer'), exist_ok=True); "
        "snapshot_download('tomzhengy/nanochat-tokenizer', local_dir=os.path.expanduser('~/.cache/nanochat/tokenizer'))"
        "\""
    )
    # pre-download CORE eval bundle
    .run_commands(
        "cd /app && uv run python -c \""
        "from nanochat.common import get_base_dir, download_file_with_lock; "
        "import os, shutil, zipfile, tempfile; "
        "EVAL_BUNDLE_URL = 'https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip'; "
        "def place_eval_bundle(fp): "
        "    bd = get_base_dir(); ed = os.path.join(bd, 'eval_bundle'); "
        "    tmpdir = tempfile.mkdtemp(); "
        "    zipfile.ZipFile(fp, 'r').extractall(tmpdir); "
        "    shutil.move(os.path.join(tmpdir, 'eval_bundle'), ed); "
        "download_file_with_lock(EVAL_BUNDLE_URL, 'eval_bundle.zip', postprocess_fn=place_eval_bundle)"
        "\""
    )
)

app = modal.App("nanochat-eval", image=image)


@app.function(
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_eval(models: list[str], evals: list[str], hf_ckpt_repo: str = None,
             ckpt_model_tag: str = None, ckpt_source: str = "base",
             max_per_task: int = -1, max_problems: int = None):
    """run evaluation on GPU and return results dict + HTML string."""
    import os
    import json
    import subprocess

    os.chdir("/app")

    # download checkpoint from HF if specified
    if hf_ckpt_repo:
        source_dirs = {
            "base": "base_checkpoints",
            "mid": "mid_checkpoints",
            "sft": "chatsft_checkpoints",
            "rl": "chatrl_checkpoints",
        }
        subdir = source_dirs.get(ckpt_source, "base_checkpoints")
        tag = ckpt_model_tag or os.path.basename(hf_ckpt_repo)
        base_dir = os.path.expanduser("~/.cache/nanochat")
        ckpt_dir = os.path.join(base_dir, subdir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        from huggingface_hub import snapshot_download
        snapshot_download(hf_ckpt_repo, local_dir=ckpt_dir)
        print(f"checkpoint downloaded as {ckpt_source}:{tag}")

        # rename to checkpoint manager format if needed
        model_pt = os.path.join(ckpt_dir, "model.pt")
        model_001 = os.path.join(ckpt_dir, "model_000001.pt")
        if os.path.exists(model_pt) and not os.path.exists(model_001):
            os.rename(model_pt, model_001)
            meta_json = os.path.join(ckpt_dir, "meta.json")
            meta_001 = os.path.join(ckpt_dir, "meta_000001.json")
            if os.path.exists(meta_json):
                os.rename(meta_json, meta_001)

    # build the command
    cmd = [
        "uv", "run", "python", "-m", "scripts.eval_compare",
        "--models", *models,
        "--evals", *evals,
        "--dashboard",
        "--no-open-dashboard",
        "--output", "/tmp/eval_results.json",
    ]
    if max_per_task != -1:
        cmd.extend(["--max-per-task", str(max_per_task)])
    if max_problems is not None:
        cmd.extend(["--max-problems", str(max_problems)])

    env = os.environ.copy()
    env["TORCH_COMPILE_DISABLE"] = "1"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"eval_compare failed with code {result.returncode}")

    # read results
    with open("/tmp/eval_results.json") as f:
        results = json.load(f)

    # read dashboard HTML
    html = ""
    html_path = "/tmp/eval_results.html"
    if os.path.exists(html_path):
        with open(html_path) as f:
            html = f.read()

    return {"results": results, "html": html}


@app.local_entrypoint()
def main(
    models: str = "hf:openai-community/gpt2",
    evals: str = "core",
    hf_ckpt_repo: str = "",
    ckpt_model_tag: str = "",
    ckpt_source: str = "base",
    max_per_task: int = -1,
    max_problems: int = -1,
    gpu: str = "A10G",
):
    """run nanochat evaluation on Modal GPU.

    args:
        models: comma-separated model specs (e.g. "base:d20,base:d20-mhc")
        evals: comma-separated eval types (e.g. "core,chat")
        hf_ckpt_repo: HuggingFace repo for checkpoint download
        ckpt_model_tag: model tag for downloaded checkpoint
        ckpt_source: checkpoint source type (base|mid|sft|rl)
        max_per_task: max examples per CORE task (-1 = all)
        max_problems: max problems per ChatCORE task (-1 = all)
        gpu: GPU type (A10G, a100, etc.)
    """
    import json
    import webbrowser
    import os

    model_list = [m.strip() for m in models.split(",")]
    eval_list = [e.strip() for e in evals.split(",")]
    mp = None if max_problems == -1 else max_problems

    print(f"running eval on Modal ({gpu}):")
    print(f"  models: {model_list}")
    print(f"  evals: {eval_list}")
    if hf_ckpt_repo:
        resolved_tag = ckpt_model_tag or os.path.basename(hf_ckpt_repo)
        print(f"  checkpoint: {hf_ckpt_repo} -> {ckpt_source}:{resolved_tag}")
        # warn if no model spec references the downloaded checkpoint tag
        same_source_specs = [s for s in model_list if s.startswith(f"{ckpt_source}:")]
        any_match = any(
            s.split(":", maxsplit=2)[1] == resolved_tag
            for s in same_source_specs if ":" in s
        )
        if same_source_specs and not any_match:
            print(f"  warning: downloaded checkpoint as '{ckpt_source}:{resolved_tag}' "
                  f"but no model spec references that tag. "
                  f"did you mean to include '{ckpt_source}:{resolved_tag}'?")

    # override GPU at runtime via with_options
    fn = run_eval.with_options(gpu=gpu) if gpu != "A10G" else run_eval
    out = fn.remote(
        models=model_list,
        evals=eval_list,
        hf_ckpt_repo=hf_ckpt_repo or None,
        ckpt_model_tag=ckpt_model_tag or None,
        ckpt_source=ckpt_source,
        max_per_task=max_per_task,
        max_problems=mp,
    )

    # write results locally
    results_path = "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(out["results"], f, indent=2)
    print(f"\nresults written to {results_path}")

    # write dashboard locally
    if out.get("html"):
        html_path = "eval_results.html"
        with open(html_path, "w") as f:
            f.write(out["html"])
        print(f"dashboard written to {html_path}")
        try:
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
        except Exception:
            pass
