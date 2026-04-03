from __future__ import annotations

import os
import shlex
import subprocess
from typing import Optional
from modal import Image, Volume, gpu
import modal
# After training, publish logs to Weights & Biases as an artifact
import time
import wandb
import os
import sys

from huggingface_hub import hf_hub_download
import re
import glob
# Download the GPT-2 tokens of Fineweb10B from huggingface into path/fineweb10B
# Same structure expected by train_gpt.py defaults.
def get(fname: str, base_path: str):
    local_dir = os.path.join(base_path, "fineweb10B")
    os.makedirs(local_dir, exist_ok=True)
    dst = os.path.join(local_dir, fname)
    if not os.path.exists(dst):
        hf_hub_download(
            repo_id="kjj0/fineweb10B-gpt2",
            filename=fname,
            repo_type="dataset",
            local_dir=local_dir,
        )

try:
    # Newer Modal API
    from modal import App  # type: ignore
except ImportError:  # Backward compatibility
    from modal import Stub as App  # type: ignore


# Two images: CPU-only for data prep, CUDA-enabled for training.

# CPU image: smaller, CPU torch. Good for downloads and preprocessing.
image_cpu = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        # CPU torch from PyPI
        "torch",
    )
    .add_local_dir(local_path=".", remote_path="/workspace", ignore=["*.pyc", "__pycache__", "img/", "records/", ".venv/"])
)

# GPU image: install CUDA-enabled torch matching the repo's Dockerfile (cu126 nightly).
image_gpu = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "numpy",
        "tqdm",
        "huggingface-hub",
        "kernels",
        "wandb",
    )
    .pip_install(
        "torch",
    )
    .add_local_dir(local_path=".", remote_path="/workspace", ignore=["*.pyc", "__pycache__", "img/", "records/", ".venv/"])
)


app = App("modded-nanogpt-modal")
data_volume = Volume.from_name("modded-nanogpt-data", create_if_missing=True)

def _run(cmd: str, cwd: Optional[str] = None):
    print(f"[modal] Running: {cmd}")
    subprocess.run(shlex.split(cmd), cwd=cwd, check=True)


@app.function(
    image=image_cpu,
    volumes={"/data": data_volume},
    timeout=60 * 60,  # 1h
)
def prep_data(num_chunks: int = 9):
    """
    Download GPT-2 tokenized FineWeb10B shards into /data/fineweb10B.
    Default is 9 chunks (~900M tokens) to speed up first runs.
    """

    # Ensure data dir exists on the mounted volume
    # os.makedirs("data", exist_ok=True)
    get("fineweb_val_%06d.bin" % 0, base_path="/data")
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i, base_path="/data")


@app.function(
    image=image_gpu,
    volumes={"/data": data_volume},
    gpu="H100:8",
    timeout=60 * 60 * 8,  # 8h cap; adjust as needed
    secrets=[modal.Secret.from_name("matopt"),]
)
def train(
    args: str = "",
):
    """
    Launch distributed training on 8×H100 using torchrun.
    Pass additional CLI args via the `args` string, e.g. "--some_flag value".
    """
    os.chdir("/workspace")
    # Ensure training code resolves data files under / (so /data/... matches defaults)
    os.environ["DATA_PATH"] = "/"

    # Build command and stream stdout to parse val_loss for W&B logging
    base_cmd = "torchrun --standalone --nproc_per_node=8 train_gpt.py"
    if args:
        base_cmd += f" {args}"
    print(f"[modal] Launching: {base_cmd}")

    # Authenticate W&B from env
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    run = None
    run_id = None
    step_re = re.compile(r"step:(\d+)/(\d+)")
    vl_re = re.compile(r"val_loss:([0-9.]+)")
    tt_re = re.compile(r"train_time:(\d+)ms")
    sa_re = re.compile(r"step_avg:([0-9.]+)ms")
    log_path_re = re.compile(r"^logs/(.+)\.txt\s*$")

    project = os.environ.get("WANDB_PROJECT", "nanogpt")
    entity = os.environ.get("WANDB_ENTITY")
    run = wandb.init(project=project, entity=entity, job_type="training-metrics")

    proc = subprocess.Popen(
        shlex.split(base_cmd),
        cwd="/workspace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(line)
            if run_id is None:
                mlog = log_path_re.match(line)
                if mlog:
                    run_id = mlog.group(1)
                    try:
                        # Update W&B run name once known
                        run.name = run_id
                        run.save()
                    except Exception:
                        pass
            if "val_loss:" in line:
                m_step = step_re.search(line)
                m_vl = vl_re.search(line)
                m_tt = tt_re.search(line)
                m_sa = sa_re.search(line)
                metrics = {}
                if m_step:
                    metrics["step"] = int(m_step.group(1))
                    metrics["total_steps"] = int(m_step.group(2))
                if m_vl:
                    metrics["val_loss"] = float(m_vl.group(1))
                if m_tt:
                    metrics["train_time_ms"] = int(m_tt.group(1))
                if m_sa:
                    metrics["step_avg_ms"] = float(m_sa.group(1))
                if metrics:
                    wandb.log(metrics, step=metrics.get("step"))
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, base_cmd)
    finally:
        # Upload logs as an artifact before finishing the run
        try:
            artifact = wandb.Artifact(name=f"logs-{int(time.time())}", type="logs")
            uploaded = False
            if os.path.isdir("logs"):
                artifact.add_dir("logs", name="logs")
                uploaded = True
            if os.path.isdir("log"):
                artifact.add_dir("log", name="log")
                uploaded = True
            if uploaded:
                run.log_artifact(artifact)
                print("[modal] Uploaded logs directory to W&B.")
        except Exception as e:
            print(f"[modal] W&B artifact upload skipped/failed: {e}")
        try:
            run.finish()
        except Exception:
            pass


@app.local_entrypoint()
def main(mode: str = "train", chunks: int = 9, args: str = ""):
    """
    Local entrypoint to make CLI use simple:
      modal run modal_train.py::main --mode prep_data --chunks 9
      modal run modal_train.py::main --mode train --args "--any_extra_flags"
    """
    if mode == "prep_data":
        prep_data.remote(num_chunks=chunks)
    elif mode == "train":
        train.remote(args=args)
    else:
        raise ValueError("mode must be one of: prep_data, train")
