"""One queue: shared baseline, five independent training runs, required evaluations."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys

from task1_train import json_write


def commands(task_dir):
    # Evaluate saved Full FT midpoint after training ends: its weights are fixed,
    # so evaluation is identical without pausing/reloading the training process.
    return [
        ("evaluate", "baseline"),
        ("train", "full"), ("evaluate", "full_mid"), ("evaluate", "full_final"),
        ("train", "l2sp"), ("evaluate", "l2sp_final"),
        ("train", "lora4"), ("evaluate", "lora4_final"),
        ("train", "lora16"), ("evaluate", "lora16_final"),
        ("train", "oplora16"), ("evaluate", "oplora16_final"),
    ]


def run(command, log_path):
    print("\nLaunching:", " ".join(str(part) for part in command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="CPU-safe queue inspection")
    mode.add_argument("--smoke", action="store_true", help="GPU: two training updates + tiny baseline evaluation")
    mode.add_argument("--run-all", action="store_true", help="GPU: full queue; completed work skips")
    args = parser.parse_args()
    task = args.task_dir
    for name in ["runner_cpu_preflight.json", "evaluation_cpu_preflight.json"]:
        report = json.loads((task / "results" / name).read_text())
        assert report["passed"], name
    assert metadata.version("lm_eval") == "0.4.13"
    code_dir = Path(__file__).parent
    if args.plan:
        print("CPU-safe plan; no model is loaded:")
        for kind, name in commands(task):
            if kind == "train":
                complete = (task / "checkpoints" / name / "finished.json").exists()
            else:
                complete = (task / "results/evaluation" / name / "finished.json").exists()
            print(f"  {kind:8} {name:16} {'COMPLETE (will skip)' if complete else 'pending'}")
        print("GPU smoke tests must pass before --run-all.")
        return
    import torch
    assert torch.cuda.is_available(), "Reconnect a CUDA GPU only when ready to launch."
    original_env = json.loads((task / "configs/gpu_environment.json").read_text())
    for package in ["transformers", "datasets", "accelerate", "peft", "lm_eval"]:
        assert metadata.version(package) == original_env["packages"][package], (
            f"{package} changed; check before launching."
        )
    log_dir = task / "results/queue_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        for method in ["full", "lora16"]:
            run([sys.executable, str(code_dir / "task1_train.py"), "--task-dir", str(task),
                 "--method", method, "--smoke-updates", "2"], log_dir / f"smoke_{method}.log")
        run([sys.executable, str(code_dir / "task1_eval.py"), "--task-dir", str(task),
             "--condition", "baseline", "--smoke"], log_dir / "smoke_evaluation.log")
        json_write(task / "results/gpu_smoke_passed.json", {"passed": True})
        print("GPU smoke tests passed. Review smoke_summary.json update times before full launch.", flush=True)
        return
    assert json.loads((task / "results/gpu_smoke_passed.json").read_text())["passed"]
    for kind, name in commands(task):
        script, flag = ("task1_train.py", "--method") if kind == "train" else ("task1_eval.py", "--condition")
        run([sys.executable, str(code_dir / script), "--task-dir", str(task), flag, name],
            log_dir / (name + ".log"))
    json_write(task / "results/queue_finished.json", {"completed_training_runs": 5,
                                                      "completed_evaluation_conditions": 7})
    print("All training and required evaluations finished. Analysis/report work remains.", flush=True)


if __name__ == "__main__":
    main()
