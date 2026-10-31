"""Short, gold-free inference batch benchmark before the seven evaluations.

Training microbatch/accumulation are untouched. Runs use a fresh FP16 original
backbone, the same stock likelihood requests, and non-thinking greedy GSM8K
prompts. Decode tests cap at 64 tokens and reserve memory for the real 1024 cap.
No benchmark generations enter the reported accuracy evaluations.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from task1_train import digest, json_write, load_inputs, seed_all
from task1_eval import (
    harness_fingerprint, inference_setup, make_task, source_directory,
)

CANDIDATES = (8, 16, 32, 64)
REPETITIONS = 2
DECODE_CAP = 64
HEADROOM_FRACTION = 0.85


def select_ids(texts, count, longest_count, rng):
    # Select by input length and seeded sampling, never labels or model accuracy.
    longest = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))[:longest_count]
    remaining = [i for i in range(len(texts)) if i not in set(longest)]
    sampled = rng.choice(remaining, size=count - len(longest), replace=False).tolist()
    return longest + [int(index) for index in sampled]


def choose_batch(records, total_memory):
    eligible = [row for row in records if not row.get("oom", False)
                and row["projected_peak_bytes"] <= total_memory * HEADROOM_FRACTION]
    if not eligible:
        raise RuntimeError("No tested batch fits the memory headroom rule. Inspect the report.")
    return min(eligible, key=lambda row: (row["mean_seconds"], row["batch_size"]))["batch_size"]


def prepare_workload(task_dir):
    import numpy as np
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer
    config = json.loads((task_dir / "configs/evaluation_config.json").read_text())
    assert config["harness_sha256"] == harness_fingerprint()
    source = source_directory(task_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    rng = np.random.default_rng(6304)
    manager = TaskManager()
    likelihood = []
    generation = []
    ids = {}
    for name in ("hellaswag", "boolq", "arc_easy", "gsm8k_cot"):
        task = make_task(manager, task_dir, name, config)
        docs = task.eval_docs
        contexts = [task.doc_to_text(doc) for doc in docs]
        selected = select_ids(contexts, 64 if name == "gsm8k_cot" else 16,
                              8 if name == "gsm8k_cot" else 4, rng)
        ids[name] = selected
        if name == "gsm8k_cot":
            for index in selected:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": contexts[index]}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                generation.append({"task": name, "row_index": index, "context": prompt,
                                   "kwargs": {**task.config.generation_kwargs,
                                              "max_gen_toks": DECODE_CAP, "do_sample": False}})
        else:
            for index in selected:
                requests = task.construct_requests(docs[index], contexts[index], metadata=(name, index, 1))
                if not isinstance(requests, list):
                    requests = [requests]
                for request in requests:
                    assert request.request_type == "loglikelihood"
                    likelihood.append({"task": name, "row_index": index, "args": list(request.args)})
    result = {"selection": "longest character-length contexts plus seeded random contexts; no correctness labels",
              "ids": ids, "likelihood": likelihood, "generation": generation,
              "harness_sha256": config["harness_sha256"]}
    json_write(task_dir / "splits/evaluation_batch_workload.json", result)
    print(f"Batch workload ready: {len(likelihood)} likelihood requests, {len(generation)} generation prompts.", flush=True)
    return result


def instance_lists(workload):
    from lm_eval.api.instance import Instance
    likelihood = [Instance(request_type="loglikelihood", doc={}, arguments=tuple(row["args"]),
                           idx=i, metadata=(row["task"], row["row_index"], 1))
                  for i, row in enumerate(workload["likelihood"])]
    generation = [Instance(request_type="generate_until", doc={},
                           arguments=(row["context"], row["kwargs"]),
                           idx=i, metadata=(row["task"], row["row_index"], 1))
                  for i, row in enumerate(workload["generation"])]
    return likelihood, generation


def benchmark(task_dir):
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert torch.cuda.is_available(), "This timing benchmark requires the GPU."
    # Batch configuration must be fixed before any main evaluation condition.
    started = list((task_dir / "results/evaluation").glob("*/identity.json"))
    assert not started, "A main evaluation already started; do not change its batch settings mid-experiment."
    seed_all(6304)
    original, _ = load_inputs(task_dir)
    config_path = task_dir / "configs/evaluation_config.json"
    settings = json.loads(config_path.read_text())
    report_path = task_dir / "results/evaluation_batch_benchmark.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report["gpu"] == torch.cuda.get_device_name(0)
        assert report["model_revision"] == original["model_revision"]
        assert settings["likelihood_batch_size"] == report["selected"]["likelihood"]
        assert settings["generation_batch_size"] == report["selected"]["generation"]
        print("Reusing completed batch benchmark:", json.dumps(report["selected"]), flush=True)
        return
    workload_path = task_dir / "splits/evaluation_batch_workload.json"
    workload = json.loads(workload_path.read_text()) if workload_path.exists() else prepare_workload(task_dir)
    assert workload["harness_sha256"] == settings["harness_sha256"]
    source = source_directory(task_dir)
    model = AutoModelForCausalLM.from_pretrained(str(source), local_files_only=True,
                                               dtype=torch.float16, attn_implementation="sdpa").to("cuda")
    inference_setup(model)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    wrapper = HFLM(pretrained=model, tokenizer=tokenizer, backend="causal", batch_size=8,
                   add_bos_token=False, enable_thinking=False, softmax_dtype=torch.float32)
    likelihood, generation = instance_lists(workload)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    records = {"likelihood": [], "generation": []}
    for phase, requests in [("likelihood", likelihood), ("generation", generation)]:
        operation = wrapper.loglikelihood if phase == "likelihood" else wrapper.generate_until
        for size in CANDIDATES:
            wrapper.batch_size_per_gpu = size
            gc.collect()
            torch.cuda.empty_cache()
            try:
                # Small kernel warmup, excluded from timings.
                operation(requests[:8], disable_tqdm=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                measurements = []
                for _ in range(REPETITIONS):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    operation(requests, disable_tqdm=True)
                    torch.cuda.synchronize()
                    measurements.append(time.perf_counter() - start)
                peak = torch.cuda.max_memory_allocated()
                projected_peak = peak
                if phase == "generation":
                    cfg = model.config
                    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
                    # Add FP16 key AND value storage for the additional 960
                    # output positions used by the real 1024-token cap.
                    projected_peak += (2 * cfg.num_hidden_layers * cfg.num_key_value_heads
                                       * head_dim * 2 * size * (1024 - DECODE_CAP))
                record = {"batch_size": size, "oom": False, "seconds": measurements,
                          "mean_seconds": sum(measurements) / len(measurements),
                          "peak_bytes": peak, "projected_peak_bytes": projected_peak,
                          "requests_per_second": len(requests) / (sum(measurements) / len(measurements))}
            except torch.OutOfMemoryError:
                record = {"batch_size": size, "oom": True}
                gc.collect()
                torch.cuda.empty_cache()
            records[phase].append(record)
            print(phase, json.dumps(record), flush=True)
            json_write(task_dir / "results/evaluation_batch_benchmark_progress.json", records)
    selected = {phase: choose_batch(rows, total_memory) for phase, rows in records.items()}
    new_settings = {**settings, "likelihood_batch_size": selected["likelihood"],
                    "generation_batch_size": selected["generation"]}
    # Preserve earlier tiny smoke output if it exists. Main evaluations cannot
    # have started because of the guard above. No training checkpoints are moved.
    previous_smoke = task_dir / "smoke_evaluation"
    if previous_smoke.exists() and digest(new_settings) != digest(settings):
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        previous_smoke.rename(task_dir / ("smoke_evaluation_before_batch_tuning_" + suffix))
    report = {"gpu": torch.cuda.get_device_name(0), "model_revision": original["model_revision"],
              "selected": selected, "records": records, "workload_sha256": digest(workload),
              "decode_benchmark_cap": DECODE_CAP, "real_decode_cap": 1024,
              "memory_fraction_limit": HEADROOM_FRACTION,
              "selection_rule": "lowest mean wall time over two repetitions, excluding OOM and memory-headroom failures",
              "note": "Fastest among tested batch sizes on this fixed workload; not a universal optimum."}
    json_write(config_path, new_settings)
    json_write(report_path, report)
    print("\nSelected evaluation batches:", json.dumps(selected), flush=True)
    print("Training remains microbatch 2, accumulation 16. Ready for Cell 14.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true", help="CPU-only workload construction")
    args = parser.parse_args()
    if args.prepare:
        prepare_workload(args.task_dir)
    else:
        benchmark(args.task_dir)


if __name__ == "__main__":
    main()
