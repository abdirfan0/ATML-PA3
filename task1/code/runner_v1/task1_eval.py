"""PA3 Task 1 fixed lm_eval=0.4.13 evaluation, qualitative output, diagnostics.

Uses the stock harness task prompts, stopping rules, metrics, and answer filters.
Only prescribed few-shot count, token cap, chat settings, dataset revisions,
and persistent dataset-cache locations are overridden.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
from torch import nn

from task1_train import (
    AdapterLinear, VERSION as TRAIN_VERSION, digest, json_write, load_inputs,
    seed_all, torch_write,
)

VERSION = "pa3-task1-eval-v1"
HARNESS_VERSION = "0.4.13"
BENCHMARKS = {"gsm8k_cot": 1319, "hellaswag": 10042, "boolq": 3270, "arc_easy": 2376}
CONDITIONS = {
    "baseline": None, "full_mid": ("full", 625), "full_final": ("full", 1250),
    "l2sp_final": ("l2sp", 1250), "lora4_final": ("lora4", 1250),
    "lora16_final": ("lora16", 1250), "oplora16_final": ("oplora16", 1250),
}
LAYERS = (0, 13, 27)


def serializable(value):
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def append_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("".join(json.dumps(serializable(row), ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


class RequestStore:
    """Append-only JSONL cache with bounded buffered writes; avoids SQLite on Drive."""

    def __init__(self, path):
        self.path = Path(path)
        self.values = {}
        self.pending = []
        damaged = False
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try:
                    row = json.loads(line)
                    self.values[row["key"]] = row["value"]
                except (json.JSONDecodeError, KeyError):
                    damaged = True
            if damaged:
                # Repair an interrupted final line before appending more JSON.
                temporary = self.path.with_suffix(".repair")
                temporary.write_text("".join(json.dumps({"key": key, "value": value}) + "\n"
                                             for key, value in self.values.items()))
                temporary.replace(self.path)

    def put(self, key, value, generation=False):
        self.values[key] = serializable(value)
        self.pending.append({"key": key, "value": serializable(value)})
        if len(self.pending) >= (8 if generation else 128):
            self.flush()

    def flush(self):
        append_rows(self.path, self.pending)
        self.pending.clear()


def token_key(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def generation_metadata(tokenizer, context_ids, generated_ids, budget, stop, eos_ids):
    ids = list(generated_ids)
    first_eos = next((i for i, token in enumerate(ids) if token in eos_ids), None)
    if first_eos is not None:
        ids = ids[:first_eos + 1]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    stopped = first_eos is not None or any(text_stop and text_stop in text for text_stop in stop)
    return {
        "prompt_token_sha256": token_key(context_ids),
        "input_tokens": len(context_ids), "output_tokens": len(ids),
        "output_token_ids": ids, "raw_text": text,
        "output_cap": budget, "reached_output_cap": len(ids) >= budget,
        "truncated": len(ids) >= budget and not stopped,
        "ended_by_eos": first_eos is not None,
    }


def recorded_hflm_class():
    from lm_eval.models.huggingface import HFLM
    from lm_eval.api.model import hash_args

    class PartialHook:
        def __init__(self, store):
            self.store = store

        def add_partial(self, attr, req, res):
            self.store.put(hash_args(attr, req), res, generation=attr == "generate_until")

    class RecordedHFLM(HFLM):
        def __init__(self, *, cache_dir, **kwargs):
            super().__init__(**kwargs)
            self.store = RequestStore(Path(cache_dir) / "requests.jsonl")
            self.raw_path = Path(cache_dir) / "raw_generations.jsonl"
            self.raw_generations = {}
            if self.raw_path.exists():
                valid = []
                for line in self.raw_path.read_text().splitlines():
                    try:
                        row = json.loads(line)
                        self.raw_generations[row["prompt_token_sha256"]] = row
                        valid.append(row)
                    except (json.JSONDecodeError, KeyError):
                        pass
                # Normalize any interrupted tail before recording more rows.
                self.raw_path.write_text("".join(json.dumps(row) + "\n" for row in valid))
            self.set_cache_hook(PartialHook(self.store))

        def _cached(self, attr, requests, delegate):
            keys = [hash_args(attr, request.args) for request in requests]
            missing = [(index, request) for index, (key, request) in enumerate(zip(keys, requests))
                       if key not in self.store.values]
            if missing:
                answers = delegate([request for _, request in missing])
                assert len(answers) == len(missing)
                for (index, _), answer in zip(missing, answers):
                    self.store.put(keys[index], answer, generation=attr == "generate_until")
            self.store.flush()
            return [self.store.values[key] for key in keys]

        def generate_until(self, requests, disable_tqdm=False):
            return self._cached("generate_until", requests,
                                lambda remaining: super(RecordedHFLM, self).generate_until(remaining, disable_tqdm))

        def loglikelihood(self, requests, disable_tqdm=False):
            return self._cached("loglikelihood", requests,
                                lambda remaining: super(RecordedHFLM, self).loglikelihood(remaining, disable_tqdm))

        def _model_generate(self, context, max_length, stop, **kwargs):
            output = super()._model_generate(context, max_length, stop, **kwargs)
            mask = kwargs["attention_mask"]
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
            rows = []
            for index in range(len(context)):
                prompt_ids = context[index][mask[index].bool()].tolist()
                new_ids = output[index, context.shape[1]:].tolist()
                row = generation_metadata(self.tokenizer, prompt_ids, new_ids,
                                          max_length - context.shape[1], stop, eos_ids)
                self.raw_generations[row["prompt_token_sha256"]] = row
                rows.append(row)
            append_rows(self.raw_path, rows)
            return output

    return RecordedHFLM


def harness_fingerprint():
    import lm_eval
    root = Path(lm_eval.__file__).parent
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".yaml", ".yml"):
            hasher.update(str(path.relative_to(root)).encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def task_config(manager, name):
    from lm_eval.tasks._yaml_loader import load_yaml
    return load_yaml(manager.task_index[name].yaml_path)


def make_task(manager, task_dir, name, evaluation_config, local_only=True):
    from lm_eval.api.task import ConfigurableTask
    from datasets import load_from_disk
    cfg = task_config(manager, name)
    cfg["dataset_kwargs"] = {
        **(cfg.get("dataset_kwargs") or {}),
        "revision": evaluation_config["datasets"][name]["revision"],
        "cache_dir": str(task_dir / "data/hf_eval_cache"),
    }
    cfg["num_fewshot"] = 0
    snapshot = task_dir / "data/evaluation_datasets" / name
    snapshot_identity = snapshot / "source_identity.json"
    if snapshot_identity.exists():
        assert json.loads(snapshot_identity.read_text()) == evaluation_config["datasets"][name]
        cfg["custom_dataset"] = lambda **kwargs: load_from_disk(str(snapshot))
    elif local_only:
        raise RuntimeError(f"Run evaluation --prepare on CPU first; missing snapshot for {name}.")
    if name == "gsm8k_cot":
        cfg["generation_kwargs"] = {
            **cfg["generation_kwargs"], "max_gen_toks": 1024, "do_sample": False,
        }
    return ConfigurableTask(config=cfg)


def prepare_evaluation(task_dir):
    """Download evaluation data and construct exact probe prompts, on CPU."""
    from huggingface_hub import HfApi
    from lm_eval.tasks import TaskManager
    config, _ = load_inputs(task_dir)
    assert metadata.version("lm_eval") == HARNESS_VERSION
    manager = TaskManager()
    destination = task_dir / "configs/evaluation_config.json"
    if destination.exists():
        evaluation_config = json.loads(destination.read_text())
        assert evaluation_config["harness_sha256"] == harness_fingerprint()
        assert evaluation_config["model_revision"] == config["model_revision"]
    else:
        probe_config = json.loads((task_dir / "configs/probe_config.json").read_text())
        revisions = {"gsm8k_cot": config["gsm8k_revision"],
                     "hellaswag": probe_config["hellaswag_revision"]}
        datasets = {}
        for name in BENCHMARKS:
            cfg = task_config(manager, name)
            revision = revisions.get(name)
            if revision is None:
                revision = HfApi().dataset_info(cfg["dataset_path"]).sha
            datasets[name] = {"path": cfg["dataset_path"], "name": cfg.get("dataset_name"),
                              "revision": revision}
        evaluation_config = {
            "version": VERSION, "lm_eval": HARNESS_VERSION, "harness_sha256": harness_fingerprint(),
            "model_revision": config["model_revision"], "datasets": datasets,
            "num_fewshot": 0, "gsm8k_max_new_tokens": 1024,
            "gsm8k_chat": True, "enable_thinking": False, "likelihood_chat": False,
            "weights_dtype": "float16", "likelihood_softmax_dtype": "float32",
            "generation_batch_size": 8, "likelihood_batch_size": 8,
            "qualitative_max_new_tokens": 512, "seed": 6304,
        }
        json_write(destination, evaluation_config)
    selected_probes = json.loads((task_dir / "splits/activation_probes.json").read_text())
    probe_prompts = []
    counts = {}
    for name, expected_count in BENCHMARKS.items():
        task = make_task(manager, task_dir, name, evaluation_config, local_only=False)
        snapshot = task_dir / "data/evaluation_datasets" / name
        if not (snapshot / "source_identity.json").exists():
            if snapshot.exists():
                shutil.rmtree(snapshot)  # Derived, incomplete evaluation-data snapshot only.
            task.dataset.save_to_disk(str(snapshot))
            json_write(snapshot / "source_identity.json", evaluation_config["datasets"][name])
        docs = task.eval_docs
        assert len(docs) == expected_count, (name, len(docs), expected_count)
        counts[name] = len(docs)
        print(f"Prepared {name}: {len(docs)} evaluation examples", flush=True)
        if name in ("gsm8k_cot", "hellaswag"):
            source_name = "gsm8k" if name == "gsm8k_cot" else "hellaswag"
            for probe in selected_probes[source_name]:
                doc = docs[probe["row_index"]]
                if name == "gsm8k_cot":
                    assert doc["question"] == probe["question"]
                else:
                    assert doc["ctx"] == probe["record"]["ctx"]
                prompt = task.doc_to_text(doc)
                assert isinstance(prompt, str)
                probe_prompts.append({"task": name, "row_index": probe["row_index"],
                                      "prompt": prompt, "chat": name == "gsm8k_cot"})
    assert len(probe_prompts) == 64
    json_write(task_dir / "splits/probe_prompts.json", probe_prompts)
    result = {"passed": True, "counts": counts, "probe_prompts": 64,
              "lm_eval": HARNESS_VERSION, "harness_sha256": evaluation_config["harness_sha256"]}
    json_write(task_dir / "results/evaluation_cpu_preflight.json", result)
    print(json.dumps(result, indent=2), flush=True)


def source_directory(task_dir):
    config, _ = load_inputs(task_dir)
    return task_dir / "assets/base_model" / config["model_revision"]


def inference_setup(model):
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    model.generation_config.max_new_tokens = None
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1
    model.generation_config.return_dict_in_generate = False


def merge_factors(model, factors):
    with torch.no_grad():
        for name, entry in factors.items():
            base = model.get_submodule(name)
            if isinstance(base, AdapterLinear):
                base = base.base
            delta = entry["scale"] * (entry["B"].to(base.weight.device).float()
                                      @ entry["A"].to(base.weight.device).float())
            base.weight.add_(delta.to(base.weight.dtype))
            parent, leaf = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), leaf, base)


def load_condition(task_dir, condition, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    source = source_directory(task_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
    ).to(device)
    inference_setup(model)
    details = {"condition": condition, "merge_check": None}
    state = None
    selected = CONDITIONS[condition]
    if selected is not None:
        method, update = selected
        directory = task_dir / "checkpoints" / method
        checkpoint = directory / f"update_{update:04d}"
        assert (checkpoint / "complete.json").exists(), f"Missing complete checkpoint: {checkpoint}"
        if method in ("full", "l2sp"):
            state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False, mmap=True)
            assert state["update"] == update and state["version"] == TRAIN_VERSION
            original, manifest = load_inputs(task_dir)
            assert state["identity"]["model_revision"] == original["model_revision"]
            assert state["identity"]["config_sha256"] == digest(original)
            assert state["identity"]["selected_ids_sha256"] == manifest["selected_ids_sha256"]
            model.load_state_dict(state["model"], strict=True)
            # Keep only three local weights needed for FP32 diagnostics, not Adam state.
            state = {f"model.layers.{layer}.mlp.down_proj.weight":
                     state["model"][f"model.layers.{layer}.mlp.down_proj.weight"] for layer in LAYERS}
        else:
            saved = torch.load(directory / "adapter_inference.pt", map_location="cpu", weights_only=True)
            original, manifest = load_inputs(task_dir)
            assert saved["identity"]["model_revision"] == original["model_revision"]
            assert saved["identity"]["config_sha256"] == digest(original)
            assert saved["identity"]["selected_ids_sha256"] == manifest["selected_ids_sha256"]
            assert saved["method"] == method
            assert saved["projected"] == (method == "oplora16")
            factors = saved["factors"]
            assert len(factors) == 196
            for name, entry in factors.items():
                base = model.get_submodule(name)
                wrapped = AdapterLinear(base, entry["A"].shape[0], 0)
                with torch.no_grad():
                    wrapped.A.copy_(entry["A"])
                    wrapped.B.copy_(entry["B"])
                wrapped.eval().requires_grad_(False)
                parent, leaf = name.rsplit(".", 1)
                setattr(model.get_submodule(parent), leaf, wrapped)
            test_input = tokenizer("Q: What is 2 + 3?\nA:", return_tensors="pt").to(device)
            with torch.no_grad():
                unmerged = model(**test_input, use_cache=False).logits.float()
                merge_factors(model, factors)
                merged = model(**test_input, use_cache=False).logits.float()
            difference = (merged - unmerged).abs()
            details["merge_check"] = {
                "dtype": "float32", "max_absolute_difference": float(difference.max()),
                "relative_l2_difference": float(difference.norm() / unmerged.norm().clamp_min(1e-8)),
            }
            torch.testing.assert_close(merged, unmerged, rtol=5e-4, atol=2e-3)
            state = factors
    return model, tokenizer, state, details


def get_probe_input(tokenizer, row):
    if row["chat"]:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
    return tokenizer.encode(row["prompt"], add_special_tokens=False)


def reference_diagnostics(task_dir, model, tokenizer, device):
    config, _ = load_inputs(task_dir)
    prompts = json.loads((task_dir / "splits/probe_prompts.json").read_text())
    identity = {"model_revision": config["model_revision"], "probes_sha256": digest(prompts),
                "activation_forward_dtype": "float32", "protection_rank": 128}
    path = task_dir / "results/diagnostics/baseline_reference.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity
        return saved
    inference_setup(model)
    assert next(model.parameters()).dtype == torch.float32
    saved = {"identity": identity, "probes": prompts, "layers": {}}
    for layer in LAYERS:
        module_name = f"model.layers.{layer}.mlp.down_proj"
        weight = model.get_submodule(module_name).weight.detach().float()
        basis_path = task_dir / "data/oplora_bases" / config["model_revision"] / (module_name + ".pt")
        if basis_path.exists():
            bases = torch.load(basis_path, map_location="cpu", weights_only=True)
        else:
            with torch.no_grad():
                u, _, vh = torch.linalg.svd(weight, full_matrices=False,
                                           **({"driver": "gesvd"} if device == "cuda" else {}))
                bases = {"U": u[:, :128].contiguous().cpu(), "V": vh[:128].T.contiguous().cpu()}
                del u, vh
            torch_write(basis_path, bases)
        saved["layers"][str(layer)] = {"W0": weight.cpu(), "U": bases["U"], "V": bases["V"], "z": []}
    captured = {}
    hooks = []
    for layer in LAYERS:
        def hook(module, inputs, selected_layer=layer):
            captured[selected_layer] = inputs[0][0, -1].detach().float().cpu()
        hooks.append(model.get_submodule(f"model.layers.{layer}.mlp.down_proj").register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            for row in prompts:
                captured.clear()
                ids = torch.tensor([get_probe_input(tokenizer, row)], device=device)
                # Run the transformer only: no unnecessary vocabulary projection.
                model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                for layer in LAYERS:
                    saved["layers"][str(layer)]["z"].append(captured[layer])
    finally:
        for hook in hooks:
            hook.remove()
    for layer in LAYERS:
        saved["layers"][str(layer)]["z"] = torch.stack(saved["layers"][str(layer)]["z"])
    torch_write(path, saved)
    return saved


def local_diagnostic(w0, delta, u, z):
    assert all(value.dtype == torch.float32 for value in (w0, delta, u, z))
    update_squared = delta.square().sum()
    rho = None if update_squared.item() == 0 else float((u.T @ delta).square().sum() / update_squared)
    eta = float(delta.norm() / w0.norm())
    ratios = (z @ delta.T).norm(dim=1) / ((z @ w0.T).norm(dim=1) + 1e-8)
    return rho, eta, ratios


def condition_diagnostics(task_dir, condition, state, reference, device):
    rows = []
    for layer in LAYERS:
        entry = reference["layers"][str(layer)]
        w0, u, z = [entry[key].to(device).float() for key in ("W0", "U", "z")]
        if condition == "baseline":
            delta = torch.zeros_like(w0)
        elif CONDITIONS[condition][0] in ("full", "l2sp"):
            delta = state[f"model.layers.{layer}.mlp.down_proj.weight"].to(device).float() - w0
        else:
            factors = state[f"model.layers.{layer}.mlp.down_proj"]
            delta = factors["scale"] * (factors["B"].to(device).float() @ factors["A"].to(device).float())
        rho, eta, ratios = local_diagnostic(w0, delta, u, z)
        rows.append({"layer": layer, "rho_k": rho, "eta": eta,
                     "activation_ratios": [{"task": probe["task"], "row_index": probe["row_index"],
                                            "R": float(ratios[index])}
                                           for index, probe in enumerate(reference["probes"])]})
    json_write(task_dir / "results/diagnostics" / (condition + ".json"), rows)


def qualitative(model, tokenizer, task_dir, output_dir, device, smoke):
    prompts = json.loads((task_dir / "splits/qualitative_prompts.json").read_text())
    if smoke:
        prompts = prompts[:2]
    path = output_dir / "qualitative.json"
    responses = json.loads(path.read_text()) if path.exists() else []
    done = {row["id"] for row in responses}
    for row in prompts:
        if row["id"] in done:
            continue
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False,
                                            return_dict=False)
        tensor = torch.tensor([ids], device=device)
        cap = 16 if smoke else 512
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor),
                                    max_new_tokens=cap, do_sample=False, use_cache=True,
                                    pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        eos = model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        record = generation_metadata(tokenizer, ids, output[0, len(ids):].tolist(), cap, [], eos)
        responses.append({**row, **record, "response": tokenizer.decode(record["output_token_ids"], skip_special_tokens=True),
                          "generation_seconds": time.perf_counter() - start})
        json_write(path, responses)
        print(f"Saved qualitative response {len(responses)}/{len(prompts)}: {row['id']}", flush=True)


def evaluate_condition(task_dir, condition, smoke=False):
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    assert torch.cuda.is_available(), "Evaluation requires CUDA; --prepare is CPU-only."
    assert metadata.version("lm_eval") == HARNESS_VERSION
    seed_all(6304)
    config_path = task_dir / "configs/evaluation_config.json"
    evaluation_config = json.loads(config_path.read_text())
    assert evaluation_config["harness_sha256"] == harness_fingerprint()
    output_dir = task_dir / ("smoke_evaluation" if smoke else "results/evaluation") / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {"version": VERSION, "evaluation_config_sha256": digest(evaluation_config),
                "condition": condition, "smoke": smoke,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, "Evaluation setup changed."
    else:
        json_write(identity_path, identity)
    if (output_dir / "finished.json").exists():
        print(f"{condition}: evaluation already finished; skipping.", flush=True)
        return
    device = "cuda"
    start = time.perf_counter()
    model, tokenizer, state, details = load_condition(task_dir, condition, device)
    # Diagnostics use original FP32 weights/activations; benchmark weights become FP16 below.
    if condition == "baseline":
        reference = reference_diagnostics(task_dir, model, tokenizer, device)
    else:
        reference = torch.load(task_dir / "results/diagnostics/baseline_reference.pt",
                               map_location="cpu", weights_only=True)
    condition_diagnostics(task_dir, condition, state, reference, device)
    del state, reference
    model.half()
    inference_setup(model)
    details["load_and_diagnostics_seconds"] = time.perf_counter() - start
    json_write(output_dir / "model_details.json", details)
    torch.cuda.reset_peak_memory_stats()
    RecordedHFLM = recorded_hflm_class()
    wrapper = RecordedHFLM(
        pretrained=model, tokenizer=tokenizer, backend="causal",
        batch_size=evaluation_config["likelihood_batch_size"],
        add_bos_token=False, enable_thinking=False, softmax_dtype=torch.float32,
        cache_dir=output_dir / "cache",
    )
    manager = TaskManager()
    # Save each complete benchmark; JSONL request cache also retains partial progress.
    for name in ("hellaswag", "boolq", "arc_easy", "gsm8k_cot"):
        wrapper.batch_size_per_gpu = evaluation_config[
            "generation_batch_size" if name == "gsm8k_cot" else "likelihood_batch_size"
        ]
        metric_path = output_dir / f"{name}_metrics.json"
        if metric_path.exists():
            print(f"{condition}/{name}: using completed results", flush=True)
            continue
        task = make_task(manager, task_dir, name, evaluation_config)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = evaluator.simple_evaluate(
            model=wrapper, tasks=[task], num_fewshot=0,
            apply_chat_template=(name == "gsm8k_cot"), fewshot_as_multiturn=False,
            limit=2 if smoke else None, log_samples=True, bootstrap_iters=0,
            random_seed=6304, numpy_random_seed=6304,
            torch_random_seed=6304, fewshot_random_seed=6304,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        samples = result.pop("samples")[name]
        sample_ids = {sample["doc_id"] for sample in samples}
        assert len(sample_ids) == (2 if smoke else BENCHMARKS[name]), (name, len(sample_ids))
        if name == "gsm8k_cot":
            for sample in samples:
                prompt = sample["arguments"][0][0]
                key = token_key(wrapper.tok_encode(prompt))
                assert key in wrapper.raw_generations, "Missing generation accounting for GSM8K sample."
                sample["generation"] = wrapper.raw_generations[key]
            flexible = [sample for sample in samples if sample["filter"] == "flexible-extract"]
            assert len(flexible) == len(sample_ids)
            json_write(output_dir / "gsm8k_extraction_summary.json", {
                "questions": len(flexible),
                "truncation_rate": sum(sample["generation"]["truncated"] for sample in flexible) / len(flexible),
                "missing_extraction_rate": sum(sample["filtered_resps"][0] in ("[invalid]", "", None)
                                               for sample in flexible) / len(flexible),
            })
        sample_path = output_dir / f"{name}_samples.jsonl"
        temporary = sample_path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(serializable(sample), ensure_ascii=False) + "\n" for sample in samples))
        temporary.replace(sample_path)
        json_write(metric_path, {"condition": condition, "smoke": smoke,
                                 "elapsed_seconds_this_session": elapsed,
                                 "evaluation": serializable(result)})
        print(f"Completed {condition}/{name}: {result['results'][name]}", flush=True)
    qualitative(model, tokenizer, task_dir, output_dir, device, smoke)
    json_write(output_dir / "finished.json", {
        **identity, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "session_seconds": time.perf_counter() - start,
        "note": "Partial request-cache hits may reduce session runtime after recovery.",
    })
    print(f"Evaluation complete: {condition}; smoke={smoke}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--condition", choices=list(CONDITIONS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare_evaluation(args.task_dir)
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        evaluate_condition(args.task_dir, args.condition, args.smoke)


if __name__ == "__main__":
    main()
