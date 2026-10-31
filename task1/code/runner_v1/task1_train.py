"""PA3 Task 1 training only. See README.md before launch.

Independent implementation of the assignment recipe. Adapter equations follow
Hu et al., arXiv:2106.09685, and Xiong & Xie, arXiv:2510.13003.
AMP follows https://docs.pytorch.org/docs/stable/notes/amp_examples.html.
No evaluation scores or gold answers enter training.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "pa3-task1-train-v1"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
CHECKPOINTS = (312, 625, 937, 1250)
METHODS = ("full", "l2sp", "lora4", "lora16", "oplora16")


@dataclasses.dataclass(frozen=True)
class Recipe:
    method: str
    seed: int = 6304
    examples: int = 40000
    updates: int = 1250
    microbatch: int = 2
    accumulation: int = 16
    warmup: int = 38
    max_length: int = 2048
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    clip: float = 1.0

    @property
    def adapter(self):
        return self.method in ("lora4", "lora16", "oplora16")

    @property
    def rank(self):
        return 4 if self.method == "lora4" else 16

    @property
    def lr(self):
        return 2e-4 if self.adapter else 2e-5

    @property
    def penalty(self):
        return 1e-3 if self.method == "l2sp" else 0.0


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def torch_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # True FP32 bases / diagnostics; AMP matmuls still use FP16.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        import numpy as np
        np.random.set_state(state["numpy"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_tree(value, memo=None):
    if memo is None:
        memo = {}
    if isinstance(value, torch.Tensor):
        key = (value.device, value.dtype, value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        if key not in memo:
            memo[key] = value.detach().cpu()
        return memo[key]
    if isinstance(value, dict):
        return {key: cpu_tree(item, memo) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item, memo) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item, memo) for item in value)
    return value


def collate(rows, pad_id, device):
    width = max(len(row["input_ids"]) for row in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        size = len(row["input_ids"])
        ids[i, :size] = torch.tensor(row["input_ids"], dtype=torch.long)
        mask[i, :size] = 1
        labels[i, :size] = torch.tensor(row["labels"], dtype=torch.long)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}, labels.to(device)


def supervised_sum(logits, labels):
    """FP32 summed CE over valid causal response targets, including EOT.

    Select supervised positions before promoting to FP32; no dense FP32
    [batch, full_sequence, vocabulary] copy is needed.
    """
    targets = labels[:, 1:]
    keep = targets != -100
    return F.cross_entropy(logits[:, :-1][keep].float(), targets[keep], reduction="sum")


class AdapterLinear(nn.Module):
    """Frozen linear + FP32 low-rank parameters; no dense projectors."""

    def __init__(self, base, rank, seed, bases=None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.scale = 2.0  # alpha = 2*r for both required ranks
        self.dropout = nn.Dropout(0.05)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = torch.empty(rank, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(a.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.register_buffer("U", None if bases is None else bases[0].to(base.weight.device))
        self.register_buffer("V", None if bases is None else bases[1].to(base.weight.device))

    def factors(self):
        # Projection must stay FP32 even inside the outer FP16 autocast context.
        with torch.autocast(self.A.device.type, enabled=False):
            a, b = self.A, self.B
            if self.U is not None:
                b = b - self.U @ (self.U.T @ b)
                a = a - (a @ self.V) @ self.V.T
        return a, b

    def forward(self, x):
        a, b = self.factors()
        correction = F.linear(F.linear(self.dropout(x), a), b) * self.scale
        return self.base(x) + correction


def module_seed(name, seed):
    # Local CPU generators keep LoRA-16 and OPLoRA-16 initialization identical,
    # independent of SVD implementation, module creation, or cache hits.
    return (int(hashlib.sha256(name.encode()).hexdigest()[:12], 16) + seed) % (2**63 - 1)


def install_adapters(model, recipe, basis_dir):
    model.requires_grad_(False)
    selected = [(name, module) for name, module in model.named_modules()
                if name.rsplit(".", 1)[-1] in TARGETS and isinstance(module, nn.Linear)]
    assert len(selected) == 28 * 7, f"Unexpected target count: {len(selected)}"
    stats = {"basis_seconds": 0.0, "basis_bytes": 0, "initial_factors_sha256": ""}
    initialization_hash = hashlib.sha256()
    for i, (name, module) in enumerate(selected):
        bases = None
        if recipe.method == "oplora16":
            path = basis_dir / (name + ".pt")
            start = time.perf_counter()
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=True)
                u, v = saved["U"], saved["V"]
            else:
                with torch.no_grad(), torch.autocast("cuda", enabled=False):
                    u_all, _, vh_all = torch.linalg.svd(
                        module.weight.detach().float(), full_matrices=False, driver="gesvd"
                    )
                    u = u_all[:, :128].contiguous().cpu()
                    v = vh_all[:128].T.contiguous().cpu()
                    del u_all, vh_all
                torch_write(path, {"U": u, "V": v})
            assert u.shape == (module.out_features, 128)
            assert v.shape == (module.in_features, 128)
            stats["basis_seconds"] += time.perf_counter() - start
            stats["basis_bytes"] += (u.numel() + v.numel()) * 4
            bases = (u, v)
            print(f"OPLoRA basis {i+1}/{len(selected)}: {name}", flush=True)
        replacement = AdapterLinear(module, recipe.rank, module_seed(name, recipe.seed), bases)
        initialization_hash.update(replacement.A.detach().cpu().numpy().tobytes())
        initialization_hash.update(replacement.B.detach().cpu().numpy().tobytes())
        parent_name, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), leaf, replacement)
    stats["initial_factors_sha256"] = initialization_hash.hexdigest()
    return stats


def parameter_groups(model, recipe):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():  # removes tied duplicates
        if not parameter.requires_grad:
            continue
        destination = no_decay if parameter.ndim < 2 or name.endswith("bias") else decay
        destination.append(parameter)
    return [{"params": decay, "weight_decay": recipe.weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def starting_penalty(model, reference):
    # Sum, not a parameter-wise average. Unique named_parameters counts ties once.
    total = torch.zeros((), dtype=torch.float32, device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        total = total + (parameter - reference[name]).square().sum() * 0.5
    return total


@torch.no_grad()
def displacement(model, reference):
    total = torch.zeros((), dtype=torch.float64, device=next(model.parameters()).device)
    for name, parameter in model.named_parameters():
        delta = parameter.float() - reference[name].to(parameter.device)
        total += delta.square().sum(dtype=torch.float64)
    return float(total.sqrt().item())


def checkpoint_model(model, recipe):
    if not recipe.adapter:
        return cpu_tree(model.state_dict())
    return {name: parameter.detach().cpu() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def restore_model(model, state, recipe):
    result = model.load_state_dict(state, strict=not recipe.adapter)
    if recipe.adapter:
        expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        assert set(state) == expected
        assert not result.unexpected_keys


def save_training_checkpoint(path, model, optimizer, scheduler, scaler, recipe,
                             identity, update, elapsed, skipped):
    torch_write(path, {
        "version": VERSION, "recipe": dataclasses.asdict(recipe), "identity": identity,
        "update": update, "data_position": update * 32,
        "model": checkpoint_model(model, recipe),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "rng": rng_state(), "training_seconds": elapsed, "skipped_updates": skipped,
    })


def latest_checkpoint(run_dir):
    ready = []
    for path in run_dir.glob("update_*/training_state.pt"):
        if path.with_name("complete.json").exists():
            ready.append(path)
    return max(ready, key=lambda path: int(path.parent.name.split("_")[-1])) if ready else None


def load_inputs(task):
    config = json.loads((task / "configs/data_config.json").read_text())
    manifest = json.loads((task / "data/metamath_40k_manifest.json").read_text())
    validation = json.loads((task / "results/data_validation.json").read_text())
    assert digest(config) == manifest["config_sha256"]
    assert validation["passed"] and validation["examples"] == 40000
    assert validation["selected_ids_sha256"] == manifest["selected_ids_sha256"]
    assert config["seed"] == 6304 and config["max_sequence_length"] == 2048
    assert config["train_examples"] == 40000 and config["enable_thinking"] is False
    return config, manifest


def prepare(task):
    """CPU-only input validation + persistent model download; no forward pass."""
    from datasets import load_from_disk
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    import yaml
    config, manifest = load_inputs(task)
    ds = load_from_disk(str(task / "data/metamath_40k"))
    assert len(ds) == 40000
    selected = [int(index) for index in ds["source_index"]]
    assert hashlib.sha256(json.dumps(selected).encode()).hexdigest() == manifest["selected_ids_sha256"]
    qualitative_path = task / "assets/first_plot_questions.yaml"
    raw = qualitative_path.read_bytes()
    entries = yaml.safe_load(raw)
    assert isinstance(entries, list) and len(entries) == 24
    assert len({entry["id"] for entry in entries}) == 24
    for entry in entries:
        assert len(entry["paraphrases"]) == 1
        assert isinstance(entry["paraphrases"][0], str)
        if "system" in entry:
            assert isinstance(entry["system"], str)
    expected = json.loads((task / "configs/qualitative_asset.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
    canonical_prompts = []
    plain_ids = [entry["id"] for entry in entries[:8]]
    for index, entry in enumerate(entries):
        format_name = ("plain", "json", "template")[index // 8]
        base_id = plain_ids[index % 8]
        expected_id = base_id + ("" if format_name == "plain" else "_" + format_name)
        assert entry["id"] == expected_id
        messages = []
        if "system" in entry:
            messages.append({"role": "system", "content": entry["system"]})
        messages.append({"role": "user", "content": entry["paraphrases"][0]})
        canonical_prompts.append({
            "id": entry["id"], "base_id": base_id, "format": format_name,
            "messages": messages, "annotate": index % 8 in (0, 7),
        })
    json_write(task / "splits/qualitative_prompts.json", canonical_prompts)
    destination = task / "assets/base_model" / config["model_revision"]
    snapshot_download(
        config["model_id"], revision=config["model_revision"], local_dir=str(destination),
        allow_patterns=["*.json", "*.safetensors", "*.model", "merges.txt", "vocab.*"],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(destination), local_files_only=True)
    assert tokenizer.convert_ids_to_tokens(tokenizer.convert_tokens_to_ids("<|im_end|>")) == "<|im_end|>"
    assert any(destination.glob("*.safetensors"))
    weights_bytes = sum(path.stat().st_size for path in destination.glob("*.safetensors"))
    # Two full-model variants x four recovery states x [FP32 weights + 2 FP32 moments].
    # Source weights are usually BF16; use parameter count from model configuration
    # only as an order-of-magnitude estimate, not as a Drive quota check.
    estimate = weights_bytes * 6 * 4 * 2
    report = {
        "passed": True, "version": VERSION, "model_path": str(destination),
        "model_revision": config["model_revision"], "dataset_rows": len(ds),
        "qualitative_prompts": len(entries), "source_weights_bytes": weights_bytes,
        "rough_full_run_recovery_storage_GiB": round(estimate / 1024**3, 1),
        "storage_note": "Four recovery states per method retained. Check actual Drive quota separately.",
    }
    json_write(task / "results/runner_cpu_preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("CPU preparation complete. No model was loaded onto GPU.", flush=True)


def train(task, method, smoke_updates=0):
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU. Use --prepare on CPU.")
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config, manifest = load_inputs(task)
    recipe = Recipe(method)
    seed_all(recipe.seed)
    source = task / "assets/base_model" / config["model_revision"]
    if not source.is_dir():
        raise RuntimeError("Run --prepare on CPU first to persist model downloads.")
    run_dir = task / ("smoke" if smoke_updates else "checkpoints") / method
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "config_sha256": digest(config), "selected_ids_sha256": manifest["selected_ids_sha256"],
        "model_revision": config["model_revision"], "runner_version": VERSION,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    run_config_path = run_dir / "run_config.json"
    run_config = {"recipe": dataclasses.asdict(recipe), "identity": identity}
    if run_config_path.exists():
        assert json.loads(run_config_path.read_text()) == run_config, "Run configuration changed."
    else:
        json_write(run_config_path, run_config)
    if not smoke_updates and (run_dir / "finished.json").exists():
        print(f"{method}: already completed; skipping.", flush=True)
        return

    local_data = Path("/content/pa3_training_data") if Path("/content").exists() else Path("/tmp/pa3_training_data")
    local_data = local_data / manifest["selected_ids_sha256"]
    if not (local_data / "READY").exists():
        if local_data.exists():
            shutil.rmtree(local_data)
        shutil.copytree(task / "data/metamath_40k", local_data)
        (local_data / "READY").write_text("complete")
    data = load_from_disk(str(local_data))
    assert len(data) == recipe.examples
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True,
        dtype=torch.float16 if recipe.adapter else torch.float32,
        attn_implementation="sdpa",
    ).to("cuda")
    assert model.config.num_hidden_layers == 28
    model.config.use_cache = False
    model_load_seconds = time.perf_counter() - start
    setup_stats = {}
    if recipe.adapter:
        setup_stats = install_adapters(model, recipe, task / "data/oplora_bases" / config["model_revision"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert all(parameter.dtype == torch.float32 for parameter in trainable)
    if recipe.adapter:
        assert all(parameter.dtype == torch.float16 for parameter in model.parameters() if not parameter.requires_grad)
    reference_start = time.perf_counter()
    reference = None
    if not recipe.adapter:
        # L2-SP keeps its reference on GPU; Full FT uses CPU reference only for
        # midpoint/final displacement. Unique named_parameters excludes ties.
        reference = {
            name: parameter.detach().clone() if recipe.penalty else parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
        }
    reference_seconds = time.perf_counter() - reference_start
    reference_bytes = sum(value.numel() * value.element_size() for value in (reference or {}).values())
    optimizer = torch.optim.AdamW(
        parameter_groups(model, recipe), lr=recipe.lr,
        betas=(recipe.beta1, recipe.beta2), eps=recipe.epsilon, foreach=False,
    )
    def schedule(step):
        if step < recipe.warmup:
            return step / recipe.warmup
        progress = (step - recipe.warmup) / (recipe.updates - recipe.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda")
    update, training_seconds, skipped = 0, 0.0, 0
    latest = None if smoke_updates else latest_checkpoint(run_dir)
    if latest:
        print(f"Resuming {latest}", flush=True)
        state = torch.load(latest, map_location="cpu", weights_only=False)
        assert state["version"] == VERSION and state["identity"] == identity
        assert state["recipe"] == dataclasses.asdict(recipe)
        restore_model(model, state["model"], recipe)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        update = state["update"]
        assert state["data_position"] == update * 32
        training_seconds, skipped = state["training_seconds"], state["skipped_updates"]
        set_rng(state["rng"])
        del state
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            opt_state = optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in opt_state:
                    assert opt_state[key].dtype == torch.float32
    log_path = run_dir / "training.jsonl"
    if log_path.exists():
        # A resumed run repeats work after its last complete recovery point.
        # Remove the abandoned tail rather than double counting it in plots.
        rows = []
        for line in log_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row["update"] <= update:
                rows.append(row)
        log_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    resources = {
        "method": method, "smoke": bool(smoke_updates), "gpu": torch.cuda.get_device_name(0),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_load_seconds": model_load_seconds, "reference_storage_bytes": reference_bytes,
        "reference_setup_seconds": reference_seconds, **setup_stats,
    }
    json_write(run_dir / "setup.json", resources)
    setup_peak_allocated = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    target = min(smoke_updates, recipe.updates) if smoke_updates else recipe.updates
    retries = 0
    while update < target:
        rows = [data[index] for index in range(update * 32, (update + 1) * 32)]
        valid_tokens = sum(sum(label != -100 for label in row["labels"][1:]) for row in rows)
        assert valid_tokens == sum(row["response_tokens"] for row in rows)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begin = time.perf_counter()
        sft_sum = 0.0
        for micro in range(recipe.accumulation):
            batch, labels = collate(rows[micro*2:(micro+1)*2], pad_id, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**batch, use_cache=False)
                loss_sum = supervised_sum(output.logits, labels)
                loss = loss_sum / valid_tokens
            scaler.scale(loss).backward()
            sft_sum += float(loss_sum.detach().item())
            del batch, labels, output, loss, loss_sum
        omega = 0.0
        if recipe.penalty:
            # Add once per effective batch, after normalized response loss.
            with torch.autocast("cuda", enabled=False):
                penalty = starting_penalty(model, reference)
            omega = float(penalty.detach().item())
            scaler.scale(recipe.penalty * penalty).backward()
            del penalty
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(trainable, recipe.clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        did_skip = scaler.get_scale() < old_scale
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        training_seconds += seconds
        if did_skip:
            skipped += 1
            retries += 1
        else:
            used_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            update += 1
            retries = 0
        row = {
            "update": update + 1 if did_skip else update, "skipped": did_skip,
            "examples_seen": update * 32, "sft_loss": sft_sum / valid_tokens,
            "omega": omega, "weighted_penalty": recipe.penalty * omega,
            "total_loss": sft_sum / valid_tokens + recipe.penalty * omega,
            "learning_rate": optimizer.param_groups[0]["lr"] if did_skip else used_lr,
            "response_tokens": valid_tokens, "gradient_norm": float(norm.item()) if torch.isfinite(norm) else None,
            "scaler": scaler.get_scale(), "update_seconds": seconds,
            "training_seconds": training_seconds, "skipped_updates_total": skipped,
        }
        with log_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if did_skip or update % 10 == 0 or smoke_updates:
            print(json.dumps(row), flush=True)
        if retries >= 16:
            raise RuntimeError("16 consecutive AMP skips. Stop and inspect before continuing.")
        if did_skip:
            continue  # Retry same examples; preserve the required accepted-update budget.
        if not smoke_updates and update in CHECKPOINTS:
            directory = run_dir / f"update_{update:04d}"
            directory.mkdir(exist_ok=True)
            if reference is not None and (update == 1250 or (method == "full" and update == 625)):
                json_write(directory / "displacement.json", {"l2": displacement(model, reference)})
            optimizer.zero_grad(set_to_none=True)
            begin_save = time.perf_counter()
            state_path = directory / "training_state.pt"
            save_training_checkpoint(state_path, model, optimizer, scheduler, scaler,
                                     recipe, identity, update, training_seconds, skipped)
            info = {"update": update, "examples": update * 32,
                    "bytes": state_path.stat().st_size,
                    "save_seconds": time.perf_counter() - begin_save}
            json_write(directory / "complete.json", info)
            print(f"Recovery checkpoint saved: {directory}", flush=True)
    optimizer.zero_grad(set_to_none=True)
    export_seconds = 0.0
    export_bytes = 0
    if recipe.adapter and not smoke_updates:
        export_start = time.perf_counter()
        factors = {}
        model.eval()
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, AdapterLinear):
                    a, b = module.factors()
                    factors[name] = {"A": a.cpu(), "B": b.cpu(), "scale": module.scale}
        export_path = run_dir / "adapter_inference.pt"
        torch_write(export_path, {"identity": identity, "method": method,
                                  "projected": method == "oplora16", "factors": factors})
        export_seconds = time.perf_counter() - export_start
        export_bytes = export_path.stat().st_size
    resources.update({
        "updates": update, "examples_seen": update * 32,
        "training_seconds": training_seconds, "skipped_updates": skipped,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "setup_peak_gpu_allocated_bytes": setup_peak_allocated,
        "overall_peak_gpu_allocated_bytes": max(setup_peak_allocated, torch.cuda.max_memory_allocated()),
        "adapter_export_seconds": export_seconds, "adapter_export_bytes": export_bytes,
        "checkpoint_bytes": sum(path.stat().st_size for path in run_dir.glob("update_*/training_state.pt")),
    })
    json_write(run_dir / ("smoke_summary.json" if smoke_updates else "finished.json"), resources)
    print(json.dumps(resources, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true", help="CPU validation and persistent download only")
    actions.add_argument("--method", choices=METHODS)
    parser.add_argument("--smoke-updates", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_updates < 0 or args.smoke_updates > 10:
        parser.error("Smoke budget must be between 0 and 10.")
    if args.prepare:
        prepare(args.task_dir)
    else:
        train(args.task_dir, args.method, args.smoke_updates)


if __name__ == "__main__":
    main()
