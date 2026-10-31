# PA3 Task 1: training runner, version 1

This package prepares and trains Task 1 only. It is not the completed Task 1
submission: benchmark evaluation, activation/weight diagnostics, qualitative
generation/annotation, plotting, and final repository organization are still to
be added. Do not treat training completion as assignment completion.

## CPU first

Use the CPU Colab runtime for installation, tests, and persistent downloads.
Existing Drive preparation is expected at `/content/drive/MyDrive/ATML/PA3/task1`.
This program never reselects or retokenizes training examples.

```python
import subprocess, sys
subprocess.run([sys.executable, '-m', 'unittest', 'discover',
                '-s', str(CODE), '-p', 'test_cpu.py', '-v'], check=True)
subprocess.run([sys.executable, str(CODE / 'task1_train.py'),
                '--task-dir', str(TASK1), '--prepare'], check=True)
```

`--prepare` validates saved input identities, builds the exact 24 qualitative
chat messages (including JSON system prompts), marks the fixed 6-prompt annotation
subset, and downloads the pinned original model and tokenizer into Drive.
It does not construct a model or run a forward pass.

## GPU only after the remaining evaluation code is ready

The following commands document the interface, not an instruction to switch GPU
now. Test two updates of Full FT and LoRA-16 on the GPU before full runs. Smoke
runs are separate from real runs and do not create large recovery states.

```
python task1_train.py --task-dir /content/drive/MyDrive/ATML/PA3/task1 --method full --smoke-updates 2
python task1_train.py --task-dir /content/drive/MyDrive/ATML/PA3/task1 --method lora16 --smoke-updates 2
```

Train each independent method with `--method full`, `l2sp`, `lora4`, `lora16`,
or `oplora16`. Use a separate subprocess for each method to release GPU memory.
All real runs start from the original pinned checkpoint. Repeating a command
resumes the latest marked-complete recovery checkpoint. A completed run skips.
The original FP32 full-model master parameters, FP32 AdamW states, scheduler,
dynamic scaler, RNG, and data position are preserved in recovery states.

## Recipe and implementation choices

- 40,000 examples in the saved source order; 1,250 successful updates; microbatch
  2 and accumulation 16. No shuffle, packing, answer truncation, quantization, or replay.
- SFT is summed causal cross-entropy on response plus EOT, divided by all valid
  response tokens across the effective batch, rather than averaging microbatch losses.
- Full FT: FP32 master parameters, FP16 autocast, peak LR 2e-5.
- L2-SP: the same, with 1e-3 times one-half summed squared displacement from the
  original unique parameters. Biases and normalization parameters are included.
  Its fixed reference stays on GPU to avoid repeated host-to-device copies.
- AdamW betas (0.9, 0.999), epsilon 1e-8; decay 0.01 except biases and tensors
  of fewer than two dimensions; cosine schedule with 38 warmup updates; clip 1.0.
  The schedule follows the conventional LambdaLR warmup starting at zero LR.
- Frozen FP16 backbone, FP32 adapters and optimizer states, ranks 4/16, alpha
  8/32, dropout 0.05, no biases, all seven prescribed targets in all 28 layers.
- Local CPU initialization generators keyed by module name ensure identical
  unprojected LoRA-16 and OPLoRA-16 factors, independent of SVD caching.
- OPLoRA uses exact FP32 reduced SVD on GPU, k=128; only small leading bases
  are persisted and moved onto GPU. Effective factor projection is FP32, with
  no dense projection matrices. Inference exports contain the projected factors.
- Gradient checkpointing is non-reentrant. AMP gradients accumulate at one
  scaler scale, then unscale, clip, and step once per effective batch.
- A skipped AMP update is logged; the same effective batch is retried at the
  reduced scale. Scheduler/data position advance only after successful steps.
  Sixteen consecutive overflows abort rather than silently consuming the dataset.

## Storage and recovery

All four required recovery states are retained per method (updates 312, 625,
937, 1250). Complete Full FT states store FP32 model weights and both FP32 Adam
moments; two full-model methods times four snapshots need roughly 55-60 GiB,
plus source weights, adapters, and prepared data. Check actual Google Drive
quota before training; Colab local-disk free space is not a Drive-quota check.
Incomplete saves are ignored on resume. Intermediate work after the last
recovery checkpoint must be recomputed following a runtime termination.
The code never deletes a completed recovery checkpoint automatically.

Each saved recovery payload uses atomic temporary-file replacement followed
by a completion marker. Resume uses trusted local PyTorch payloads with
`weights_only=False` because Python/NumPy RNG state is included. Do not load
checkpoints from an untrusted source.

Training logs include SFT/regularization losses, learning rate, scaler skips,
tokens, update times, memory, and method-specific overhead. Full-model
displacement is saved at the required Full FT midpoint/final and L2-SP final.
Adapters use a small custom checkpoint format, not a PEFT adapter directory;
downstream evaluation must use the matching loader and merge projected factors.

## Validation and scope

Eight small CPU tests cover token-weighted accumulation gradients, causal
prompt/padding/EOT masks, tied-parameter L2-SP, weight-decay exclusions, double
projection, matching initialization, merged-versus-unmerged toy outputs,
adapter checkpoint contents, exact CPU recovery including dropout RNG, and
incomplete-checkpoint selection. These are arithmetic/recovery checks; they
do not substitute for actual Qwen GPU/AMP smoke tests or benchmark evaluation.

## Attribution

Implementation produced with ChatGPT coding assistance for this assignment.
No external training implementation has been copied. Equations and API usage:

- Assignment: *ATML-PA3*, released 23 October 2026, Task 1.
- Hu et al., *LoRA*, https://arxiv.org/abs/2106.09685
- Xiong and Xie, *OPLoRA*, Sections 3.2-3.3 and Proposition 2,
  https://arxiv.org/abs/2510.13003
- PyTorch AMP and gradient accumulation:
  https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Qwen model/tokenizer: https://huggingface.co/Qwen/Qwen3-0.6B
- MetaMathQA: https://huggingface.co/datasets/meta-math/MetaMathQA

You remain responsible for understanding and verifying the submitted code.
