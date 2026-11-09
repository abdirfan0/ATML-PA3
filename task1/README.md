# Task 1: mathematical adaptation and retention

**Status:** five completed runs, seven completed evaluation conditions, 28 aggregate benchmark results, 168 qualitative responses; all 42 required qualitative annotations reviewed by Abdullah Irfan. Small-file evidence audit passed. The CPU Drive finalizer checks the large-state file sizes and the per-example benchmark outputs; its result belongs in `analysis/drive_audit.json`. No further GPU work is scheduled.

## Files

| Path | Contents |
|---|---|
| `code/runner_v1/` | Exact executed trainer, evaluator, queue, batch benchmark and CPU tests; code hashes match run identities |
| `code/prepare_training_data.py` | Reconstructed CPU preparation with exact saved-ID and aggregate-token checks; see caveat below |
| `configs/` | Pinned model/dataset revisions, package versions and experiment/evaluation settings |
| `splits/` | Exact training source IDs, activation probes and canonical qualitative prompts |
| `assets/first_plot_questions.yaml` | Supplied assignment prompt asset; hash recorded in configs |
| `checkpoints/` | Small training logs, setup, run identities, displacement and recovery metadata; tensor states excluded |
| `results/evaluation/` | Original metrics and qualitative responses; finalizer adds `*_samples.jsonl.gz` |
| `results/diagnostics/` | FP32 local weight/activation diagnostic results |
| `analysis/` | CPU audit, tables, response grouping, reviewed annotations, preserved original drafts and interpretation |
| `figures/` | PNG/PDF figures rebuilt from saved evidence |

## Recorded recipe

Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, seed 6304, thinking disabled. First 40,000 eligible MetaMathQA examples in source order; no shuffling or answer truncation. Empty examples, formatted length over 2048, and normalized GSM8K-test overlap are excluded. Assistant-response plus end-of-turn tokens are supervised. All runs contain 7,100,056 supervised tokens.

One epoch, 1,250 successful updates, microbatch 2 × accumulation 16; AdamW betas (0.9,0.999), epsilon 1e-8, weight decay 0.01 excluding biases and parameters with fewer than two dimensions, global clipping 1.0, cosine schedule with 38 warmup updates. FP16 AMP with dynamic scaling and gradient checkpointing; no quantization. Full FT uses FP32 parameters, adapters use frozen FP16 backbone and FP32 factors. There were no skipped updates in the completed runs.

Full FT/L2-SP peak LR 2e-5; L2-SP λ=1e-3 and Ω is half the sum of squared displacement over unique original parameters. LoRA ranks 4/16, α=8/32, LR 2e-4, dropout 0.05, no trainable biases, all seven attention/MLP projections in all 28 layers. OPLoRA uses rank 16, α=32, protection rank 128, matched unprojected initialization, FP32 SVD bases and double-projected effective factors. Exported inference factors are projected.

Recovery updates: 312, 625, 937, 1250. Evaluated: baseline, Full FT midpoint, and five final variants. lm_eval 0.4.13; GSM8K test 1319, HellaSwag validation 10042, BoolQ validation 3270, ARC-Easy test 2376. Zero-shot throughout; GSM8K non-thinking chat/greedy/1024-token cap/stock filters; multiple-choice stock completion likelihood prompts without chat. Generation batch 64 and likelihood batch 16 were selected by the saved benchmark. Qualitative cap 512, one greedy response per prompt.

## Finish the current experiment on CPU

From the repo root in Colab, with Drive mounted:

```bash
python finalize_task1.py --source-task-dir /content/drive/MyDrive/ATML/PA3/task1
```

It preserves the original experiment folder, checks all 20 state-file sizes, validates full per-example coverage and recomputes scores, compresses the 28 sample JSONLs into this repository, and copies the original preparation manifest. It does not load any model or require torch. Source Arrow datasets, cached weights and recovery states remain on Drive. It does not publish to GitHub.

Regenerate small-file analysis/figures with NumPy and Matplotlib on CPU:

```bash
python task1/analysis/analyze_task1.py
python task1/analysis/annotate_task1.py
```

All 42 required responses were reviewed individually by Abdullah Irfan with assistant guidance on 9 November 2026. `analysis/qualitative_annotations_reviewed.csv` is authoritative; the original assistant drafts remain as provenance. `annotate_task1.py` now prefers the reviewed file and regenerates JSON, summary counts and a review receipt without overwriting the CSV. See `analysis/qualitative_review_summary.md` and `analysis/qualitative_review_receipt.json`. No model inference is performed.

## Reproduce from scratch

The originally executed training/evaluation source is preserved byte for byte. Do not edit these files in an existing resumable run: identities bind source SHA256 to checkpoint/evaluation settings. The source-pinned environments are recorded in `configs/preprocessing_environment.json` and `configs/gpu_environment.json`. Torch CPU and CUDA wheels differ; use a matching CUDA build on a fresh GPU runtime. The recorded GPU was an A100-SXM4-40GB.

Keep the existing `configs`, `splits`, prompt asset and original small preparation manifest in a new task directory. Rebuild data on CPU with `python task1/code/prepare_training_data.py --task-dir NEW_TASK_DIR`, then use `runner_v1/task1_train.py --task-dir NEW_TASK_DIR --prepare` to validate and persist weights. The preparation script was reconstructed after the experiment from the recorded rules. It has been syntax checked but not rerun against downloaded datasets here; it fails before enabling training if selected source IDs, supervised-token total or maximum length differ. The original Colab notebook was not included in the review archive. Preserve it under `notebooks/` if available.

On CPU, install the pinned evaluation requirements without replacing protected package versions, run `runner_v1/task1_eval.py --task-dir NEW_TASK_DIR --prepare`, and run the CPU unit tests. Only then allocate a GPU for smoke, batch benchmarking and the queue. See the executed runner README files for CLI options. The completed run's saved metrics, logs and completion flags should be kept in this repository, not copied into an empty reproduction run where they could trigger skipping.

The queue's `--run-all` resumes from complete recovery checkpoints, skips already complete jobs, and writes a queue completion flag. Automatic Colab release is supplied by the outer notebook's `finally: runtime.unassign()`, not by the command-line runner itself. The first session disconnected during LoRA-16 before a recovery checkpoint; its abandoned log is preserved separately as `training_before_disconnect.jsonl` and excluded from successful learning curves.

## Resource accounting and limitations

`analysis/resources.csv` reports cumulative successful-run optimizer-loop time, allocated/reserved GPU peaks separately, total bytes for all four recovery states, and method-specific setup/export overhead. Training time excludes checkpoint writes, model loading, evaluation, downloads and the abandoned partial LoRA-16 attempt; it is not total charged Colab time. No compute-unit total was recorded. Full FT's reference is stored on CPU for displacement; L2-SP's reference is held on GPU for its penalty.

Large tensors were intentionally excluded from the review archive. The Drive finalizer checks state existence/byte count and does not reload optimizer tensors, independently verify checkpoint contents, or recompute GPU diagnostics. Saved projected-factor merge checks and FP32 diagnostic results are included. All comparisons are one seed; small score differences do not establish statistical significance or a universal ranking.

Keep `.gitignore` when publishing. Do not add raw weights, raw dataset snapshots, `.pt` activation caches, credentials, or the large recovery states. Compressed per-example outputs are generated model outputs and small benchmark records needed to audit the reported results; they are preserved by the finalizer.
