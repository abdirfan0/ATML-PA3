# Task 2 — Learning a visual interface

The required 196-token SigLIP–Qwen3-0.6B run completed all three stages, 4500 updates and 72000 example presentations. Final evaluation contains 1280 generated answers across two fixed test suites and four image conditions. The optional 49-token experiment was not run.

- [Task2.ipynb](Task2.ipynb): refined notebook with saved outputs; default execution is CPU report mode. GPU reproduction is disabled unless explicitly enabled.
- [Results and interpretation](analysis/RESULTS.md), [complete final tables](analysis/final_results.md), [resources](analysis/resources.md), [audit](analysis/audit.json).
- [Learning curves](figures/learning_curves.png), [swap penalties](figures/image_swap_penalties.png), [final accuracy](figures/final_accuracy.png), [question types](figures/accuracy_by_type.png), [swap behavior](figures/swap_answer_behavior.png), and five preselected [qualitative panels](figures/qualitative).
- `analysis/*.csv` and `analysis/qualitative_examples.json`: machine-readable tables and raw illustrative outputs.

## Reproduce the analysis on CPU

From this folder, with NumPy, Matplotlib and Pillow installed:

```bash
python code/task2_analyze.py --task-dir .
```

For the original Drive experiment, add `--require-images` to regenerate the qualitative panels directly from saved JPEG TARs. Repository copies already contain those panels. No torch import, model download, training or inference is required for analysis. The clean notebook uses the same command. Open its saved outputs directly on GitHub or run it on CPU in Colab.

## Reproduce the experiment

The original experiment remains at `/content/drive/MyDrive/ATML/PA3/task2`; original Qwen assets are shared from Task 1's original base-model snapshot, not Task 1 fine-tuned weights.

1. Read `code/DATA_PREPARATION_README.md`, `code/IMAGE_PREPARATION_README.md`, and `code/VLM_PREFLIGHT_README.md` for CPU preparation and fixed selection details. Sources and asset revisions/hashes are recorded under `configs/`. Data selection requires `ijson==3.6.0` in addition to the recorded environment packages.
2. Keep the original JPEG bytes and create the CPU selection/image/preflight records before allocating a GPU. The pinned Qwen and SigLIP files must be available at their recorded paths. The preparation scripts deliberately fail on changed assets/identities rather than silently replace data.
3. On CPU: `python code/task2_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task2 --prepare`.
4. On the recorded A100 CUDA environment (`configs/gpu_environment_task2.json`): `python code/task2_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task2 --run-all`. The queue caches frozen vision features, tests the real GPU path, trains the three-stage curriculum, and evaluates the final Stage 3 model. The notebook's optional GPU cell releases Colab after the queue. See `code/RUNNER_README.md` for checkpoint/resume and implementation details.
5. Run CPU finalization and the notebook; inspect the interpretation and qualitative examples before incorporating findings into the combined report.

The run used PyTorch 2.11.0+cu130, Transformers 5.20.2, datasets 4.8.5, accelerate 1.15.0, peft 0.21.2, Pillow 11.3.0, NumPy 2.1.3, tokenizers 0.23.3 and huggingface_hub 1.33.0. Preserve CUDA torch when pinning a fresh GPU runtime; do not install the CPU torch build on GPU. Data preparation and report viewing require no paid GPU.

## Files and evidence limits

`code/` preserves the executed preparation/model/training/evaluation scripts, original tests, and the separate CPU analysis script. `configs/` and `splits/` preserve the selected IDs and recipe. `results/` includes all per-example final predictions, original references, NLLs, flags, image controls and development records. `checkpoints/main/` contains training logs, resource/calibration records and checkpoint completion markers. `features/patch196/` contains feature identity, benchmark and shard metadata. The image archive manifest is retained under `data/images/`.

Model weights, raw metadata archives, JPEG TARs, feature arrays and checkpoint tensors stay in the Drive experiment, outside git. The CPU audit checks available record bytes and recomputes statistics. Tensor/array content cannot be verified from metadata alone.

Annotation-derived answers have limitations. All 22 grounding contact sheets were inspected by the assistant before training; 30/88 pairs were flagged for possible ambiguity, not confirmed wrong. This was not independent human adjudication. No primary examples were removed or relabeled. Natural VQA results are high-consensus local subsets. One training seed and pair-bootstrap intervals do not capture all training uncertainty. No-image outputs often become verbose or capped, so zero answer scores combine missing visual evidence with format changes.

This analysis is a draft for student review. Keep the final combined submission consistent with the course's authorship and assistance-disclosure rules.
