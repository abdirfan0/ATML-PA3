# Task 2 visual interface — CPU preflight

This package adds reusable VLM components and checks; it is not the full training or GPU caching runner.

## Run on CPU

Run `python -m unittest discover -s CODE -p test_task2_vlm_cpu.py -v`, then `python CODE/task2_cpu_preflight.py --task-dir TASK2`.

The preflight verifies saved SigLIP file hashes, deterministic image processing and a single real training image's feature shape. It loads only the frozen SigLIP vision model on **CPU** and makes one forward pass. The complete pretrained checkpoint includes unused text-tower weights; the loader may report these as unexpected. The preflight permits only those known text/logit leftovers and asserts there are no missing or mismatched vision weights.

A forward hook verifies that the final vision LayerNorm runs exactly once and that its output equals `last_hidden_state`. Features are already normalized: the later cache must not apply the final LayerNorm again. Pooling is unused. The connector shape is 196×1024 and has 1,837,056 FP32 trainable parameters.

No full feature cache, Qwen weight download, training loop, baseline-development evaluation or GPU allocation occurs. The CPU result is saved to `results/vlm_cpu_preflight.json`.

## Visual interface

`task2_vlm.py` contains the two-layer 768→1024→1024 GELU connector, Qwen non-thinking message formatting, attention-only LoRA, masks/positions, and per-example causal loss. Visual embeddings are inserted immediately after the user-role header, before the question. Prompt/visual/padding labels are masked. Response labels include the end-of-turn token for SFT. Content-only labels omit termination for reference-answer NLL. Removing images removes exactly the visual-token span, retaining the text instructions and rebuilding masks/positions.

Original language-model parameters remain frozen. Autograd is retained through the language forward so answer losses train the connector. Non-reentrant checkpointing and adapter gradients are tested with a tiny random Qwen, not by training the real model on CPU. Stage 2/3 LoRA targets only q/k/v/o projections, rank 16, alpha 32, dropout 0.05; FP32 adapter factors and zero B preserve the initial function. The main runner must seed and manage the three stages, sampling, optimization and exact resume state separately.

The loss averages supervised token losses within each example, then the effective batch must average those example losses. It is deliberately different from Task 1's token-weighted loss. A fixed training-only global RMS calibration helper is included; actual calibration must be run after feature caching on the saved 128 training image IDs and training text tokens, before any Stage 0 development evaluation. The preflight does not calibrate or save a training connector.

## Image processing

Use PIL RGB conversion, deterministic bicubic resize to 224×224, FP32 division by 255, then `(x - 0.5) / 0.5`, with no random augmentations. These are the pinned SigLIP preprocessing settings. Record and match Pillow/NumPy/Transformers versions for later caching. The CPU preflight uses FP32 vision computation; later caching stores FP16 features and must record its computation dtype and environment.

## Label review

`task2_label_review_notes.json` records assistant inspection of all 22 supplied sheets, covering 88 grounding pairs/176 images, before training or predictions. Thirty pairs are flagged for uncertainty, small/occluded objects, reflections/depictions, or category/grouping conventions. These are not thirty proven wrong labels. Contact-sheet inspection has resolution limits and is not independent human adjudication.

The preflight validates that these notes refer to the exact supplied split/config/sheet-index hashes before copying them to `results/label_review_notes.json`. Primary COCO-derived targets and required dataset sizes are preserved; no examples are relabeled, dropped or resampled using review notes or model outputs. Keep these limitations in the eventual interpretation. Do not describe an annotation-derived count as an assured unique-physical-object count.

## Validation performed before packaging

Nine CPU tests passed: connector architecture/dtype/count; example-weighted loss and gradients; prompt/visual/padding/EOT masking; no-image span removal and left-padding positions; connector gradients through frozen checkpointed Qwen; adapter targets/identity/gradients; deterministic adapter initialization; RMS-calibration arithmetic; deterministic RGB preprocessing.

A real-image integration preflight additionally loaded the pinned pretrained SigLIP vision weights, confirmed no missing vision keys, checked a training image's 196×768 feature shape and single final LayerNorm, projected to 196×1024, and verified the real Qwen tokenizer's non-thinking prefix/EOT. These checks do not replace the eventual GPU/AMP smoke test and sampler/checkpoint recovery tests.
