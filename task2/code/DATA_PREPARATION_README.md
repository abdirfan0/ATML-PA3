# Task 2 data selection — CPU

Run `test_task2_data.py`, then `task2_select_data.py --task-dir TASK2 --qwen-source ORIGINAL_QWEN_SNAPSHOT`. Use the Qwen revision saved for Task 1, not any fine-tuned Task 1 model. Dependencies: Transformers 5.20.2, NumPy (the recorded CPU version), and ijson. No image downloads, feature extraction, model weights or GPU calls occur here.

## Required sets

- Captions: 6,000 training images with three captions each; 96 development images with one caption each.
- Natural VQA: training 1,536 pairs (640 yes/no, 256 number, 640 other); development 32 pairs (12/8/12); final test 96 pairs (36/24/36). Every retained side has at least eight of ten normalized human answers agreeing, and the two normalized consensus answers differ. Both sides have identical question text, different images and matching dataset answer types. All ten raw answers and an original consensus-group target string are retained.
- Annotation grounding: reusable training pools of 4,096 presence pairs and 4,096 counting pairs. The manual does not prescribe a unique training-pool size. Development contains 12 presence + 12 counting pairs; final test contains 32 + 32. These equal splits are our fixed design choice, not an additional manual requirement.

## Selection rules

One NumPy `default_rng(6304)` controls selection. The exact order is in `configs/data_selection_config.json`. Inputs are sorted before seeded selection. Exact image, question, caption, source-pair index and pair IDs are preserved in the split JSONs.

Natural pairs are selected with unique images within each selected split. Training natural images are included in the caption-training image pool, which is filled to exactly 6,000 images. Annotation-derived training pairs also use those 6,000 images. Training image reuse across stages is allowed and reduces image downloads and feature caching. Pair pools are sampled later with the manual's probabilities; this preparation does not impose exact training-mixture quotas.

All five evaluation image sets (caption dev, grounding dev/test, natural dev/test) are mutually image-disjoint, and no evaluation image enters training/replay. There are 6,000 training images and 528 evaluation images, 6,528 total, if all selection checks pass. FP16 196×768 features require approximately 1.83 GiB before metadata/format overhead.

A presence positive has any annotation of the category, including crowd. A negative has none; COCO annotations may omit objects, so absence of annotation is not proof of visible absence. Counting uses non-crowd instances at counts 1–4, excluding image/category combinations with crowd annotations. Pair IDs distinguish question kind, category and the two images. Evaluation images/labels should be inspected independently of model predictions after images are downloaded, before training. Do not choose evaluation examples on model success.

Caption/answer/question content limits are checked with the pinned Qwen tokenizer (64/16/80). Overlength captions and natural pairs are filtered before selection; no answer is truncated. Canonical targets are used for normalized matching, while original strings are retained for training and raw references for scoring.

The `task2_answers.py` normalizer adapts the official VQA punctuation, number/article and contraction rules from GT-Vision-Lab/VQA commit a013f0043c1e2cdc995922dfe257f7149aa9af06. It always normalizes both predictions and references, as required by this assignment, rather than using the old evaluator's conditional branch. The local soft-consensus formula implements leave-one-out averaging across all ten references; it is not simply min(matches/3,1). Empty predictions receive zero.

Mismatched-image maps are fixed now, before generation, within each final-test suite. A seeded pair permutation maps each pair to its cyclic successor; one image from that other pair is chosen per example. Both images of the original pair are excluded. Complementary-swap references remain the original references. Development evaluation uses correct, complementary and no-image conditions; it does not need mismatched maps.

## Saved outputs

`configs/data_selection_config.json`: source/model/tokenizer/code/environment identity and exact selection rules.

`splits/`: caption records, natural pairs, generated grounding pools, selected image catalog, fixed image controls, 128 calibration image IDs from training only, and a SHA256 selection manifest.

`results/data_selection.json`: required sizes/type counts, token-cap and image-isolation checks, filtered candidate statistics, and feature-storage estimate.

Rerunning reuses a completed selection only if its identity and saved output hashes match. Preserve established splits; do not silently replace them after inspecting development or final-test predictions. Final-test results must not be used for checkpoint selection.
