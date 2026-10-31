# Task 2 results and interpretation

The final Stage 3 checkpoint was evaluated on fixed, image-disjoint local subsets. No final-test score selected the checkpoint. Grounding uses normalized exact match and natural VQA uses the ten-answer soft consensus score. Intervals resample complete pairs 4000 times, reusing pair indices across image conditions.

## Curriculum

Caption development loss decreased from 5.8840 before training to 2.5623 after caption alignment and 2.5400 after Stage 3. This is reference-likelihood evidence for caption alignment; no caption-generation evaluation was performed. Grounding development swap penalties were near zero after Stage 1 (0.0179), reached 3.1992 after Stage 2, and ended at 2.1299 after natural VQA specialization. Natural VQA penalties grew from 0.1334 after Stage 1 to 1.1892 after Stage 2 and ended at 1.1419. Visual instruction training introduced substantially stronger image-conditioned answer likelihoods. Later natural specialization did not improve every development measure monotonically. Training losses across stages mix different target lengths and tasks, so their absolute levels are not directly comparable.

## Final accuracy and grounding

Grounding correct-image exact match was 69.53 [60.94, 77.34]%; complementary swaps scored 12.50 [7.03, 18.75]%. The paired advantage was 57.03 [44.53, 68.75] percentage points. Presence was easier than counting: 87.50% versus 51.56%.

Natural VQA correct-image soft accuracy was 58.96 [52.45, 65.42]%; complementary swaps scored 28.70 [23.28, 34.38]%. Its paired advantage was 30.26 [21.30, 39.48] percentage points. Correct-image scores were 75.83% yes/no, 44.79% number, and 51.53% other.

## What the controls measure

Correct images improve both generated accuracy and teacher-forced original-reference likelihood. Complementary swaps preserve the question and visual-token count, making their paired comparison stronger evidence of image use than the no-image contrast alone. Mismatched inputs use an unrelated image from the same suite and keep original references fixed. Swap correctness is always scored against the original image references; partner-answer following is reported separately.

For grounding, 79.69% of answers changed after swapping. Among all swapped predictions, 69.53% matched the partner canonical answer, 12.50% matched the original canonical answer, and 17.97% matched neither. These are canonical-string diagnostics; natural VQA soft credit can differ.

For natural, 47.92% of answers changed after swapping. Among all swapped predictions, 55.21% matched the partner canonical answer, 24.48% matched the original canonical answer, and 20.31% matched neither. These are canonical-string diagnostics; natural VQA soft credit can differ.

Both no-image suites scored zero under the required exact/soft answer scoring, with no empty outputs. This does not mean the model produced no relevant words: many responses became explanatory sentences rather than canonical short answers. Grounding had 50/128 capped outputs and natural VQA had 114/192. Omitting 196 visual tokens also shortens the prefix and shifts the model away from its trained image-conditioned input format. The zero score therefore combines lost visual information with format and distribution changes; it is not a clean estimate of language-only knowledge. No substring extraction or post-hoc answer rewriting was applied.

## Pair-level and qualitative evidence

Pair accuracy requires both responses to match their original canonical targets. Identical-answer rates count equal normalized responses, including empty responses if present. The complete tables report both metrics alongside reference NLL and generation time. Five illustrative pairs were fixed before predictions: the first sorted pair per question type, excluding grounding pairs flagged in the pre-training review. They illustrate behavior and are not a random estimate of success frequency. Consult the aligned original-image panels and raw responses rather than inferring visual correctness from answer changes alone.

In the selected fire-hydrant presence pair, correct-image outputs are yes/no and complementary outputs reverse to no/yes, matching the partner-image answers. In the stop-sign counting pair, the model predicts 1/2 against annotation targets 3/4; swapping reverses its predictions to 2/1 but neither original target is recovered. This is image sensitivity without accurate counting.

The selected natural pairs show failures that aggregate accuracy alone hides. The model answers 2 for both bird-eye images (targets 2/1), closed for both shower curtains (closed/open), and yes for both shorts questions (yes/no). These answers remain unchanged under complementary swaps, so each pair has one canonical hit but zero pair accuracy. The no-image bird response contains the numeral 2 inside a full sentence and receives zero under the required whole-answer scoring; this illustrates the format limitation without changing the scorer.

## Limits

- Checkpoint tensor files, cached feature arrays, original model weights and image TARs are not included in this review export; their recorded hashes are retained but their bytes were not re-audited here.
- Raw tokenized examples and calibration token lists are not in the review export. The prior CPU and GPU preflight records and frozen input hashes are available.
- The 30 annotation-ambiguity flags came from assistant contact-sheet inspection before training, not independent human adjudication; primary scores retain every fixed example.
- Single seed and selected high-consensus subsets; bootstrap intervals quantify pair sampling variation, not training-seed uncertainty or full-benchmark performance.
- Annotation-derived absence means no recorded object annotation, and counts depend on COCO object instances, occlusion and category conventions. Thirty of 88 reviewed development/test grounding pairs were flagged for possible ambiguity; this is not evidence that all 30 labels are wrong. All examples remain in primary scores.
- A 16-new-token cap can affect verbose outputs, particularly the no-image condition. Timing reflects this A100 run, batch size and implementation; per-answer timings are amortized over generation batches.
- Only the required 196-token recipe was run; the optional 49-token experiment was not performed.
- These notes are an analysis draft for student review and the combined report, not an independently adjudicated annotation study.
