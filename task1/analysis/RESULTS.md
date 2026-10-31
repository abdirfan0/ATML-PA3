# Task 1 results and analysis notes

These notes are a draft grounded in the saved experiment outputs. They are not the combined PA3 report. All values below refer to a single fixed-seed run per method. Metrics and source files remain the authoritative evidence.

## Mathematical transfer and retention

See [the seven-condition table](benchmark_table.md), `benchmarks.csv` (including raw multiple-choice accuracy), and `../figures/benchmark_changes.pdf`. Scores are percentages and changes are percentage points.

All methods improve GSM8K over the shared 43.21% baseline. Full FT reaches 56.86% at midpoint and 63.46% at final. Its HellaSwag and BoolQ drops appear by midpoint (−1.54 and −5.02 pp) and remain at final (−1.21 and −4.74 pp), while ARC-Easy improves +7.28 pp. These tasks respond differently; the evidence does not support calling all non-math performance worse.

L2-SP improves GSM8K to 64.90%, 1.44 pp above Full FT, while recovering 4.37 pp on BoolQ relative to Full FT (64.13% versus 59.76%). It also recovers 0.45 pp on HellaSwag, with a 0.63 pp lower ARC-Easy score than Full FT. Its final total original-parameter displacement is 6.2661, versus Full FT 10.6215 (41.01% smaller); Full FT midpoint displacement is 9.9909. The recorded final pre-update penalty is Ω=19.6318, λΩ=0.01963 versus SFT loss 0.16453. The penalty and displacement concern slightly different instants (the last update's objective versus the final stored weights), so do not treat them as an exact arithmetic identity.

Increasing LoRA rank from 4 to 16 quadruples trainable adapter parameters (2,523,136 to 10,092,544) and improves GSM8K by 2.88 pp and BoolQ by 6.24 pp. HellaSwag changes +0.12 pp and ARC-Easy −0.21 pp. LoRA-16 and Full FT have similar math scores (63.08% versus 63.46%) but different retention: LoRA-16 is +1.36 pp higher on HellaSwag and +11.62 pp on BoolQ, with +0.42 pp on ARC-Easy. This is stronger evidence of a retention trade-off than a comparison where one model barely adapts.

OPLoRA-16 and LoRA-16 have matched rank and unprojected initial-factor hashes. OPLoRA gains +0.68 pp on GSM8K, but is −0.16 pp on HellaSwag, −0.34 pp on BoolQ and −0.13 pp on ARC-Easy relative to LoRA-16. These small differences show no clear general retention advantage from orthogonal protection in this run. No repeated-seed or paired significance analysis was performed, so avoid statistical superiority claims.

## Learning and regularization

`../figures/learning_curves.pdf` separates the five SFT curves from the L2-SP objective terms. The combined panel uses non-overlapping 25-update averages weighted by supervised token counts; raw losses remain in each training JSONL. The L2-SP panel uses trailing arithmetic means for each objective term. Full FT and L2-SP end below the adapters on SFT loss. Comparable training loss alone does not establish generalization or preservation. Because all runs use the same non-shuffled source order, short-range variation can also reflect successive data batches.

## Resource costs

| Method | Trainable parameters | Training hours | Peak allocated GiB | Four recovery states GiB |
|---|---:|---:|---:|---:|
| Full FT | 596,049,920 | 1.60 | 14.06 | 26.65 |
| L2-SP | 596,049,920 | 1.60 | 16.28 | 26.65 |
| LoRA-4 | 2,523,136 | 2.68 | 5.80 | 0.115 |
| LoRA-16 | 10,092,544 | 2.50 | 5.91 | 0.453 |
| OPLoRA-16 | 10,092,544 | 3.33 | 6.22 | 0.453 |

Training time is the recorded optimizer-loop sum, not billed session duration. Checkpoint-write overhead adds 110.21/112.40/1.05/1.94/1.97 seconds respectively. Model loading, evaluation and downloads are separate. The failed first LoRA-16 attempt is excluded from the successful-run total; its log is preserved for transparency. Allocated memory differs from reserved allocator memory, also recorded in `resources.csv`.

Adapters save memory and storage but are slower than Full FT in this implementation. OPLoRA takes 33.26% more optimizer-loop time than LoRA-16, with 308 MiB of basis storage and 37.25 seconds of measured basis setup. Its projected export is 38.62 MiB and takes 0.20 seconds; plain LoRA-16 export has the same size and takes 0.15 seconds. The recorded basis timer should not be generalized to other hardware, cached runs or SVD implementations. These are implementation/hardware-specific results, not a claim that LoRA is intrinsically slower.

The batch benchmark selected likelihood batch 16 and generation batch 64. Likelihood 32 and 64 ran out of memory on the fixed stress workload. Generation used a 64-token timing workload with a projected KV-cache headroom check for the actual 1024-token cap. The selected sizes are fastest among tested candidates on that workload, not proven universal optima.

## Geometry and activations

`diagnostics.csv` includes all three layers for baseline, Full FT midpoint and every final model; `../figures/diagnostics.pdf` compares the five final variants. Baseline η and R are zero; ρ is undefined for a zero update and is recorded as null.

OPLoRA's ρ₁₂₈ is 5.80e−14, 5.86e−14 and 1.22e−13 across layers 0, 13, 27, versus approximately 0.11–0.16 for the other methods. This verifies the measured output-subspace constraint numerically, rather than establishing improved benchmark retention. Relative update magnitudes are still nonzero: OPLoRA η is 0.01487, 0.01930, 0.00914. Orthogonal protection restricts direction without eliminating adaptation.

All trained corrections respond to HellaSwag probe inputs. For example, at layer 13, OPLoRA mean math R=0.09075 and HellaSwag R=0.02950; LoRA-16 gives 0.08529 and 0.02763. Protecting the leading singular subspaces does not make local corrections vanish on unrelated inputs. At layer 27, OPLoRA's HellaSwag mean R=0.01096 is lower than LoRA-16's 0.01288, but that local result is not mirrored by a better HellaSwag aggregate score. L2-SP generally reduces η and these local R means relative to Full FT at the inspected layers.

These diagnostics reuse 64 original-model activation inputs at the last prompt token, in FP32. They isolate local down-projection changes; they do not measure a full trained-model activation trajectory, all layers, or a causal mechanism for behavior.

## Extraction and qualitative behavior

GSM8K truncation rates are below 1% for every condition. Flexible extraction has no missing answers except one LoRA-16 example (1/1319). The stock strict-match metric is zero in all conditions; this is not the assigned primary score. The strict filter expects an exact phrase pattern; the saved strict score alone does not distinguish failed extraction from an incorrect extracted answer. Report flexible-extraction exact match, retain stock filters, and interpret extraction separately from correctness. The CPU Drive audit also recomputes these rates and exact scores from per-example saved outputs.

The fixed qualitative subset uses two predetermined underlying questions × three formats × seven conditions. Draft row-level annotations and excerpts are in `qualitative_annotations_draft.csv`; the rubric distinguishes strict format validity, relevance, repetition, unnecessary mathematical framing, inappropriate content and cap flags. They require student review.

Aligned illustrative examples were chosen from this required subset to show role confusion, repetition and format compliance; they were not selected as representative accuracy estimates:

- **Wish, plain:** baseline already repeats the user's offer. Full FT midpoint adds “The answer is: 1”; final adds “#### 1” and “The answer is: 1”. This supports mathematical answer-style intrusion, while recognizing that the baseline was already not answering the intended wish question.
- **Husband, plain:** baseline offers concise support. Full FT midpoint and final and L2-SP final become repetitive and reach the 512-token cap; all three adapter finals terminate without hitting it. Repetition, rather than merely response length, is visible in the saved text.
- **Husband, JSON:** baseline wraps an object in Markdown fences and thus fails strict whole-response JSON. LoRA-4 returns a bare valid object; Full FT/L2-SP emit prose plus an answer-prefixed object, LoRA-16 emits prose plus a fenced object, and OPLoRA ignores JSON entirely. OPLoRA's small weight-subspace interference therefore does not guarantee structured-output compliance.
- **Wish, template:** baseline leaves both placeholders unfilled. Full FT midpoint/final, L2-SP and both plain LoRA ranks fill the two requested slots; OPLoRA adds an extra final-answer line. Structured behavior can improve in one format while worsening in another.

No overt harmful recommendation was observed in these 42 responses. This limited subset cannot establish a general safety property or justify claims of emergent misalignment. Unhelpful role reversal, vague advice and occasional dismissive language remain relevant quality issues.

## Remaining submission work

Run the CPU Drive finalizer to verify and preserve per-example benchmark outputs and the original preparation manifest. Review the draft qualitative labels; preserve the original Colab notebook if available. Import the resulting folder into the public PA3 GitHub repository. Tasks 2/3 and the combined eight-page report remain separate work. No additional Task 1 GPU run is indicated by this small-file audit.
