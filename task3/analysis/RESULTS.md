# Task 3 results

Held-out evaluation: 800 questions per dataset, ten nested candidate orders per question. Intervals bootstrap questions after averaging orders (1,000 paired resamples, seed 6304). The two model–dataset combinations do not isolate a causal effect of vision.

## Fixed budget: N = 32

| Dataset | Method | Accuracy (%) | 95% CI | Δ vs SC (pp) | Δ 95% CI (pp) | Token cost |
| --- | --- | --- | --- | --- | --- | --- |
| MATH · Qwen2.5-3B | FirstAns | 64.350 | [61.34, 66.89] | -9.438 | [-11.101, -7.849] | 669.8 |
| MATH · Qwen2.5-3B | SC | 73.787 | [70.42, 76.86] | +0.000 | [0.0, 0.0] | 21347.0 |
| MATH · Qwen2.5-3B | LogPBest | 64.000 | [60.50, 67.25] | -9.788 | [-12.412, -7.2] | 21347.0 |
| MATH · Qwen2.5-3B | SelfCertBest | 70.625 | [67.37, 73.75] | -3.162 | [-5.563, -0.837] | 21347.0 |
| MATH · Qwen2.5-3B | SelfCertRWC | 75.550 | [72.17, 78.38] | +1.762 | [0.675, 2.801] | 21347.0 |
| MATH · Qwen2.5-3B | Shortest | 62.375 | [58.88, 65.71] | -11.413 | [-14.088, -8.962] | 21347.0 |
| MathVista · Qwen2.5-VL-3B | FirstAns | 53.600 | [51.22, 56.11] | -8.075 | [-10.025, -6.162] | 277.8 |
| MathVista · Qwen2.5-VL-3B | SC | 61.675 | [58.37, 64.89] | +0.000 | [0.0, 0.0] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | LogPBest | 48.125 | [44.75, 51.62] | -13.550 | [-17.0, -10.262] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | SelfCertBest | 54.625 | [51.25, 58.25] | -7.050 | [-10.151, -3.8] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | SelfCertRWC | 62.500 | [59.25, 65.88] | +0.825 | [-0.55, 2.138] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | Shortest | 53.850 | [50.54, 57.03] | -7.825 | [-11.027, -4.923] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | VisionBest | 53.625 | [50.12, 57.12] | -8.050 | [-11.337, -4.95] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | VisionRWC | 62.875 | [59.75, 66.12] | +1.200 | [-0.312, 2.838] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | VisConfBest | 57.750 | [54.50, 61.38] | -3.925 | [-6.976, -0.896] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | VisConfRWC | 63.125 | [59.88, 66.62] | +1.450 | [0.037, 2.975] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | TokenwiseBest | 56.250 | [53.12, 59.63] | -5.425 | [-8.425, -2.362] | 8897.8 |
| MathVista · Qwen2.5-VL-3B | TokenwiseRWC | 62.625 | [59.38, 66.13] | +0.950 | [-0.413, 2.438] | 8897.8 |

Self-certainty RWC reaches 75.55% on MATH versus 73.7875% for SC. MathVista VisConf RWC reaches 63.125% versus 61.675% for SC. These compare selectors on identical revealed candidate sets. Oracle coverage is a retrospective upper bound, not an implementable selector. Shortest-rollout control and score–length associations prevent treating every quality gain as evidence of calibrated correctness.

## Adaptive sampling

| Dataset | Controller | τ | γ | Accuracy (%) | Mean token cost | Mean rollouts | Tokens saved vs SC32 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| math | agreement | 0.625 | 0.0 | 73.638 | 10140.0 | 11.824 | 52.50 |
| math | quality | 0.5 | 0.0 | 73.175 | 7663.1 | 9.008 | 64.10 |
| mathvista | agreement | 0.5 | 0.0 | 61.262 | 3290.3 | 9.891 | 63.02 |
| mathvista | quality | 0.5 | 0.0 | 61.562 | 3613.0 | 10.930 | 59.39 |

Thresholds were selected on 200 development questions per dataset and frozen before held-out summaries. Both controllers return current SC. The 1 pp tolerance constrains development accuracy, not a guaranteed held-out loss. Compare the full fixed-SC cost curve in the figure; nearest-cost rows in matched_cost.csv are approximate observed comparisons, not exactly equal-cost trials. Incorrect and recoverable early-stop rates use all question-order outcomes as denominator. Full-pool correctness buckets are retrospective diagnostics only.

## Signal and fusion findings

MathVista calibrated confidence and visual engagement have mean within-rollout correlation −0.20934 (rollouts averaged within each question). Tokenwise fusion minus the rollout product averages −0.00224338, exactly matching the saved population covariance. This negative relationship makes the two fusion definitions different. The fixed six MathVista trajectories were selected before metrics; their confidence uses the complete N=32 set, appropriate only for this retrospective display. Attention associations do not prove causal grounding.

## Live comparison

| Strategy | Correct / 50 | Accuracy (%) | Input tokens | Output tokens | Time (s) | Selected truncation (%) | Extraction failure (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single long | 30 | 60.0 | 108.38 | 631.80 | 24.32 | 0.0 | 0.0 |
| Parallel voting | 30 | 60.0 | 325.14 | 1840.10 | 70.83 | 18.0 | 18.0 |
| Sequential revision | 29 | 58.0 | 856.48 | 1176.68 | 45.29 | 2.0 | 2.0 |

All three have the same 3,072-token output allowance; actual use differs. Output cost includes EOS; retained-token counts are separate in the CSV. Parallel samples were executed serially, so runtime is summed; the shared initial call is charged to both parallel and sequential. Sequential includes all critique tokens and the longer input prompt. Loading, smoke and disk writes are outside inference timing. The shared initial answer was correct on 28/50: revision corrected one error, spoiled none, and retained 28 correct answers. Long/parallel each score 30/50; revision scores 29/50. A 50-question comparison supports limited precision rather than a universal strategy ranking.

| Paired accuracy difference | Mean (pp) | 95% CI (pp) |
| --- | --- | --- |
| parallel_minus_single_long | +0.00 | [-12.0, 12.0] |
| sequential_minus_single_long | -2.00 | [-14.0, 10.0] |
| sequential_minus_parallel | -2.00 | [-10.0, 4.0] |

## Audit scope and limitations

All stored chunk and live-call hashes, split/order IDs, selection membership, retained-token costs, development grid selection, fixed/adaptive metrics, and saved confidence intervals are rechecked. Live parsing, grading, revision messages, seeds, limits and costs are independently rechecked on CPU. This report export excludes raw parquet and prepared NPZ arrays; it cannot rerun the original full-distribution token alignment or recompute every VisConf value from raw arrays. Original preparation audits and hashes are retained. Model tensor bytes and GPU timings are recorded evidence, not re-measured by CPU report mode. No gold/future candidates enter selection or stopping.
