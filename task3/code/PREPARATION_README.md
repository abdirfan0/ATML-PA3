# Task 3 CPU token audit and prepared pools

Run after scorer v2 has passed and been frozen. No model, image, GPU, or held-out accuracy calculation is involved.

The script verifies the frozen scorer/environment and exact saved 200/800/50 split permutation. It creates ten seeded permutations of the 32 rollout IDs for each question, using a separate NumPy default_rng(6304) for each dataset, iterating questions lexicographically. These orders are saved once and must be reused across methods, candidate budgets and adaptive stopping; all budgets use nested prefixes. They do not depend on answers or gold labels.

Each source parquet is copied to local CPU storage while checking its full SHA256 against Cell 2's inventory. Completed local copies are reused only after hash validation. Arrow tables are joined using `(dataset, sample_id, rollout_id)`, never by row position. One dataset is handled at a time. Every list is checked against the retained generated-token count; the attention file's `n_generated_text_tokens` is checked as the preceding-count sequence, not confused with the generations file's scalar length. Probability/attention arrays must be finite and have valid ranges. Eligible attention groups are normalized by their sum, without dividing by group sizes. The first generated-text attention must be zero.

MATH's supplied extractions are cross-checked against the README's last-literal-box balanced-brace rule. Exact export totals must match: MATH 20,742,094 tokens / 523 capped / 521 missing; MathVista 8,961,703 tokens / 487 capped / 3,032 missing. No rollout is dropped or replaced. Zero-token rollouts, if any, are retained and reported; undefined per-rollout means are represented as JSON null rather than silently fabricated.

Prepared caches retain raw text/answers, integer token IDs, FP32 logp/KL, and FP32 normalized visual attention. Fifty questions share each compressed NPZ, using packed arrays and offsets. Metadata is UTF-8 bytes, never pickled Python objects. Per-question means use FP64 reductions. These are input features, not correctness or quality-selection results. Prefix-specific VisConf percentiles must still be recomputed later from only revealed KL tokens, not these mean summaries. The cache contains gold/reference fields for later grading, but methods must not consume them for selection or stopping. MATH equivalence groups must be formed from the revealed prefix using the frozen scorer.

The complete dataset cache and its file hashes can be reused on repeated runs. An interrupted partial dataset is audited again. Original data is untouched. Raw parquets and prepared arrays belong outside git. Full manifest/identity records and fixed split/order IDs belong in the eventual repository. Local working copies can be discarded after successful CPU preparation; they can be recreated from the source.

`splits/live_math_questions.json` contains only the 50 fixed question IDs, original problems and original row indices; it deliberately omits gold answers from the future generation input. `splits/trajectory_examples.json` fixes three held-out questions and two rollout IDs per dataset before looking at quality or correctness for illustrative signal trajectories. These illustrative IDs do not select policies or checkpoints.

Commands:

```bash
python -m unittest discover -s code -p test_task3_prepare_cpu.py -v
python code/task3_prepare.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
```

Preparation adds its own frozen identity and does not edit scorer v2 or existing question splits. Next stages compute mathematical equivalence/correctness, offline method comparisons, development-only stopping-policy selection, and held-out analysis. GPU allocation remains deferred until the live inference runner and assets are ready.
