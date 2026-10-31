# Task 3 live scoring

CPU only. No torch import, model download, model loading, or forward pass.
Run task3_live_score.py --task-dir /content/drive/MyDrive/ATML/PA3/task3.

First validates the frozen runner, asset, scorer, prompt, reference and split identities; all 250 call hashes and completion markers; seeds, sample IDs, FP16 model revision, sampling settings, token caps, EOS accounting, literal final-box extractions, original prompts, verbatim revision inputs, shared initial hashes, and all recorded strategy costs. Generation identity is preserved.

Uses the frozen math-verify scorer. Parallel selection groups only the three predictions and chooses the most frequent group, breaking ties by earliest sample. Missing predictions retain their votes and remain incorrect. Sequential always selects the revision, regardless of correctness. All selections precede gold grading.

Saves per-question results, a 150-row strategy CSV, summary with 1,000 paired question-bootstrap percentile intervals (seed 6304), revision transition counts and up to two lexicographically first examples per transition. Cost includes complete generated output including final EOS, full revision input and critique, and the shared initial call in both strategies. Parallel samples were executed serially at batch size one; report the sum of their runtimes. Peak memory is the maximum allocated memory across calls, not a sum. Generation timing excludes loading, smoke and disk writes.

Outputs are under results/live_scoring. finished.json hashes all output files. No input, frozen config or existing generation file is changed. Figures, final notebook and repository organization follow this stage.
