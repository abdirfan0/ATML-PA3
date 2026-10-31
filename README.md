# ATML PA3

Task 1 compares Full FT, L2-SP, LoRA-4, LoRA-16 and OPLoRA-16 on Qwen3-0.6B. Five training runs and all seven required evaluation conditions have finished. Tasks 2 and 3 are not included yet.

See [Task 1](task1/README.md) for the experiment recipe, reproduction commands, output layout and evidence limits; [results](task1/analysis/RESULTS.md) for the analysis; and [benchmark table](task1/analysis/benchmark_table.md) for scores and percentage-point changes.

This folder is ready to import into the assignment's public GitHub repository after the CPU finalization step and student review. It has not been published to GitHub. Model weights, raw dataset snapshots, activation tensors and request caches remain in the original Drive experiment folder. The finalizer adds compressed per-example benchmark outputs and the original data manifest without copying large tensors.

The assignment requires an individual submission and one combined eight-page NeurIPS report across all three tasks. The analysis notes and qualitative labels here are assistant drafts for the student to review, revise and disclose as appropriate under their course rules; they are not a finished submission report.
