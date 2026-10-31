# Task 3 live generation runner

## CPU first

Run the runner CPU tests, then `python code/task3_live_run.py --task-dir PATH --prepare` on CPU. This verifies the pinned assets/input identities, creates the fixed 250-call plan and freezes the new runner source hashes. It does not instantiate the 3B model. Unit tests include a tiny randomly initialized Qwen2 model on CPU; these fixture outputs are never assignment results.

## One GPU queue

After CPU preparation passes, select an A100. `task3_live_gpu_setup.py --task-dir PATH` pins the recorded packages, preserving the installed CUDA torch build. `task3_live_run.py --task-dir PATH --run-all` verifies/stages the model once, loads original weights explicitly in FP16 with SDPA, freezes parameters and uses batch 1 throughout. No quantization, optimizer, fine-tuned checkpoint, grader feedback or final-answer-based selection enters generation.

The checkpoint's native repetition penalty is 1.05; the runner explicitly overrides it to the prescribed 1.0. Temperature .7, top-p .8, top-k 20, sampling, one beam and one return sequence are explicit. The original tokenizer/chat template and model-configured EOS tokens are used. Base prompts/IDs must match CPU preparation exactly. The model's native BF16 storage is cast to FP16 for this controlled live experiment.

Per question, generate parallel_0/1/2 at 1024 new tokens each, one single_long at 3072, and one revision at 2048. Parallel alternatives are sampled independently but executed serially to obey the required batch size 1. The revision reuses parallel_0's response verbatim as an assistant message, followed by the prescribed critique/revision instruction. Whitespace and all nontermination response content are retained. Each call uses its independently derived fixed seed. No gold reference file is opened by the GPU queue. The first parallel response is physically generated once and its cost is charged to both compared strategies. There are 250 physical calls, 150 strategy/question comparisons, and a maximum of 409600 physically generated output tokens.

A fixed non-test arithmetic prompt supplies separate warmup/smoke calls, with at most 16 and 64 output tokens. The smoke checks real FP16 inference, cache use and frozen parameters and reports throughput/memory. It is not an evaluation result and does not tune the mandated batch size. Each new loading session has its own smoke, excluded from main-call timing.

## Timing, output and resume

CUDA-synchronized per-call timing includes prompt construction/tokenization, input transfer, generation, output decoding and balanced-box extraction. It excludes model loading, smoke/warmup and disk writes; those are recorded separately. Revision timing includes its longer prompt and full critique/revision generation. The output cap and actual output token count include final EOS; retained token IDs/counts exclude the final EOS and are also saved. Ending with EOS exactly at the cap is distinguished from truncation. Empty/missing extractions remain saved.

Each completed response is atomically saved under `results/live/calls/` with a complete marker/hash. Requests/messages, IDs, seeds, decoding settings, raw text, extraction, cap/EOS flags, time, memory and the revision parent hash are retained. Interrupted calls without a completed marker rerun with the same seed; valid completed calls are reused. Changed identities or corrupt marked calls fail rather than silently generate a replacement. A resume requires the same pinned packages and GPU environment. No exact bitwise agreement across different CUDA/software/hardware environments is claimed.

The finished queue writes `results/live/finished.json`, the per-call manifest and per-strategy cost records, then returns. The outer Colab cell must call `runtime.unassign()` in `finally`, including on error. The runner itself never grades answers on GPU. Once the queue completes, switch to CPU for mathematical grading, three-way accuracy/cost comparisons, extraction/truncation rates, revision transitions, final plots, clean notebook and repository organization. Model weights, prepared arrays and caches stay outside git.
