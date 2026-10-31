# Evaluation batch benchmark patch

Apply this patch before Cell 14 and before any main evaluation has started.
It updates only the evaluation runner, adds a timing script/tests, and updates
the package manifest. Training microbatch 2 and accumulation 16 are untouched.

`python task1_batch_benchmark.py --task-dir TASK1 --prepare` constructs workload
inputs on CPU. This step can also run on a connected GPU runtime: it performs
only input preparation and does not load a model or launch GPU kernels.

`python task1_batch_benchmark.py --task-dir TASK1` requires the GPU. It loads
the original FP16 model from its existing local snapshot and compares batches
8, 16, 32, 64 separately for candidate likelihood and greedy generation. Each
size has a small warmup and two measured repetitions on the same requests.
Completed timings are persisted. No timing generations become reported results.

The fixed workload uses 16 contexts from each likelihood task and 64 GSM8K
questions, mixing long input contexts and seed-6304 random choices. Length is
measured in characters for selecting the stress contexts. Gold labels and answer
accuracy never choose the workload or the winning batch.

Generation timing uses a 64-token cap to keep calibration short. Its memory
screen adds estimated FP16 KV storage for 960 further output tokens, then
requires the projected peak to remain at most 85% of GPU memory. Likelihood
also requires 15% memory headroom. OOM cases are excluded. Among eligible sizes,
lowest mean measured wall time wins; exact ties choose the smaller size.
This is the fastest tested size on these representative/stress inputs, not a
guarantee of globally optimal speed or immunity to every possible OOM.

The selected sizes are saved in configs/evaluation_config.json and results/
evaluation_batch_benchmark.json. The updated evaluator reads the appropriate
size for each benchmark. All seven reported conditions reuse those settings.
The main queue and Cell 14 remain unchanged. If a prior tiny smoke evaluation
exists and settings change, it is renamed and preserved before rerunning smoke.
The script refuses to change batches once main evaluation identities exist.

Apply this patch after the original evaluation package. Do not re-extract the
original evaluation ZIP afterward, because it would replace the updated runner.
The saved manifest checks the combined original-plus-patch source files.

The GPU benchmark should be run with Cell 13 already completed, then followed
by the existing Cell 14 (GPU smoke, full queue, automatic runtime release).
You do not need to wait for ChatGPT confirmation once calibration succeeds.
