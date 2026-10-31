# Task 2 required 196-token curriculum runner

This package adds the complete feature-cache, sampler, training/recovery, development and final-evaluation pipeline. It uses the previously tested `task2_vlm.py`, `task2_answers.py`, `task2_cpu_preflight.py`, and image-archive reader without modifying them. Do not change source files after a main run begins: saved identities bind the runner, input split/token files and feature cache to checkpoints.

## CPU preparation

Run the ten checks in `test_task2_runner_cpu.py`, then:

```bash
python task2_run.py --task-dir TASK2 --prepare
```

This verifies saved split identities, hashes the original Qwen snapshot, formats all 37,984 selected caption/QA records and stores text IDs in `data/tokenized_examples.json`. It also stores calibration text IDs from all three captions and questions of the fixed 128 training images (384 training caption examples). No model is loaded or trained and no GPU is used. Illustrative complementary pairs are also fixed now: first sorted pair ID per question type, excluding grounding pairs flagged during the pre-training label review; no predictions enter this selection. The locked recipe and package/input/source identities are saved in `configs/training_recipe.json` and `results/runner_cpu_preflight.json`.

## GPU queue (only after CPU checks)

Select an A100, restore the recorded CPU package versions while preserving the CUDA torch wheel, and run `task2_run.py --task-dir TASK2 --run-all`. The notebook's outer `finally: runtime.unassign()` releases Colab on completion/error; the command-line runner itself does not disconnect a runtime.

Each job is a fresh subprocess:

1. **Frozen feature cache.** Stage the 26 image archives locally. Choose the fastest tested SigLIP inference batch among 32/64/128 with 80% allocated-memory headroom, using one training image repeated for timing, two timed forwards per candidate. No correctness labels enter selection. Run frozen FP16 SigLIP without pooling, verify a single final LayerNorm, and cache FP16 196×768 features in 26 resumable NPY shards. Image preprocessing is the CPU-verified deterministic PIL RGB/bicubic/normalize rule. Save per-shard hashes/timings and `results/feature_caching.json`; remove SigLIP by exiting its subprocess before loading Qwen.
2. **Real-model GPU smoke.** Use a separate temporary model, two updates in Stage 1 and two in Stage 2 with the required microbatch 8/accumulation 2, then a short greedy-generation check. This does not advance the main curriculum or write main development results. Save stage timings/memory and `results/gpu_smoke_task2.json`. The main run is gated on this flag and matching identities.
3. **Main training.** Start from the original frozen FP16 Qwen and a seeded FP32 connector, calibrate once on training data only, and continue the same connector through all three stages. Stage 1: 2,250 updates, two seeded shuffled passes over 18,000 captions. Stage 2: 1,500 updates, independent per-draw probabilities .50 presence/.25 counting/.25 captions. Stage 3: 750 updates, .70 natural/.10 presence/.10 counting/.10 captions. Each draw contains a complete pair or two independently sampled captions. Sampling uses replacement in Stages 2/3 and does not enforce exact mixture quotas.
4. **Final evaluation.** Evaluate only the completed Stage 3 model, with greedy decoding, cap 16, batch size 8, and correct/complementary/mismatched/no-image inputs on both full final subsets.

Generation batch size and training batches are prescribed and are not tuned. Only the frozen feature-cache batch is timed for throughput.

## Optimization choices

Stage connector peak LRs: 1e-3, 1e-4, 5e-5. Stage 2/3 attention-only LoRA peak LRs: 2e-5, 1e-5; rank 16/alpha 32/dropout .05/q,k,v,o/no trainable original biases. The connector has 1,837,056 trainable parameters; attention adapters add 4,587,520.

Fresh AdamW and scheduler at each stage; betas .9/.999, epsilon 1e-8, weight decay .01 on all trainable parameters including connector biases. Carry the dynamic gradient scaler between stages. Per-stage ceil(5%) warmup then cosine decay to 10% peak; conventional LambdaLR starts at zero. Original backbone parameters stay frozen, trainable tensors/optimizer states stay FP32, computation uses FP16 autocast, no quantization, non-reentrant gradient checkpointing, unscale then global norm clip 1.0. Loss means tokens within an example, then examples within the effective batch; it does not use Task 1's global token normalization.

AMP overflow attempts are logged and retried on the same already-sampled 16 examples. Only successful updates advance schedule/curriculum progress. Twenty consecutive overflows abort. Retried attempts add physical compute and are counted separately from the prescribed 72,000 successful example presentations.

## Recovery and identities

Save complete model-connector/adapter, optimizer, scheduler, scaler, global RNG, sampler RNG/decks/positions/counters, stage/update and resource state at every 250 global updates: 18 recovery states. Original backbone weights are not duplicated. Hash each state before writing a completion marker; ignore unmarked saves. Export connector/adapter weights at 2,250, 3,750 and 4,500. On resume, preserve abandoned log tails in a backup and replay only work after the last complete state. Stage transitions reset optimizer/scheduler exactly as above. Trusted local recovery files use `weights_only=False` for Python/NumPy RNG; inference exports use tensor/basic-data loading.

Completed jobs skip and completed feature shards are verified/reused. Dataset and code identity changes abort rather than silently mixing experiments. Stage optimizer-loop time and allocated/reserved GPU peaks are distinct from loading, cache creation, development evaluation, checkpoint writes and failed-session costs. Feature-cache timings distinguish this-session work from sums across completed shards. Main training logs buffer 25 updates before a Drive write, and flush before every recovery state, reducing Drive I/O.

## Development and final metrics

Development runs at update 0; every 500 main updates; and 2,250/3,750/4,500. Save caption-reference loss (including EOT, correct image only), and per-example answer-content NLL (excluding termination) on both VQA development suites under correct/complementary/no-image inputs. Image-swap penalty is NLL(partner) − NLL(correct). No development-generated accuracy or final-test score selects an endpoint; the fixed 4,500-update Stage 3 export is final.

Final scoring preserves original references under all controls and includes empty outputs in denominators. Grounding: normalized EM, presence/counting separately. Natural: local leave-one-out VQA soft consensus against all ten human answers, yes/no/number/other separately. Also save canonical pair accuracy (both original canonical answers correct), identical-answer rate (including equal empties), empty rate, answer-content NLL, amortized batched generation time, and partner-canonical-answer following. Pair-bootstrap CIs use 4,000 complete-pair draws, the same sampled pair IDs across image conditions, with 95% percentile intervals. Correct-image accuracy advantage is correct − complementary; NLL advantage is complementary − correct. Raw fractions, not percentages, are stored in JSON summaries.

Final records include raw questions/references/responses, original/used image IDs, original/partner canonical answers, token IDs, EOS/cap flags, NLL, prompt text-token spans and per-example scores. One atomic file per completed condition allows final evaluation to skip completed conditions after an interruption. A failed partial condition is rerun in full. Generation timing includes feature lookup/transfer, prompt construction, cached autoregressive decoding and response decoding; vision encoding is counted separately. Batch times are amortized over answers, not claimed as individual latency measurements.

## CPU validation

Ten checks passed: two-pass caption coverage; stochastic pair integrity and exact sampler resume; varying mixtures without exact batch quotas; trainable-state/AdamW/scheduler/RNG recovery with dropout; incomplete-state exclusion; feature lookup mapping; cached decoding versus full-prefix and single-example decoding; independent EOS termination; control/reference preservation; and common-index complete-pair bootstrap with empty answers.

Token formatting was also exercised on all 37,984 actual selected records using the pinned Qwen tokenizer. Real GPU AMP/smoke and complete cached-feature checks still occur on the A100 before the main run. This CPU package does not claim the main training/evaluations have already run.
