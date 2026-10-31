# PA3 Task 1 evaluation and queue extension

Place these files beside `task1_train.py` and `test_cpu.py` in the existing
`task1/code/runner_v1` folder. This extension does not modify that trainer.
Use lm_eval[hf]==0.4.13. Preserve the original core package versions recorded
in `task1/configs/gpu_environment.json` for GPU launch.

## CPU preparation

Run `python -m unittest discover -s CODE -p test_eval_cpu.py -v`.
Seven tests cover stock task construction, arithmetic diagnostics, truncation
accounting, factor merging, interrupted-cache recovery, tiny randomly initialized
Qwen generation/caching, and an end-to-end harness generation/likelihood check.
They complement the original eight training/recovery tests, not the real-model
CUDA smoke test.

Run `python task1_eval.py --task-dir TASK1 --prepare` on CPU. It pins benchmark
dataset revisions, downloads the datasets, checks the complete split sizes, saves
raw datasets as local Arrow snapshots on Drive, and constructs the exact 64
diagnostic prompts using stock harness task formatting. The real GPU evaluation
uses these snapshots with HF Hub/datasets offline mode; there are no benchmark
downloads on GPU. The saved package fingerprint locks the harness source files.

`python task1_run.py --task-dir TASK1 --plan` is CPU-safe and lists pending work.

## GPU smoke test, then full queue

Once CPU preparation passes and an A100 is connected with the original package
versions, run `python task1_run.py --task-dir TASK1 --smoke`. This performs two
Full FT updates, two LoRA-16 updates, then tiny baseline benchmark/qualitative
checks. Real-model FP32 baseline diagnostic activations and three exact SVDs are
also cached once. Smoke benchmark scores are saved separately from final scores.
Review the smoke outputs and update timing before leaving an overnight run.

After smoke checks pass, run `python task1_run.py --task-dir TASK1 --run-all`:

1. Shared unchanged-model baseline evaluation.
2. Full FT training, then evaluations of its fixed midpoint and final snapshots.
3. L2-SP training and final evaluation.
4. LoRA-4 training and final evaluation.
5. LoRA-16 training and final evaluation.
6. OPLoRA-16 basis preprocessing, training, and final evaluation.

Each job runs in a fresh subprocess, releasing the previous model's GPU memory.
The queue stops on an error. Repeating it skips completed work and resumes saved
training checkpoints and partial/completed evaluations. It is not an external
scheduler: the Colab runtime must remain alive; no fixed wall-clock duration is
promised. Closing a tab or leaving the laptop running does not itself guarantee
that Colab keeps executing. Recovery is from the last saved complete state.

## Evaluation protocol

- Complete GSM8K test (1,319), HellaSwag validation (10,042), BoolQ validation
  (3,270), ARC-Easy test (2,376), all zero-shot, fixed lm_eval source fingerprint.
- GSM8K stock gsm8k_cot prompts/stops/filters, non-thinking chat, greedy output,
  cap 1,024. Flexible-extraction exact match is primary. Raw generation tokens,
  extraction outputs, missing extractions, and cap/termination flags are saved.
- Other tasks use stock completion prompts and candidate likelihood without
  chat formatting. Both accuracy and normalized accuracy are retained when
  available. FP16 model weights, FP32 likelihood softmax, batch size 8.
- Full FT uses the fixed trained weight snapshot. Adapters are exported from the
  trainer's custom format, with OPLoRA factors already projected, and merged into
  fresh original backbones. FP32 merged/unmerged logits on a small batch must
  agree before the merged model is converted to FP16 for benchmarks.
- JSONL request caching uses the harness's own request-key algorithm and partial
  result hooks. Completed tasks and qualitative responses are saved separately.
  No SQLite database is written onto the Drive FUSE mount.
- One greedy, non-thinking, 512-token response per each of 24 supplied prompts
  for seven conditions (168 total). JSON system messages and template wording
  are preserved. The fixed first/last underlying questions are marked for later
  human annotation (42 responses across conditions). The YAML's GPT judge and
  100-sample settings are unused; no paid judge/API calls occur.

## Diagnostics and remaining work

Baseline down_proj activations at the final prompt token are cached from an
FP32 original-model transformer forward pass for the fixed 32+32 inputs.
Layers 0, 13, and 27 use exact FP32 starting-weight singular bases (k=128).
Weight corrections and the same cached inputs produce rho_k, eta, and per-input
activation ratios in FP32. Baseline is the zero-update control; six trained
conditions have the prescribed diagnostics. Full-model displacement is recorded
by the trainer. Local measurements do not capture all changes in the network.

After the queue, work remains: inspect/annotate qualitative outputs, aggregate
tables and plots, interpret results, write the report, and organize the repository.
Every GPU benchmark result remains unverified until the real-model smoke/full
runs complete. CPU tests use a tiny random Qwen, not the assignment checkpoint.

## Sources and assistance

Original orchestration, logging, merge, cache, and diagnostic glue were created
with ChatGPT coding assistance. Stock prompts, filters, metrics, and the HFLM
backend are used from EleutherAI's lm-evaluation-harness (0.4.13):
https://github.com/EleutherAI/lm-evaluation-harness . The tiny-Qwen CPU tests use
Hugging Face Transformers and Tokenizers. Training equations and other sources
are attributed in the original `README.md`. No external implementation was
copied into these extension files.
