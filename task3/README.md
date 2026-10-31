# Task 3: inference-time compute allocation

Author: Abdullah Irfan. ATML PA3. Analysis seed 6304.

## Recorded experiments

- Offline: 32 supplied candidates for 1,000 MATH questions and 1,000 MathVista questions. Each dataset has 200 development and 800 held-out questions, with ten saved nested candidate orders per question.
- Live: fixed first 50 held-out MATH questions; pinned original Qwen2.5-3B-Instruct, FP16, no quantization, batch 1; long response, three-sample voting, and initial-plus-revision strategies each have 3,072 output tokens of allowance.
- All required GPU generation is complete. Default notebook execution uses saved results on CPU.

## Read or rerun the CPU report

Open `Task3.ipynb` and run all cells on CPU. It installs the frozen scorer/analysis dependencies if needed, verifies saved evidence, regenerates figures/tables, and displays the results and examples. It contains no upload, debugging or Git publishing cells. GPU generation is disabled by default.

From this folder:

```bash
python -m pip install -r requirements-analysis.txt
python code/task3_analyze.py --task-dir .
```

`analysis/RESULTS.md` gives the compact result tables, paired differences and interpretation. `analysis/QUALITATIVE.md` aligns selection disagreements, recoverable premature stops and revision examples. `analysis/audit.json` states exactly which checks were rerun and their limits.

## Files

- `code/`: frozen scorer, data preparation, selectors, offline analysis, live generation, CPU reporting and tests.
- `configs/`: pinned source/environment/asset identities, full development grid, frozen thresholds, GPU environment.
- `splits/`: question IDs, all candidate orders, fixed live IDs and trajectory IDs.
- `results/offline/`: derived per-question candidates/decisions, summaries, paired intervals and completion hashes.
- `results/live/`: all 250 prompts/responses/token IDs/seeds/caps, cost records and measured GPU session information.
- `results/live_scoring/`: per-question live grading, 150 strategy rows, paired intervals and revision examples.
- `data/live_prompts.json`, `data/live_math_references.json`: saved live experiment inputs and separately verified references. References were not loaded during generation.
- `figures/`, `analysis/`: reproducible PNG figures, CSV tables, narrative and aligned examples.

Raw supplied parquet, prepared NPZ arrays, full MathVista metadata, model weights and caches remain outside Git. Derived question-level records are retained to reproduce the analysis without inference. Existing Task 1/2 folders are not changed by the finalizer.

## Reproduce from supplied data

The original Colab experiment lives at `/content/drive/MyDrive/ATML/PA3/task3`. The frozen inventory contains these absolute original paths; use that layout for an exact replay. Place or directly open the shared ATML-PA3-Data shortcut under `data/raw/ATML-PA3-Data` with the `qwen25-math` and `qwen25vl-mathvista` subfolders. Start traversal inside the shortcut: Drive shortcuts may not be traversed by rglob from a parent folder. Verify supplied file hashes against `configs/supplied_data_inventory.json`. The original data source is linked in the assignment manual.

Restore the pinned MathVista metadata and source script described in `code/SCORER_README.md` under `assets/mathvista/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb/`; restore the supplied README to `assets/supplied_README.MD`. `code/task3_restore_assets.py` fetches the two pinned public files and verifies the frozen hashes, and copies the supplied README from the shared folder. It never downloads models. Keep all saved frozen configurations and split files unchanged.

```bash
python code/task3_restore_assets.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
python code/task3_prepare.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
python code/task3_offline.py --task-dir /content/drive/MyDrive/ATML/PA3/task3 --workers 2
python code/task3_live_assets.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
python code/task3_live_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task3 --prepare
```

These are CPU steps. `task3_prepare.py` performs full supplied-data alignment checks and constructs the prepared cache. `task3_offline.py` resumes verified derived records, selects thresholds from development questions and checks the existing freeze before held-out summaries. For a full recomputation rather than resume, copy the experiment to a separate location while preserving recorded inputs/configuration and clear only its derived offline chunk outputs. Do not edit original completed evidence.

Only after CPU preparation, select the recorded GPU environment and run:

```bash
python code/task3_live_gpu_setup.py --task-dir /content/drive/MyDrive/ATML/PA3/task3
python code/task3_live_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task3 --run-all
```

The GPU runner skips valid completed calls and resumes interrupted queues. The optional notebook cell wraps this in `runtime.unassign()` on success or failure. Existing frozen GPU identities intentionally reject environment drift. Fresh hardware/software experiments require separate identities and must not be mixed into these results. For a clean independent live run, use a separate experiment copy and retain the original call evidence.

## Scoring and interpretation

Scorer version `pa3-task3-answer-scorer-v2`; math-verify 0.9.0, SymPy 1.14.0. MATH equivalence is symmetric and strict; MathVista uses pinned released type/choice/precision rules, with exact reference choice text decoded separately. Missing answers remain in votes/cost/denominators and are always incorrect. Gold and unrevealed candidates do not influence selection or stopping. Ties use earliest supporting candidates. VisConf calibration is recomputed from each revealed prefix. All intervals bootstrap questions, not rollouts.

Offline retained-token cost omits prefill, attention collection and scoring overhead. Live output counts include termination tokens, with retained counts separately saved; shared initial cost and full revision input/output are charged to both relevant strategies. Parallel samples were executed serially at batch one, so the recorded latency is the sum of three calls. Fifty live questions give limited statistical precision. Differences between the two offline model–dataset combinations do not isolate the causal effect of vision.

See `ATTRIBUTION.md` for external sources. The combined PA3 report is prepared separately after code/results organization.
