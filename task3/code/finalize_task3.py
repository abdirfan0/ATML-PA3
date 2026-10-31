"""Audit Task 3 on CPU and copy only its source/results into the PA3 repo folder."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
from task3_analyze import analyze,read,write,sha

README='''# Task 3: inference-time compute allocation

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
'''
ATTRIBUTION='''# Sources and attribution

- Assignment-supplied ATML-PA3-Data and README: original frozen parquet hashes, generation settings, token measurements and field definitions are retained in `configs/supplied_data_inventory.json` and preparation receipts. Raw exports are not redistributed here.
- MATH: DigitalLearningGmbH/MATH-lighteval, revision `0530c78699ea5e8eb5530600900e1f328b48acad`. Original test problems and solution-derived boxed references were checked for the fixed live subset.
- Live model: Qwen/Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`. Model weights remain external; hashes and model/license file identity are in `configs/live_assets.json`.
- MathVista released metadata and answer normalization: https://github.com/lupantech/MathVista/tree/c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb . Prediction normalization in `code/task3_answers.py` adapts the released `evaluation/calculate_score.py` rules; exact reference decoding is separated to handle literal parenthesized choice text. The source repository's CC BY-SA 4.0 license and original source are retained with the external metadata assets.
- Math-Verify: https://github.com/huggingface/Math-Verify . Parser and verifier are called through their public API; dependencies are pinned.
- Methods: Kang et al., Scalable Best-of-N Selection for Large Language Models via Self-Certainty; Usman et al., VisConf: Quality-Aware Consensus for Best-of-N in Vision-Language Model Reasoning; Wang et al., Self-Consistency Improves Chain of Thought Reasoning in Language Models; Madaan et al., Self-Refine: Iterative Refinement with Self-Feedback.
- Software: NumPy, SciPy, pandas, PyArrow, Matplotlib, PyTorch, Hugging Face Transformers/Datasets/Hub/Tokenizers/Accelerate, and Safetensors. Versions are saved in the frozen configurations.

The implementation, experiment packaging, analysis and notebook were developed with ChatGPT assistance. The author should review the final submission and provide any AI-use disclosure required by the course policy.
'''

def notebook(task):
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(keepends=True)})
    def code(s):cells.append({'cell_type':'code','metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('# Task 3 — Inference-time compute allocation\n\nAbdullah Irfan · ATML PA3 · Analysis seed 6304\n\n**Default: CPU report mode.** Run all cells to verify the saved evidence and display the results. No model is loaded. GPU reproduction is disabled unless explicitly enabled in the final cell.\n\nOffline: MATH / Qwen2.5-3B and MathVista / Qwen2.5-VL-3B. Live: 50 fixed MATH questions, original pinned Qwen2.5-3B-Instruct in FP16.')
    code('''from pathlib import Path
import os, sys, json, subprocess, importlib.metadata as metadata
from IPython.display import display, Markdown, Image

if Path('/content').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')

override = os.environ.get('PA3_TASK3_REPORT_DIR')
candidates = ([Path(override)] if override else []) + [
    Path.cwd(), Path.cwd() / 'task3',
    Path('/content/drive/MyDrive/ATML/PA3/pa3_repo/task3'),
    Path('/content/drive/MyDrive/ATML/PA3/task3'),
]
TASK3 = next((p for p in candidates if (p / 'analysis/audit.json').is_file()), None)
assert TASK3 is not None, 'Set PA3_TASK3_REPORT_DIR to the finalized task3 folder.'
print('Task 3 saved-result report · CPU mode')
''')
    md('## Setup and isolation\n\nEach offline dataset has 1,000 questions × 32 rollouts. Sort IDs lexicographically and permute with default_rng(6304): 200 development, 800 held-out. Ten seeded candidate orders are shared across methods and stopping rules, with nested budgets 1, 2, 4, 8, 16, 32. The first 50 MATH held-out IDs form the live subset. All candidate rollouts remain within their question.\n\nThe scorer was frozen before performance evaluation. Supplied extracted answers are used for offline scoring; only live responses use the balanced-brace last-box extractor. Missing answers remain in voting, costs and accuracy denominators. Candidate groups and all stopping inputs exclude gold and future candidates.')
    code('''versions = json.loads((TASK3 / 'configs/scorer_frozen.json').read_text())['packages']
versions['matplotlib'] = '3.10.8'
missing = []
for name, version in versions.items():
    try:
        if metadata.version(name) != version:
            missing.append(f'{name}=={version}')
    except metadata.PackageNotFoundError:
        missing.append(f'{name}=={version}')
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *missing], check=True)
print('Pinned CPU analysis dependencies ready.')
''')
    md('## Audit the recorded evidence\n\nRecheck the frozen source/input identities, all saved offline chunks, splits and nested candidate orders, selected IDs, costs, development threshold selection and held-out metrics/intervals. Recheck every live call, seed, cap, EOS count, revision input, shared cost and final grade. This does not re-read raw parquet/NPZ/model tensor bytes; their original preparation hashes and audit receipts remain available.')
    code('''subprocess.run([
    sys.executable, str(TASK3 / 'code/task3_analyze.py'),
    '--task-dir', str(TASK3),
], check=True)
audit = json.loads((TASK3 / 'analysis/audit.json').read_text())
assert audit['passed'] and not audit['gpu_used']
for check in audit['checks']:
    print('PASS:', check)
''')
    md('## Sampling opportunity and selection errors\n\nOracle coverage asks whether any consumed candidate is correct. It is a retrospective upper bound. Self-consistency can miss a correct candidate when a wrong answer receives more votes. FirstAns consumes only one response even when compared at a larger available pool budget. Distinct groups include NO_ANSWER. The error decomposition is shown for self-consistency.')
    code("display(Image(filename=str(TASK3 / 'figures/sampling_selection.png')))")
    md('## Quality scores and aggregation\n\nMean sampled-token log-probability and mean KL(U∥p) are read from raw full-distribution measurements. Self-certainty is divergence from uniform, not a calibrated probability of correctness. RWC uses descending average ranks and weights (N−rank)/(N−1), with earliest-supporting-candidate ties. Shortest selection is a length control. All N=32 methods except FirstAns consume the same pool.')
    code("display(Image(filename=str(TASK3 / 'figures/quality_comparison.png')))")
    md('## What the signals capture\n\nQuartiles contain eight candidates within each question; top-minus-bottom correctness differences are then averaged across held-out questions. Associations are within question. Undefined Spearman correlations are excluded from the mean and counted in the CSV. Score–length association helps identify possible length effects. These are associations, not calibrated correctness estimates.')
    code("display(Image(filename=str(TASK3 / 'figures/signal_associations.png')))\nprint((TASK3 / 'analysis/signal_associations.csv').read_text())")
    md('## Visual confidence and tokenwise fusion\n\nVisual engagement divides image-attention mass by image + prompt-text + generated-text mass. Per-prefix KL calibration uses pooled token 5th/95th linear percentiles. VisConf multiplies rollout mean calibrated confidence by rollout mean visual engagement; tokenwise fusion averages their token products. Their difference equals within-rollout population covariance. For this retrospective N=32 analysis, confidence uses all 32 candidates; live stopping recalibrates only on revealed prefixes.')
    code("display(Image(filename=str(TASK3 / 'figures/visual_fusion.png')))\ndisplay(Image(filename=str(TASK3 / 'figures/fixed_trajectories.png')))")
    md('## Adaptive allocation\n\nBoth controllers inspect completed prefixes at 4, 8, 16 and otherwise stop at 32, and both return current SC. Agreement needs a unique nonmissing winner with support ≥τ. Quality also needs SC/RWC agreement and normalized RWC margin ≥γ; MATH uses self-certainty, MathVista uses VisConf. Lowest development token cost within 1 pp of development SC32 selects the threshold; held-out labels do not tune it. Error bars represent question-bootstrap accuracy intervals. The 1 pp development rule does not guarantee the same held-out loss.')
    code("display(Image(filename=str(TASK3 / 'figures/adaptive_inference.png')))\ndisplay(Image(filename=str(TASK3 / 'figures/adaptive_retrospective.png')))\nprint((TASK3 / 'analysis/adaptive.csv').read_text())\nprint((TASK3 / 'analysis/matched_cost.csv').read_text())")
    md('## Live breadth versus refinement\n\nAll three strategies have 3,072 output tokens of allowance. Single long: one call capped at 3,072. Parallel: three independently seeded calls capped at 1,024, followed by SC. Sequential: reuse parallel_0 verbatim, then one 2,048-token critique/revision call; always return the revision. Native template, original problem, prescribed system/instructions, FP16, no quantization, batch 1, temperature 0.7, top-p 0.8, top-k 20 and repetition penalty 1.0 are fixed.\n\nActual tokens differ because of early termination. Complete outputs include EOS; retained counts are separate. Sequential includes the entire critique and longer revision prompt, with initial cost charged to both strategies. Parallel samples ran serially, so times sum. Same A100, synchronized inference timing, excluding loading/smoke/disk writes. Only 50 questions: avoid strong ranking claims.')
    code("display(Image(filename=str(TASK3 / 'figures/live_strategies.png')))\nlive = json.loads((TASK3 / 'results/live_scoring/summary.json').read_text())\nprint('Revision transitions:', live['revision_transitions'])\nprint('Paired accuracy differences:')\nfor name, row in live['paired_differences'].items():\n    print(name, row['correct'])")
    md('## Resource and per-call diagnostics\n\nTruncation and extraction failures remain in scoring. The per-call table separates the three short samples, long response and revision; strategy-selected failure rates appear above. Physical runtime counts each generated call once, whereas strategy comparisons charge the shared initial response to both strategies.')
    code("print((TASK3 / 'analysis/live_call_rates.csv').read_text())\nresources = json.loads((TASK3 / 'analysis/resources.json').read_text())\nprint(json.dumps(resources, indent=2))")
    md('## Results and interpretation')
    code("display(Markdown((TASK3 / 'analysis/RESULTS.md').read_text()))")
    md('## Aligned qualitative evidence\n\nOffline examples are the first lexicographic held-out questions satisfying each saved condition, using the earliest qualifying order. Live examples are the first two lexicographic questions per transition. These illustrate selection disagreement, recoverable premature stopping and revision; they are not representative frequency estimates. Full raw response text for the offline pools remains in the supplied parquet. All live initial/revision text is retained here.')
    code("display(Markdown((TASK3 / 'analysis/QUALITATIVE.md').read_text()))")
    md('## Optional GPU reproduction\n\nThe recorded required run is complete. Leave this switch False for Run all on CPU. The README explains data/assets preparation. Only enable on the recorded GPU/package environment and prepared Drive experiment. Valid completed calls are skipped; new experiments must preserve the original evidence separately. An enabled queue releases the runtime on completion or error.')
    code('''RUN_GPU_QUEUE = False
if RUN_GPU_QUEUE:
    from google.colab import runtime
    EXPERIMENT = Path('/content/drive/MyDrive/ATML/PA3/task3')
    try:
        for script, flags in [
            ('task3_live_gpu_setup.py', []),
            ('task3_live_run.py', ['--run-all']),
        ]:
            subprocess.run([
                sys.executable, str(EXPERIMENT / 'code' / script),
                '--task-dir', str(EXPERIMENT), *flags,
            ], check=True)
    finally:
        runtime.unassign()
else:
    print('GPU queue disabled. CPU report complete.')
''')
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells):c['id']=f'task3-{i:02d}'
    write(task/'Task3.ipynb',nb)

def finalize(source,repo):
    source=Path(source).resolve();repo=Path(repo).resolve();destination=repo/'task3'
    assert source!=destination and source not in destination.parents and destination not in source.parents
    assert (source/'results/live_scoring/finished.json').is_file(),'Run CPU live scoring first.'
    analyze(source)
    (source/'README.md').write_text(README);(source/'ATTRIBUTION.md').write_text(ATTRIBUTION)
    versions=read(source/'configs/scorer_frozen.json')['packages'];versions['matplotlib']='3.10.8'
    (source/'requirements-analysis.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
    notebook(source)
    destination.mkdir(parents=True,exist_ok=True)
    # Scoped copy only. Never operate on another task or assume Drive is a Git checkout.
    allowed={'.py','.md','.json','.csv','.txt','.png','.gz','.ipynb'}
    for folder in ['code','configs','splits','results','analysis','figures']:
        for path in sorted((source/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in allowed and '__pycache__' not in path.parts:
                out=destination/path.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    for name in ['README.md','ATTRIBUTION.md','requirements-analysis.txt','Task3.ipynb']:
        shutil.copy2(source/name,destination/name)
    for name in ['live_prompts.json','live_math_references.json']:
        out=destination/'data'/name;out.parent.mkdir(exist_ok=True);shutil.copy2(source/'data'/name,out)
    top=repo/'README.md'
    existing=top.read_text() if top.exists() else '# ATML PA3\n'
    start='<!-- TASK3-START -->';end='<!-- TASK3-END -->'
    section=start+'\n## Task 3: inference-time compute allocation\n\n[Notebook](task3/Task3.ipynb) · [Reproduction and methods](task3/README.md) · [Results](task3/analysis/RESULTS.md)\n\nOffline selection/signal/adaptive experiments and the 50-question live comparison are complete. Run the notebook on CPU to audit saved records and regenerate figures.\n'+end
    if start in existing:
        assert end in existing
        before,rest=existing.split(start,1);_,after=rest.split(end,1);updated=before+section+after
    else:updated=existing.rstrip()+'\n\n'+section+'\n'
    top.write_text(updated)
    manifest={'passed':True,'task':'task3','gpu_used':False,'files':{str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='repository_manifest.json'},'note':'Task3.ipynb may be replaced by its executed CPU copy after Run all; refresh its hash before publishing.'}
    write(destination/'repository_manifest.json',manifest)
    print(json.dumps({'passed':True,'repository_task3_files':len(manifest['files']),'notebook':str(destination/'Task3.ipynb'),'other_tasks_modified':False,'gpu_used':False},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-task-dir',type=Path,required=True);p.add_argument('--repo-dir',type=Path,required=True);a=p.parse_args();finalize(a.source_task_dir,a.repo_dir)
