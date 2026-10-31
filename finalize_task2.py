"""Audit Task 2 on CPU, render selected image pairs, and organize the repo.

Original model weights, images, feature arrays and checkpoint tensors stay in
the experiment folder. Existing Task 1 files are preserved.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, shutil, sys, zipfile
from pathlib import Path
from task2_analyze import analyze, read, write

ALLOWED={'.py','.md','.json','.jsonl','.csv','.tsv','.txt','.yaml','.yml','.png','.jpg','.jpeg','.pdf','.log'}

def copy_records(source,destination):
    for folder in ['code','configs','splits','results','figures','analysis','checkpoints','features']:
        for path in sorted((source/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in ALLOWED and '__pycache__' not in path.parts:
                out=destination/path.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,out)
    image_manifest=source/'data/images/manifest.json'
    if image_manifest.exists():
        out=destination/'data/images/manifest.json';out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image_manifest,out)

def make_notebook(task):
    """Create a clean CPU report notebook with authentic saved outputs.

    The optional GPU reproduction cell is disabled by default. No git commands,
    upload dialogs, environment debugging, or old failed cells are included.
    """
    cells=[];count=0
    def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)})
    def code(text,outputs=None):
        nonlocal count
        count+=1;cells.append({'cell_type':'code','metadata':{},'source':text.splitlines(keepends=True),
                             'execution_count':count if outputs is not None else None,'outputs':outputs or []})
    def stdout(text):return [{'output_type':'stream','name':'stdout','text':text.splitlines(keepends=True)}]
    def display_file(path):
        if path.suffix=='.png':return {'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(path.read_bytes()).decode(),'text/plain':['<saved Task 2 figure>']}}
        return {'output_type':'display_data','metadata':{},'data':{'text/markdown':path.read_text().splitlines(keepends=True),'text/plain':[path.name]}}
    md('# Task 2 — Learning a visual interface\n\nSigLIP–Qwen3-0.6B, a trainable connector, and a three-stage curriculum. The required 196-token experiment is complete.\n\n**Default: CPU report mode.** Run all cells to audit the saved records and display the results. This does not load model weights or perform inference. GPU reproduction is an explicit opt-in at the end. Original weights, images, feature arrays and tensor checkpoints remain in the Drive experiment folder; the repository contains code and recorded evidence.\n\nAuthor: Abdullah Irfan · ATML PA3 · Seed 6304')
    setup='''from pathlib import Path
import os, sys, json, subprocess
from IPython.display import display, Markdown, Image

# Run from the repository's task2 folder, or from Colab with the existing Drive experiment.
if Path('/content').is_dir():
    from google.colab import drive
    drive.mount('/content/drive')

override = os.environ.get('PA3_TASK2_REPORT_DIR')
candidates = ([Path(override)] if override else []) + [
    Path.cwd(), Path.cwd() / 'task2',
    Path('/content/drive/MyDrive/ATML/PA3/pa3_repo/task2'),
    Path('/content/drive/MyDrive/ATML/PA3/task2'),
]
TASK2 = next((p for p in candidates if (p / 'analysis/audit.json').is_file()), None)
assert TASK2 is not None, 'Set PA3_TASK2_REPORT_DIR to the finalized task2 folder.'
print('Report folder:', TASK2)
print('Mode: CPU analysis of saved records')
'''
    code(setup,stdout('Report folder: task2\nMode: CPU analysis of saved records\n'))
    md('## Experimental setup\n\nThe frozen SigLIP encoder supplies 196 × 768 final patch features, normalized exactly once and cached in FP16. The connector is Linear(768,1024) → GELU → Linear(1024,1024), with 1,837,056 parameters. Its output scale was calibrated on 128 training images only and fixed at 0.05425837. Original Qwen weights remain frozen. Stages 2–3 add rank-16 attention LoRA (alpha 32, dropout 0.05), giving 6,424,576 total trainable parameters. Thinking is disabled.\n\n| Stage | Updates | Connector peak LR | LoRA peak LR | Sampling |\n| --- | ---: | ---: | ---: | --- |\n| Caption alignment | 2250 | 1e-3 | — | Two passes over 18,000 captions |\n| Visual instruction | 1500 | 1e-4 | 2e-5 | Presence 50%, counting 25%, captions 25% |\n| Natural VQA | 750 | 5e-5 | 1e-5 | Natural 70%, presence/counting/captions 10% each |\n\nMicrobatch 8 × accumulation 2; 72,000 successful example presentations. Each stage uses fresh AdamW, weight decay 0.01, 5% warmup then cosine decay toward 10% of peak LR, global clipping 1.0, FP16 AMP with dynamic scaling, FP32 trainables and optimizer states, and gradient checkpointing. Loss averages response-plus-EOT tokens within examples, then examples within the effective batch. Five initial AMP overflows were retried on the same batch.\n\nCode: `code/task2_vlm.py`, `code/task2_sampler.py`, `code/task2_train.py`; full recipe: `configs/training_recipe.json`.')
    code("recipe = json.loads((TASK2 / 'configs/training_recipe.json').read_text())\nprint(json.dumps(recipe, indent=2))",stdout(json.dumps(read(task/'configs/training_recipe.json'),indent=2)+'\n'))
    md('## Data and isolation\n\nTraining uses 6000 train2014 images: three captions per image, reusable grounding pools, and 1536 natural complementary pairs. Development has 96 caption examples, 24 grounding pairs and 32 natural pairs. Final test has 64 grounding pairs and 96 natural pairs. All five evaluation image sets are mutually disjoint and excluded from training/replay. Natural pairs have different canonical answers with at least 8/10 annotator agreement per image. Selection IDs and the mismatched-image mapping were fixed before predictions.\n\nQuestion content is capped at 80 tokens, captions at 64, and short-answer targets at 16; eligible examples were selected without truncating their original targets. Exact selection details are in `code/DATA_PREPARATION_README.md`.')
    code("selection = json.loads((TASK2 / 'results/data_selection.json').read_text())\nfor key in ['caption_train_examples', 'natural_pairs', 'grounding_pairs', 'evaluation_images_by_set', 'image_disjoint']:\n    print(f'{key}: {selection[key]}')",stdout('\n'.join(f'{k}: {read(task/"results/data_selection.json")[k]}' for k in ['caption_train_examples','natural_pairs','grounding_pairs','evaluation_images_by_set','image_disjoint'])+'\n'))
    md('## Audit the saved run\n\nThis CPU command rechecks available source/split hashes, evaluation isolation, every seeded mixture draw, stage learning rates, development means, final reference/image mappings, exact and soft scores, and all 4000-resample pair-bootstrap intervals. It regenerates the tables and plots. Model and checkpoint tensors are not loaded. Their recorded hashes are retained, but the exported records alone cannot verify their bytes. Full-resolution qualitative panels are regenerated only when the original image archive manifest is available.')
    auditcode="subprocess.run([sys.executable, str(TASK2 / 'code/task2_analyze.py'), '--task-dir', str(TASK2)], check=True)\naudit = json.loads((TASK2 / 'analysis/audit.json').read_text())\nassert audit['passed']\nfor check in audit['checks']:\n    print('PASS:', check)"
    code(auditcode,stdout('\n'.join('PASS: '+c for c in read(task/'analysis/audit.json')['checks'])+'\n'))
    md('## Learning curves\n\nDashed boundaries mark updates 2250 and 3750. Loss changes at stage transitions reflect changing training mixtures as well as learning. Caption loss includes EOT; answer NLL excludes it. Positive swap penalty means the correct image makes the original reference more likely.')
    code("for name in ['learning_curves', 'image_swap_penalties']:\n    display(Image(filename=str(TASK2 / 'figures' / f'{name}.png')))",[display_file(task/'figures'/f'{x}.png') for x in ['learning_curves','image_swap_penalties']])
    md('## Final evaluation\n\nOnly the final Stage 3 model is used. Generation is greedy, thinking disabled, batch 8, maximum 16 new tokens. Correct, complementary, mismatched and no-image conditions retain the original questions and references. No-image inputs omit visual tokens and rebuild masks/positions.\n\nGrounding uses normalized exact match. Natural VQA uses the mean leave-one-out consensus score over all ten human answers. Empty outputs remain in denominators. Bootstrap intervals use 4000 complete-pair resamples and shared pair indices across conditions. These are local subset scores, not full-benchmark server results.')
    code("display(Image(filename=str(TASK2 / 'figures/final_accuracy.png')))\ndisplay(Image(filename=str(TASK2 / 'figures/accuracy_by_type.png')))\ndisplay(Markdown((TASK2 / 'analysis/final_results.md').read_text()))",[display_file(task/'figures/final_accuracy.png'),display_file(task/'figures/accuracy_by_type.png'),display_file(task/'analysis/final_results.md')])
    md('## Answer changes under image swaps\n\nChanging an answer is not automatically correct grounding. The chart separates matching the original canonical answer, following the partner canonical answer, and matching neither. Natural VQA soft scores may award partial credit outside these canonical-string categories.')
    code("display(Image(filename=str(TASK2 / 'figures/swap_answer_behavior.png')))\nprint((TASK2 / 'analysis/swap_behavior.csv').read_text())",[display_file(task/'figures/swap_answer_behavior.png'),*stdout((task/'analysis/swap_behavior.csv').read_text())])
    md('## Aligned complementary-image examples\n\nFive pairs were fixed before predictions: the first sorted pair per question type, excluding grounding pairs flagged in the pre-training annotation review. Both original images share the same question. Each panel shows original references and responses under all four conditions. This is a stated illustrative selection, not a random sample of successes. All required evaluation examples remain in primary scoring.')
    status=read(task/'analysis/qualitative_status.json')
    qualoutputs=[display_file(task/p) for p in status['pages'] if (task/p).is_file()]
    code("qual = json.loads((TASK2 / 'analysis/qualitative_status.json').read_text())\nprint(qual['selection_rule'])\nfor relative in qual['pages']:\n    path = TASK2 / relative\n    assert path.is_file(), f'Missing qualitative panel: {path}'\n    display(Image(filename=str(path)))",[*stdout(status['selection_rule']+'\n'),*qualoutputs])
    cells[-1]['metadata']['tags']=['requires_original_images']
    md('## Resources\n\nResource timing scopes are separated below. Training time is the sum of update calls, including overflow retries; it excludes development evaluation, loading and checkpoint writes. Generation timing is amortized over batches. Allocated and reserved GPU memory are different measures. The feature benchmark selected batch 128; training and final-generation batches follow the prescribed recipe.')
    code("display(Markdown((TASK2 / 'analysis/resources.md').read_text()))",[display_file(task/'analysis/resources.md')])
    md('## Interpretation and limitations')
    code("display(Markdown((TASK2 / 'analysis/RESULTS.md').read_text()))",[display_file(task/'analysis/RESULTS.md')])
    md('## Optional reproduction on the existing prepared Drive experiment\n\nThe recorded main run is complete. Leave the following switch `False` for CPU report viewing. For a new run, first reproduce CPU data/image preparation using the scripts and pinned assets described in the README. The GPU command below uses the existing prepared Drive experiment and verifies its frozen identities. Completed phases are skipped and interrupted training resumes from its latest marked recovery state. Reproduction requires the recorded CUDA/package environment. It does not redownload missing assets automatically.\n\nOnly enable this cell after selecting the appropriate GPU runtime. The runtime is released in `finally` once an enabled queue finishes or fails.')
    code('''RUN_GPU_QUEUE = False
if RUN_GPU_QUEUE:
    from google.colab import runtime
    EXPERIMENT = Path('/content/drive/MyDrive/ATML/PA3/task2')
    assert (EXPERIMENT / 'data/tokenized_examples.json').is_file()
    try:
        subprocess.run([
            sys.executable, str(EXPERIMENT / 'code/task2_run.py'),
            '--task-dir', str(EXPERIMENT), '--run-all',
        ], check=True)
    finally:
        runtime.unassign()
else:
    print('GPU queue disabled. Saved-result analysis is complete.')
''',stdout('GPU queue disabled. Saved-result analysis is complete.\n'))
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
      'language_info':{'name':'python','version':'3.13'},'colab':{'provenance':[]}},'nbformat':4,'nbformat_minor':5}
    for i,c in enumerate(cells):c['id']=f'task2-{i:02d}'
    write(task/'Task2.ipynb',notebook)

README='''# Task 2 — Learning a visual interface

The required 196-token SigLIP–Qwen3-0.6B run completed all three stages, 4500 updates and 72000 example presentations. Final evaluation contains 1280 generated answers across two fixed test suites and four image conditions. The optional 49-token experiment was not run.

- [Task2.ipynb](Task2.ipynb): refined notebook with saved outputs; default execution is CPU report mode. GPU reproduction is disabled unless explicitly enabled.
- [Results and interpretation](analysis/RESULTS.md), [complete final tables](analysis/final_results.md), [resources](analysis/resources.md), [audit](analysis/audit.json).
- [Learning curves](figures/learning_curves.png), [swap penalties](figures/image_swap_penalties.png), [final accuracy](figures/final_accuracy.png), [question types](figures/accuracy_by_type.png), [swap behavior](figures/swap_answer_behavior.png), and five preselected [qualitative panels](figures/qualitative).
- `analysis/*.csv` and `analysis/qualitative_examples.json`: machine-readable tables and raw illustrative outputs.

## Reproduce the analysis on CPU

From this folder, with NumPy, Matplotlib and Pillow installed:

```bash
python code/task2_analyze.py --task-dir .
```

For the original Drive experiment, add `--require-images` to regenerate the qualitative panels directly from saved JPEG TARs. Repository copies already contain those panels. No torch import, model download, training or inference is required for analysis. The clean notebook uses the same command. Open its saved outputs directly on GitHub or run it on CPU in Colab.

## Reproduce the experiment

The original experiment remains at `/content/drive/MyDrive/ATML/PA3/task2`; original Qwen assets are shared from Task 1's original base-model snapshot, not Task 1 fine-tuned weights.

1. Read `code/DATA_PREPARATION_README.md`, `code/IMAGE_PREPARATION_README.md`, and `code/VLM_PREFLIGHT_README.md` for CPU preparation and fixed selection details. Sources and asset revisions/hashes are recorded under `configs/`. Data selection requires `ijson==3.6.0` in addition to the recorded environment packages.
2. Keep the original JPEG bytes and create the CPU selection/image/preflight records before allocating a GPU. The pinned Qwen and SigLIP files must be available at their recorded paths. The preparation scripts deliberately fail on changed assets/identities rather than silently replace data.
3. On CPU: `python code/task2_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task2 --prepare`.
4. On the recorded A100 CUDA environment (`configs/gpu_environment_task2.json`): `python code/task2_run.py --task-dir /content/drive/MyDrive/ATML/PA3/task2 --run-all`. The queue caches frozen vision features, tests the real GPU path, trains the three-stage curriculum, and evaluates the final Stage 3 model. The notebook's optional GPU cell releases Colab after the queue. See `code/RUNNER_README.md` for checkpoint/resume and implementation details.
5. Run CPU finalization and the notebook; inspect the interpretation and qualitative examples before incorporating findings into the combined report.

The run used PyTorch 2.11.0+cu130, Transformers 5.20.2, datasets 4.8.5, accelerate 1.15.0, peft 0.21.2, Pillow 11.3.0, NumPy 2.1.3, tokenizers 0.23.3 and huggingface_hub 1.33.0. Preserve CUDA torch when pinning a fresh GPU runtime; do not install the CPU torch build on GPU. Data preparation and report viewing require no paid GPU.

## Files and evidence limits

`code/` preserves the executed preparation/model/training/evaluation scripts, original tests, and the separate CPU analysis script. `configs/` and `splits/` preserve the selected IDs and recipe. `results/` includes all per-example final predictions, original references, NLLs, flags, image controls and development records. `checkpoints/main/` contains training logs, resource/calibration records and checkpoint completion markers. `features/patch196/` contains feature identity, benchmark and shard metadata. The image archive manifest is retained under `data/images/`.

Model weights, raw metadata archives, JPEG TARs, feature arrays and checkpoint tensors stay in the Drive experiment, outside git. The CPU audit checks available record bytes and recomputes statistics. Tensor/array content cannot be verified from metadata alone.

Annotation-derived answers have limitations. All 22 grounding contact sheets were inspected by the assistant before training; 30/88 pairs were flagged for possible ambiguity, not confirmed wrong. This was not independent human adjudication. No primary examples were removed or relabeled. Natural VQA results are high-consensus local subsets. One training seed and pair-bootstrap intervals do not capture all training uncertainty. No-image outputs often become verbose or capped, so zero answer scores combine missing visual evidence with format changes.

This analysis is a draft for student review. Keep the final combined submission consistent with the course's authorship and assistance-disclosure rules.
'''

ROOT_README='''# ATML PA3

Programming assignment experiments and results by Abdullah Irfan.

| Task | Status | Notebook | Evidence |
| --- | --- | --- | --- |
| 1. Fine-tuning and behavioral change | Five training runs and seven evaluation conditions complete | [Task 1 notebook](task1/Task1.ipynb) | [Recipe](task1/README.md), [results](task1/analysis/RESULTS.md) |
| 2. Learning a visual interface | Three-stage 196-token run and all four final image controls complete | [Task 2 notebook](task2/Task2.ipynb) | [Recipe](task2/README.md), [results](task2/analysis/RESULTS.md) |
| 3. Test-time compute | Pending | — | — |

The notebooks display recorded results and default to CPU analysis. Large model weights, image archives, cached tensors and training-state tensors remain in the original Drive experiment folders. Code, fixed splits/configurations, raw per-example evaluation records, plots and resource summaries are retained here. See each task README for reproduction and evidence limits.

The final combined report is prepared separately. Analysis notes and annotation observations should be reviewed by the student and disclosed as required by the course.
'''

def organize(source,repo):
    source=source.resolve();repo=repo.resolve();destination=repo/'task2'
    assert source!=destination,'The source must be the original experiment folder.'
    assert (repo/'task1/README.md').is_file(),'Use the existing PA3 repo so Task 1 is preserved.'
    original_task1={str(p.relative_to(repo/'task1')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (repo/'task1').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    copy_records(source,destination)
    shutil.copy2(Path(__file__).with_name('task2_analyze.py'),destination/'code/task2_analyze.py')
    if Path(__file__).resolve()!=(repo/'finalize_task2.py').resolve():shutil.copy2(Path(__file__),repo/'finalize_task2.py')
    # Keep a sibling import for the standalone finalizer command.
    if Path(__file__).with_name('task2_analyze.py').resolve()!=(repo/'task2_analyze.py').resolve():
        shutil.copy2(Path(__file__).with_name('task2_analyze.py'),repo/'task2_analyze.py')
    (destination/'README.md').write_text(README);(repo/'README.md').write_text(ROOT_README)
    make_notebook(destination)
    ignore=repo/'.gitignore';old=ignore.read_text() if ignore.exists() else ''
    additions=['*.npy','*.npz','*.tar','*.tar.gz','*.tar.*','/task2/assets/','/task2/data/annotations/','/task2/data/downloads/','/task2/data/tokenized_examples.json','/task2/data/calibration_text_ids.json']
    ignore.write_text(old.rstrip()+'\n'+'\n'.join(x for x in additions if x not in old.splitlines())+'\n')
    for rel,digest in original_task1.items():assert hashlib.sha256((repo/'task1'/rel).read_bytes()).hexdigest()==digest,'Task 1 changed: '+rel
    # Snapshot only organized Task 2 files; no stale cross-task manifest is reused.
    manifest={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.rglob('*')) if p.is_file() and p.name!='organized_manifest.json' and '__pycache__' not in p.parts}
    write(destination/'organized_manifest.json',{'files':manifest,'large_assets_in_git':False,'task1_files_preserved':len(original_task1)})
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-task-dir',type=Path,required=True);parser.add_argument('--repo-dir',type=Path)
    args=parser.parse_args();source=args.source_task_dir.resolve();repo=args.repo_dir or source.parent/'pa3_repo'
    analyze(source,require_images=True)
    destination=organize(source,repo)
    assert read(destination/'analysis/qualitative_status.json')['rendered_images']
    result={'passed':True,'cpu_only':True,'repo_task2':str(destination),'notebook':str(destination/'Task2.ipynb'),
      'task1_preserved':True,'qualitative_pairs_rendered':5,'git_commit_performed':False}
    write(destination/'analysis/finalization.json',result)
    task_manifest=read(destination/'organized_manifest.json')
    task_manifest['files']['analysis/finalization.json']=hashlib.sha256((destination/'analysis/finalization.json').read_bytes()).hexdigest()
    write(destination/'organized_manifest.json',task_manifest)
    repo=destination.parent
    files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(repo.rglob('*'))
      if p.is_file() and p.name!='package_manifest.json' and '__pycache__' not in p.parts and '.git' not in p.relative_to(repo).parts}
    write(repo/'package_manifest.json',{'version':'pa3-tasks1-2-organized-v1','files':files})
    # A small visual-review export for the final QA, rather than another data dump.
    review=source.parent/'task2_final_review.zip'
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted((destination/'analysis').glob('*')):
            if p.is_file():z.write(p,str(p.relative_to(destination)))
        for p in sorted((destination/'figures/qualitative').glob('*.png')):z.write(p,str(p.relative_to(destination)))
        z.write(destination/'Task2.ipynb','Task2.ipynb')
    print(json.dumps({**result,'review_export':str(review)},indent=2))

if __name__=='__main__':main()
