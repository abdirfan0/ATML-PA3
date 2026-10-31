"""CPU-only Drive audit and preservation of submission outputs. No source files are deleted."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

METHODS=('full','l2sp','lora4','lora16','oplora16')
CONDITIONS=('baseline','full_mid','full_final','l2sp_final','lora4_final','lora16_final','oplora16_final')
TASKS={'gsm8k_cot':1319,'hellaswag':10042,'boolq':3270,'arc_easy':2376}

def read(p):return json.loads(p.read_text())

def verify_samples(path,metrics,task,summary=None):
    count=TASKS[task]
    groups={};ids={};trunc=missing=0
    with path.open() as stream:
        for line in stream:
            row=json.loads(line); filt=row.get('filter','none')
            key=row['doc_id'];seen=ids.setdefault(filt,set())
            assert key not in seen,(path,'duplicate doc_id',filt,key)
            seen.add(key)
            keys=['exact_match'] if task=='gsm8k_cot' else (['acc'] if task=='boolq' else ['acc','acc_norm'])
            for metric in keys:
                value=row[metric]
                assert isinstance(value,(int,float,bool)) and math.isfinite(value)
                sums=groups.setdefault(filt,{m:0.0 for m in keys});sums[metric]+=float(value)
            if task=='gsm8k_cot':
                assert 'generation' in row and 'truncated' in row['generation']
                if filt=='flexible-extract':
                    trunc+=bool(row['generation']['truncated'])
                    missing+=row['filtered_resps'][0] in ('[invalid]','',None)
    expected_filters={'strict-match','flexible-extract'} if task=='gsm8k_cot' else {'none'}
    assert set(groups)==expected_filters,(path,groups.keys())
    for filt,sums in groups.items():
        assert ids[filt]==set(range(count)),(path,filt,len(ids[filt]))
        for metric,total in sums.items():
            assert math.isclose(total/count,metrics[f'{metric},{filt}'],abs_tol=1e-12), (path,metric,filt,total/count,metrics)
    if summary is not None:
        assert summary['questions']==count
        assert math.isclose(summary['truncation_rate'],trunc/count,abs_tol=1e-12)
        assert math.isclose(summary['missing_extraction_rate'],missing/count,abs_tol=1e-12)
    return {'unique_examples':count,'rows':sum(len(v) for v in ids.values()),'filters':sorted(ids)}

def same_copy(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert source.read_bytes()==destination.read_bytes(),f'Conflicting file: {destination}'
    else:shutil.copyfile(source,destination)

def main(source,repo):
    assert source.resolve()!=repo.resolve()
    assert (repo/'analysis/audit.json').exists(),'Extract the review package before finalizing.'
    checks=[]
    for relative in ('configs/data_config.json','configs/evaluation_config.json','results/data_validation.json'):
        assert read(source/relative)==read(repo/relative),f'Source evidence changed: {relative}'
    # Verify large states in place by recorded byte count; never read entire states into RAM.
    for method in METHODS:
        for update in (312,625,937,1250):
            folder=source/f'checkpoints/{method}/update_{update:04d}'
            complete=read(folder/'complete.json'); state=folder/'training_state.pt'
            assert state.is_file() and state.stat().st_size==complete['bytes'],state
            checks.append({'path':str(state),'bytes':state.stat().st_size,'verification':'exists and matches recorded byte count, not tensor-content validation'})
        if method.startswith(('lora','oplora')):
            export=source/f'checkpoints/{method}/adapter_inference.pt'
            # The actual trainer puts the inference export in update_1250.
            if not export.exists():export=source/f'checkpoints/{method}/update_1250/adapter_inference.pt'
            assert export.is_file(),export
            assert export.stat().st_size==read(source/f'checkpoints/{method}/finished.json')['adapter_export_bytes'],export
    for condition in CONDITIONS:
        folder=source/f'results/evaluation/{condition}'
        for task in TASKS:
            path=folder/f'{task}_samples.jsonl'
            metric=read(folder/f'{task}_metrics.json')['evaluation']['results'][task]
            summary=read(folder/'gsm8k_extraction_summary.json') if task=='gsm8k_cot' else None
            checked=verify_samples(path,metric,task,summary)
            dest=repo/f'results/evaluation/{condition}/{task}_samples.jsonl.gz'
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_suffix('.gz.tmp')
            with path.open('rb') as incoming, temporary.open('wb') as raw:
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as outgoing:
                    shutil.copyfileobj(incoming,outgoing)
            os.replace(temporary,dest)
            checks.append({'path':str(path.relative_to(source)),**checked,
                           'compressed_bytes':dest.stat().st_size,'compressed_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
            print(f'PASS {condition}/{task}: {checked["unique_examples"]} examples',flush=True)
    manifest=source/'data/metamath_40k_manifest.json'
    same_copy(manifest,repo/'data/metamath_40k_manifest.json')
    # Preserve notebook source if it was saved inside the Task 1 folder.
    for notebook in source.rglob('*.ipynb'):
        if any(part in ('data','assets','checkpoints','.ipynb_checkpoints') for part in notebook.relative_to(source).parts):continue
        same_copy(notebook,repo/'notebooks'/notebook.name)
    baseline_reference=source/'results/diagnostics/baseline_reference.pt'
    assert baseline_reference.is_file(),baseline_reference
    result={'passed':True,'source_task_dir':str(source),'repository_task_dir':str(repo),
      'training_states':20,'benchmark_output_files':28,'checks':checks,
      'baseline_reference_bytes':baseline_reference.stat().st_size,
      'note':'Large states, dataset snapshots, and activation reference remain in the original Drive task directory. No GPU/model load or deletion occurred.'}
    (repo/'analysis/drive_audit.json').write_text(json.dumps(result,indent=2))
    print('\nDrive audit PASS. Repo files:',repo.parent)
    print('Original checkpoints, datasets and caches retained:',source)
    print('Qualitative draft annotations still require your review. Public GitHub publishing remains a separate step.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-task-dir',required=True,type=Path)
    parser.add_argument('--repo-task-dir',type=Path,default=Path(__file__).resolve().parent/'task1')
    args=parser.parse_args();main(args.source_task_dir,args.repo_task_dir)
