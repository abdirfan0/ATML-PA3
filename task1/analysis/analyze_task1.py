"""Rebuild Task 1 tables/figures and audit metadata on CPU, without model libraries."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

METHODS = ['full', 'l2sp', 'lora4', 'lora16', 'oplora16']
CONDITIONS = ['baseline', 'full_mid', 'full_final', 'l2sp_final', 'lora4_final', 'lora16_final', 'oplora16_final']
LABELS = ['Baseline', 'Full FT midpoint', 'Full FT', 'L2-SP', 'LoRA-4', 'LoRA-16', 'OPLoRA-16']
TASKS = {'gsm8k_cot': (1319, 'exact_match,flexible-extract'), 'hellaswag': (10042, 'acc_norm,none'), 'boolq': (3270, 'acc,none'), 'arc_easy': (2376, 'acc_norm,none')}

def read(p):
    return json.loads(p.read_text())

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def main(task):
    out = task / 'analysis'; out.mkdir(exist_ok=True)
    figures = task / 'figures'; figures.mkdir(exist_ok=True)
    config = read(task / 'configs/data_config.json')
    evaluation_config = read(task / 'configs/evaluation_config.json')
    ids = read(task / 'splits/metamath_source_indices.json')
    validation = read(task / 'results/data_validation.json')
    assert len(ids) == len(set(ids)) == 40000 and ids == sorted(ids)
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == validation['selected_ids_sha256']
    assert validation['passed'] and validation['supervised_tokens'] == 7100056
    assert hashlib.sha256((task/'assets/first_plot_questions.yaml').read_bytes()).hexdigest() == read(task/'configs/qualitative_asset.json')['sha256']
    train_sha = hashlib.sha256((task/'code/runner_v1/task1_train.py').read_bytes()).hexdigest()
    eval_sha = hashlib.sha256((task/'code/runner_v1/task1_eval.py').read_bytes()).hexdigest()
    resources, logs = [], {}
    checkpoints = (312, 625, 937, 1250)
    for method in METHODS:
        d = task/'checkpoints'/method
        finish, run = read(d/'finished.json'), read(d/'run_config.json')
        assert finish['updates'] == 1250 and finish['examples_seen'] == 40000 and not finish['smoke']
        assert run['identity']['runner_sha256'] == train_sha
        assert run['identity']['config_sha256'] == digest(config)
        assert run['identity']['selected_ids_sha256'] == validation['selected_ids_sha256']
        assert run['identity']['model_revision'] == config['model_revision']
        for step in checkpoints:
            complete = read(d/f'update_{step:04d}/complete.json')
            assert complete['update'] == step and complete['examples'] == 32 * step
        records = [json.loads(line) for line in (d/'training.jsonl').read_text().splitlines()]
        successful = [r for r in records if not r['skipped']]
        assert [r['update'] for r in successful] == list(range(1,1251))
        assert sum(r['response_tokens'] for r in successful) == validation['supervised_tokens']
        assert all(math.isfinite(r['sft_loss']) for r in successful)
        assert finish['skipped_updates'] == sum(r['skipped'] for r in records)
        if method == 'l2sp':
            assert all(abs(r['weighted_penalty'] - 0.001*r['omega']) < 1e-10 for r in successful)
            assert all(abs(r['total_loss']-r['sft_loss']-r['weighted_penalty']) < 1e-10 for r in successful)
        logs[method] = successful
        resources.append({'method':method, 'trainable_parameters':finish['trainable_parameters'],
            'trainable_percent_of_original':100*finish['trainable_parameters']/596049920,
            'training_hours':finish['training_seconds']/3600,
            'peak_allocated_GiB':finish['peak_gpu_allocated_bytes']/2**30,
            'peak_reserved_GiB':finish['peak_gpu_reserved_bytes']/2**30,
            'four_recovery_states_GiB':finish['checkpoint_bytes']/2**30,
            'checkpoint_save_seconds':sum(read(d/f'update_{s:04d}/complete.json')['save_seconds'] for s in checkpoints),
            'reference_GiB':finish['reference_storage_bytes']/2**30,
            'reference_setup_seconds':finish['reference_setup_seconds'],
            'basis_MiB':finish.get('basis_bytes',0)/2**20, 'basis_seconds':finish.get('basis_seconds',0),
            'export_MiB':finish['adapter_export_bytes']/2**20,
            'export_seconds':finish['adapter_export_seconds'], 'skipped_updates':finish['skipped_updates']})
    assert read(task/'checkpoints/lora16/setup.json')['initial_factors_sha256'] == read(task/'checkpoints/oplora16/setup.json')['initial_factors_sha256']
    baseline = {}
    benchmarks, diagnostics, extraction, qual_all = [], [], [], []
    prompt_hashes = {}
    expected_prompts = {r['id']:r for r in read(task/'splits/qualitative_prompts.json')}
    expected_probes = [(r['task'],r['row_index']) for r in read(task/'splits/probe_prompts.json')]
    for condition, label in zip(CONDITIONS,LABELS):
        d = task/'results/evaluation'/condition
        identity = read(d/'identity.json'); finish=read(d/'finished.json')
        assert identity['code_sha256'] == eval_sha and identity['evaluation_config_sha256'] == digest(evaluation_config)
        assert finish['condition'] == condition and not finish['smoke']
        assert all(finish[k] == v for k,v in identity.items())
        row = {'condition':condition,'label':label}
        for name,(count,primary) in TASKS.items():
            result=read(d/f'{name}_metrics.json')
            assert result['condition']==condition and not result['smoke']
            e=result['evaluation']; scores=e['results'][name]; cfg=e['configs'][name]
            assert e['n-samples'][name] == {'original':count,'effective':count}
            assert scores['sample_len']==count and e['config']['limit'] is None
            assert cfg['num_fewshot']==0
            assert cfg['dataset_kwargs']['revision']==evaluation_config['datasets'][name]['revision']
            if name=='gsm8k_cot':
                assert cfg['generation_kwargs']['max_gen_toks']==1024 and cfg['generation_kwargs']['do_sample'] is False
            value=scores[primary]*100
            if condition=='baseline': baseline[name]=value
            row[name+'_percent']=value; row[name+'_delta_pp']=value-baseline[name]
            if name in ('hellaswag','arc_easy'): row[name+'_raw_accuracy_percent']=100*scores['acc,none']
        benchmarks.append(row)
        extract=read(d/'gsm8k_extraction_summary.json')
        extraction.append({'condition':condition,**extract})
        q=read(d/'qualitative.json')
        assert len(q)==len({r['id'] for r in q})==24
        for r in q:
            canonical=expected_prompts[r['id']]
            assert all(r[k]==canonical[k] for k in ('base_id','format','messages','annotate'))
            if condition=='baseline':prompt_hashes[r['id']]=r['prompt_token_sha256']
            assert r['prompt_token_sha256']==prompt_hashes[r['id']]
            assert r['output_cap']==512 and r['output_tokens']==len(r['output_token_ids'])<=512
            qual_all.append({'condition':condition,**r})
        for entry in read(task/f'results/diagnostics/{condition}.json'):
            assert [(r['task'],r['row_index']) for r in entry['activation_ratios']]==expected_probes
            assert len(entry['activation_ratios'])==64
            diagnostics.append({'condition':condition, 'layer':entry['layer'], 'rho_k':entry['rho_k'], 'eta':entry['eta'],
                **{f'R_{suite}_mean':mean(r['R'] for r in entry['activation_ratios'] if r['task']==suite) for suite in ('gsm8k_cot','hellaswag')}})
        details=read(d/'model_details.json')
        if condition.startswith(('lora','oplora')):
            assert details['merge_check']['dtype']=='float32'
            assert details['merge_check']['relative_l2_difference']<1e-4
    assert len(qual_all)==168 and sum(r['annotate'] for r in qual_all)==42
    assert all(d['rho_k']<1e-10 for d in diagnostics if d['condition']=='oplora16_final')
    write_csv(out/'benchmarks.csv',benchmarks); write_csv(out/'resources.csv',resources)
    write_csv(out/'diagnostics.csv',diagnostics); write_csv(out/'gsm8k_extraction.csv',extraction)
    # Group all saved responses by underlying question and format, then condition.
    order={name:i for i,name in enumerate(expected_prompts)}
    qual_all.sort(key=lambda r:(order[r['id']],CONDITIONS.index(r['condition'])))
    (out/'qualitative_all_168.json').write_text(json.dumps(qual_all,indent=2,ensure_ascii=False))
    audit={'passed':True,'training_runs':5,'updates_each':1250,'examples_each':40000,'supervised_tokens_each':7100056,
           'evaluation_conditions':7,'complete_benchmark_splits':28,'qualitative_responses':168,'fixed_annotation_subset':42,
           'matched_rank16_initialization':True,'OPLoRA_max_rho':max(d['rho_k'] for d in diagnostics if d['condition']=='oplora16_final'),
           'evidence_scope':'Uploaded small-file metadata, logs, metrics and responses. Large states and per-example benchmark JSONL require the CPU finalization audit on Drive.'}
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    table=['| Condition | GSM8K | HellaSwag | BoolQ | ARC-Easy |','|---|---:|---:|---:|---:|']
    for row in benchmarks:
        cells=[f"{row[t+'_percent']:.2f} ({row[t+'_delta_pp']:+.2f})" for t in TASKS]
        table.append('| '+row['label']+' | '+' | '.join(cells)+' |')
    (out/'benchmark_table.md').write_text('Scores in percent; parentheses are changes from baseline in percentage points. GSM8K uses flexible extraction; HellaSwag/ARC use normalized accuracy.\n\n'+'\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
    colors=['#2864a5','#e08d26','#319968','#9b59a6','#cb4d4d']
    def save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=190); fig.savefig(figures/(name+'.pdf')); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),gridspec_kw={'width_ratios':[1.7,1]})
    for method,label,color in zip(METHODS,LABELS[2:],colors):
        rs=logs[method]; raw=np.array([r['sft_loss'] for r in rs])
        # Token-weighted block averages, exact original loss retained in JSONL.
        xx=[]; yy=[]
        for start in range(0,len(rs),25):
            block=rs[start:start+25]; xx.append(mean(r['update'] for r in block))
            yy.append(sum(r['sft_loss']*r['response_tokens'] for r in block)/sum(r['response_tokens'] for r in block))
        axes[0].plot(xx,yy,label=label,color=color,lw=1.5)
    axes[0].set(xlabel='Optimizer update',ylabel='SFT loss (25-update token-weighted blocks)',title='Shared data order; one epoch');axes[0].legend(ncol=2,fontsize=8)
    for key,label in [('sft_loss','SFT loss'),('weighted_penalty','λΩ'),('total_loss','Total objective')]:
        rs=logs['l2sp']; axes[1].plot([r['update'] for r in rs],[mean(v[key] for v in rs[max(0,i-24):i+1]) for i in range(len(rs))],label=label,lw=1)
    axes[1].set(xlabel='Optimizer update',ylabel='L2-SP objective terms',title='Trailing 25-update means');axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'learning_curves')
    fig,ax=plt.subplots(figsize=(7.9,3.8))
    matrix=np.array([[r[t+'_delta_pp'] for t in TASKS] for r in benchmarks[1:]])
    image=ax.imshow(matrix,cmap='RdBu',vmin=-22,vmax=22,aspect='auto')
    ax.set_xticks(range(4),['GSM8K','HellaSwag','BoolQ','ARC-Easy']);ax.set_yticks(range(6),LABELS[1:])
    for i in range(6):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontsize=10)
    fig.colorbar(image,ax=ax,label='Change from baseline (percentage points)');fig.tight_layout();save(fig,'benchmark_changes')
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for metric,ax,title in zip(['rho_k','eta','R_gsm8k_cot_mean','R_hellaswag_mean'],axes.flat,['Subspace interference ρ₁₂₈','Relative update norm η','Math probes: mean R(z)','HellaSwag probes: mean R(z)']):
        for c,label,color in zip(CONDITIONS[2:],LABELS[2:],colors):
            subset=[d for d in diagnostics if d['condition']==c]
            vals=[d[metric] for d in subset]; ax.plot([d['layer'] for d in subset],vals,'o-',label=label,color=color,markersize=4,lw=1.2)
        ax.set(title=title,xlabel='down_proj layer',xticks=[0,13,27]);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7,ncol=2);fig.tight_layout();save(fig,'diagnostics')
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for ax,key,label in [(axes[0],'training_hours','Recorded training time (hours)'),(axes[1],'peak_allocated_GiB','Peak allocated GPU memory (GiB)')]:
        ax.bar(range(5),[r[key] for r in resources],color=colors);ax.set_xticks(range(5),LABELS[2:],rotation=20);ax.set_ylabel(label)
    fig.tight_layout();save(fig,'resources')
    print(json.dumps(audit,indent=2));print('\n'+'\n'.join(table))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,default=Path(__file__).resolve().parents[1]);main(parser.parse_args().task_dir)
