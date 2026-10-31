"""CPU audit and report generation for completed PA3 Task 2 records.

No torch import, model loading, training, generation, or external requests.
Run: python task2_analyze.py --task-dir PATH [--require-images]
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, importlib.util, json, math, sys, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS = ('correct', 'complementary', 'mismatched', 'no_image')
LABELS = {'correct':'Correct', 'complementary':'Complementary', 'mismatched':'Mismatched', 'no_image':'No image'}
COLORS = dict(zip(CONDITIONS, ['#1976a3', '#d07728', '#7c629d', '#737b83']))
DEVELOPMENT = [0,500,1000,1500,2000,2250,2500,3000,3500,3750,4000,4500]

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
def same(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:same(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=1e-10,atol=1e-10),(a,b)
    else:assert a==b,(a,b)

def pair_values(rows, ids):
    grouped=collections.defaultdict(list)
    for row in rows:grouped[row['pair_id']].append(row)
    values=collections.defaultdict(list)
    for pair in ids:
        a,b=grouped[pair];assert a['example_id']!=b['example_id']
        values['accuracy'].append((a['score']+b['score'])/2)
        values['reference_nll'].append((a['reference_nll']+b['reference_nll'])/2)
        values['pair_accuracy'].append(float(a['canonical_exact_match'] and b['canonical_exact_match']))
        values['identical_answer_rate'].append(float(a['normalized_prediction']==b['normalized_prediction']))
        values['empty_answer_rate'].append(((a['normalized_prediction']=='')+(b['normalized_prediction']==''))/2)
        values['partner_answer_rate'].append((a['partner_exact_match']+b['partner_exact_match'])/2)
        values['generation_seconds_per_answer'].append((a['generation_seconds_amortized']+b['generation_seconds_amortized'])/2)
    return {k:np.asarray(v,dtype=float) for k,v in values.items()}

def summarize(records,pairs,suite):
    result={'bootstrap_resamples':4000,'unit':'complete question pair','seed':6304,'groups':{}}
    groups={'overall':pairs,**{t:[p for p in pairs if p['question_type']==t] for t in sorted({p['question_type'] for p in pairs})}}
    for name,subset in groups.items():
        ids=sorted(p['pair_id'] for p in subset);allowed=set(ids)
        salt=int.from_bytes(hashlib.sha256((suite+':'+name).encode()).digest()[:4],'little')
        indices=np.random.default_rng(6304+salt).integers(len(ids),size=(4000,len(ids)))
        arrays={c:pair_values([r for r in rows if r['pair_id'] in allowed],ids) for c,rows in records.items()}
        def interval(v):return {'mean':float(v.mean()),'ci95':np.percentile(v[indices].mean(axis=1),[2.5,97.5]).tolist()}
        result['groups'][name]={'pairs':len(ids),'questions':2*len(ids),
          'conditions':{c:{k:interval(v) for k,v in a.items()} for c,a in arrays.items()},
          'correct_image_advantages':{
            'accuracy_correct_minus_complementary':interval(arrays['correct']['accuracy']-arrays['complementary']['accuracy']),
            'nll_complementary_minus_correct':interval(arrays['complementary']['reference_nll']-arrays['correct']['reference_nll'])}}
    return result

def audit(task):
    checks=[];unavailable=[]
    def passed(label):checks.append(label)
    export=task/'review_export_manifest.json'
    if export.exists():
        for rel,record in read(export).items():
            p=task/rel;assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256'],rel
        passed('All exported file sizes and SHA256 hashes match the export manifest.')
    answer=load_module('task2_answer_audit',task/'code/task2_answers.py')
    normalize=answer.normalize_answer
    finished=read(task/'checkpoints/main/finished.json');identity=finished['identity']
    assert finished['passed'] and finished['updates']==4500 and finished['example_presentations']==72000
    for name,digest in identity['code_sha256'].items():assert sha(task/'code'/name)==digest,name
    for rel,digest in identity['inputs_sha256'].items():
        if (task/rel).is_file():assert sha(task/rel)==digest,rel
        else:unavailable.append(rel)
    assert sha(task/'features/patch196/manifest.json')==identity['feature_manifest_sha256']
    selection=read(task/'splits/selection_manifest.json')
    for rel,digest in selection['files'].items():assert sha(task/rel)==digest,rel
    for rel in ['results/gpu_smoke_task2.json','results/final/finished.json']:
        assert read(task/rel)['identity']==identity and read(task/rel)['passed']
    passed('Training, smoke, final evaluation, source code, available inputs, and selected-split hashes agree.')
    splits={n:read(task/f'splits/{n}.json') for n in ['captions_train','captions_dev','grounding_train','grounding_dev','grounding_test','natural_train','natural_dev','natural_test']}
    assert len(splits['captions_train'])==18000 and len(splits['captions_dev'])==96
    train_images=collections.Counter(e['image_id'] for e in splits['captions_train'])
    assert len(train_images)==6000 and set(train_images.values())=={3}
    expected={'natural_train':1536,'natural_dev':32,'natural_test':96,'grounding_train':8192,'grounding_dev':24,'grounding_test':64}
    image_sets={}
    for name,rows in splits.items():
        examples=rows if name.startswith('captions') else [e for p in rows for e in p['examples']]
        assert len({e['example_id'] for e in examples})==len(examples)
        image_sets[name]={e['image_id'] for e in examples}
        assert all(e['image_split']==('train2014' if name.endswith('train') else 'val2014') for e in examples)
        if name in expected:
            assert len(rows)==expected[name]
            assert len({p['pair_id'] for p in rows})==len(rows)
            for p in rows:
                a,b=p['examples'];assert a['question']==b['question']==p['question']
                assert a['image_id']!=b['image_id'] and a['canonical_answer']!=b['canonical_answer']
                if name.startswith('natural'):
                    for e in (a,b):
                        refs=[normalize(x) for x in e['answers']]
                        assert len(refs)==10 and refs.count(e['canonical_answer'])>=8
                        assert normalize(e['target'])==e['canonical_answer'] and e['target'] in e['answers']
                else:
                    assert all(normalize(e['target'])==e['canonical_answer'] for e in (a,b))
                    assert {a['canonical_answer'],b['canonical_answer']}<=({'yes','no'} if p['question_type']=='presence' else {'1','2','3','4'})
    eval_names=['captions_dev','natural_dev','natural_test','grounding_dev','grounding_test']
    for i,name in enumerate(eval_names):
        assert not (image_sets[name]&set(train_images))
        for other in eval_names[i+1:]:assert not image_sets[name]&image_sets[other],(name,other)
    for name in ['natural_train','grounding_train']:assert image_sets[name]<=set(train_images)
    for split,counts in [('train',{'yes/no':640,'number':256,'other':640}),('dev',{'yes/no':12,'number':8,'other':12}),('test',{'yes/no':36,'number':24,'other':36})]:
        assert dict(collections.Counter(p['question_type'] for p in splits['natural_'+split]))==counts
    passed('Dataset sizes, original consensus targets, pair structure, natural type quotas, and all evaluation image isolation checks pass.')
    features=read(task/'features/patch196/manifest.json')
    keys=[(e['image_split'],e['image_id']) for s in features['shards'] for e in s['images']]
    assert len(keys)==len(set(keys))==6528
    assert set(keys)=={(e['image_split'],e['image_id']) for e in read(task/'splits/selected_images.json')}
    assert len(features['shards'])==26 and features['identity']['dtype']=='float16'
    for shard in features['shards']:
        assert shard['final_layer_norm_once']
        same(shard,read(task/'features/patch196'/Path(shard['file_name']).with_suffix('.json')))
    passed('All 26 feature shard records cover the 6528 selected images exactly; FP16 and one final normalization are recorded.')
    log=[json.loads(line) for line in (task/'checkpoints/main/training.jsonl').read_text().splitlines() if line.strip()]
    assert len(log)==4500 and [r['update'] for r in log]==list(range(1,4501))
    sampler_module=load_module('task2_sampler_audit',task/'code/task2_sampler.py')
    pools=splits['grounding_train']
    sampler=sampler_module.CurriculumSampler(splits['captions_train'],[p for p in pools if p['question_type']=='presence'],[p for p in pools if p['question_type']=='counting'],splits['natural_train'])
    for row in log:
        n=row['update'];stage=1 if n<=2250 else (2 if n<=3750 else 3);offset={1:0,2:2250,3:3750}[stage]
        assert row['stage']==stage and row['stage_update']==n-offset and row['examples_seen']==16*n
        assert all(math.isfinite(row[k]) for k in ['loss','gradient_norm_before_clip','scale','update_seconds'])
        batch,draws=sampler.batch(stage)
        assert draws==row['mixture_draws'] and dict(sampler.counts[stage])==row['cumulative_mixture_draws']
        length={1:2250,2:1500,3:750}[stage];step=n-offset-1;warm=math.ceil(.05*length)
        factor=step/warm if step<warm else .1+.9*.5*(1+math.cos(math.pi*(step-warm)/(length-warm)))
        for group,lr in row['learning_rates'].items():
            peak={'connector':{1:1e-3,2:1e-4,3:5e-5}[stage],'lora':{1:0,2:2e-5,3:1e-5}[stage]}[group]
            assert np.isclose(lr,peak*factor,rtol=1e-10,atol=1e-14)
    assert sampler.caption_position==36000
    assert sum(r['skipped_attempts'] for r in log)==finished['skipped_attempts']==5
    markers=sorted((task/'checkpoints/main').glob('update_*/complete.json'))
    assert [read(p)['update'] for p in markers]==list(range(250,4501,250))
    assert sum(read(p)['bytes'] for p in markers)==finished['recovery_state_bytes']
    for s in (1,2,3):
        stage=finished['stages'][str(s)];assert stage['mixture_draws']==dict(sampler.counts[s])
        assert np.isclose(stage['training_seconds'],sum(x['update_seconds'] for x in log if x['stage']==s))
    passed('4500 successful updates, 72000 presentations, 18 checkpoint markers, every seeded mixture draw, and all stage learning rates verified.')
    dev=[read(task/f'results/development/update_{n:04d}.json') for n in DEVELOPMENT]
    assert len(list((task/'results/development').glob('update_*.json')))==12
    for d,n in zip(dev,DEVELOPMENT):
        assert d['update']==n and d['identity']==identity
        assert len(d['caption_examples'])==96
        assert np.isclose(d['caption_reference_loss'],np.mean([r['loss'] for r in d['caption_examples']]))
        for suite in ('grounding','natural'):
            expected_ids={e['example_id'] for p in splits[suite+'_dev'] for e in p['examples']}
            conditions=d['suites'][suite]['conditions']
            for c,record in conditions.items():
                assert {r['example_id'] for r in record['examples']}==expected_ids
                assert all(math.isfinite(r['nll']) for r in record['examples'])
                assert np.isclose(record['mean_answer_nll'],np.mean([r['nll'] for r in record['examples']]))
            assert np.isclose(d['suites'][suite]['image_swap_penalty'],conditions['complementary']['mean_answer_nll']-conditions['correct']['mean_answer_nll'])
    passed('All 12 development evaluations, per-example means, and image-swap penalty calculations verified.')
    summaries={};records={};controls=read(task/'splits/image_controls.json')
    for suite in ('grounding','natural'):
        pairs=splits[suite+'_test'];original={e['example_id']:(p,side,e) for p in pairs for side,e in enumerate(p['examples'])}
        mismatch={r['example_id']:r for r in controls[suite+'_test']};records[suite]={}
        for c in CONDITIONS:
            file=read(task/f'results/final/{suite}_{c}.json');assert file['identity']==identity
            rows=file['examples'];assert len(rows)==len(original) and {r['example_id'] for r in rows}==set(original)
            records[suite][c]=rows
            for row in rows:
                pair,side,ref=original[row['example_id']];partner=pair['examples'][1-side]
                assert row['pair_id']==pair['pair_id'] and row['question_type']==pair['question_type'] and row['condition']==c
                assert row['question']==ref['question'] and row['references']==ref['answers'] and row['reference_string']==ref['target']
                assert row['canonical_answer']==ref['canonical_answer'] and row['partner_canonical_answer']==partner['canonical_answer']
                expected_image={'correct':ref['image_id'],'complementary':partner['image_id'],'mismatched':mismatch[row['example_id']]['mismatched_image_id'],'no_image':None}[c]
                assert row['original_image_id']==ref['image_id'] and row['image_id_used']==expected_image
                if c=='mismatched':assert expected_image in image_sets[suite+'_test'] and expected_image not in [ref['image_id'],partner['image_id']]
                pred=normalize(row['response']);assert pred==row['normalized_prediction']
                refs=[normalize(v) for v in ref['answers']]
                # Direct leave-one-out definition, independent of the run's scoring function.
                score=sum(min(1.,sum(pred==v for j,v in enumerate(refs) if j!=i)/3) for i in range(10))/10 if suite=='natural' and pred else float(bool(pred) and pred==ref['canonical_answer'])
                assert np.isclose(score,row['score'])
                assert row['canonical_exact_match']==(bool(pred) and pred==ref['canonical_answer'])
                assert row['partner_exact_match']==(bool(pred) and pred==partner['canonical_answer'])
                assert math.isfinite(row['reference_nll']) and row['reference_nll']>=0
                assert len(row['output_token_ids'])==row['output_tokens']<=16
                assert row['ended_by_eos']==(row['output_token_ids'][-1] in file['eos_ids'])
                assert row['truncated']==(not row['ended_by_eos'] and row['output_tokens']==16)
        baseline={r['example_id']:r for r in records[suite]['correct']}
        for c in CONDITIONS:
            for row in records[suite][c]:
                for key in ['prompt_before_ids','prompt_after_ids','question','references','reference_string']:
                    assert row[key]==baseline[row['example_id']][key]
        summaries[suite]=summarize(records[suite],pairs,suite)
        same(summaries[suite],read(task/f'results/final/{suite}_summary.json'))
    passed('All 1280 final predictions, references, image controls, exact/soft scores, EOS/cap flags, pair metrics, and 4000-resample paired intervals independently recomputed.')
    limitations=[
      'Checkpoint tensor files, cached feature arrays, original model weights and image TARs are not included in this review export; their recorded hashes are retained but their bytes were not re-audited here.',
      'Raw tokenized examples and calibration token lists are not in the review export. The prior CPU and GPU preflight records and frozen input hashes are available.',
      'The 30 annotation-ambiguity flags came from assistant contact-sheet inspection before training, not independent human adjudication; primary scores retain every fixed example.',
      'Single seed and selected high-consensus subsets; bootstrap intervals quantify pair sampling variation, not training-seed uncertainty or full-benchmark performance.'
    ]
    report={'passed':True,'checks':checks,'unavailable_input_bytes':unavailable,'limits':limitations,
      'successful_updates':4500,'presentations':72000,'checkpoint_markers':18,'development_evaluations':12,'generated_answers':1280}
    return report,log,dev,summaries,records

def csv_write(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def markdown_table(rows, columns):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(row.get(c,'')) for c in columns)+' |' for row in rows)+'\n'
def cell(value, percentage=False):
    k=100 if percentage else 1;digits=2 if percentage else 3
    return f"{k*value['mean']:.{digits}f} [{k*value['ci95'][0]:.{digits}f}, {k*value['ci95'][1]:.{digits}f}]"

def tables(task, summaries, records, dev):
    out=task/'analysis';out.mkdir(exist_ok=True)
    flat=[];pretty=[];adv=[];advpretty=[]
    rate_metrics={'accuracy','pair_accuracy','identical_answer_rate','empty_answer_rate','partner_answer_rate'}
    for suite,s in summaries.items():
        for group,data in s['groups'].items():
            for c,metrics in data['conditions'].items():
                row={'suite':suite,'question_type':group,'condition':c,'pairs':data['pairs'],'questions':data['questions']}
                for metric,value in metrics.items():
                    for key,v in [('mean',value['mean']),('ci95_low',value['ci95'][0]),('ci95_high',value['ci95'][1])]:row[metric+'_'+key]=v
                raw=[r for r in records[suite][c] if group=='overall' or r['question_type']==group]
                row['truncation_rate']=float(np.mean([r['truncated'] for r in raw]));row['mean_output_tokens']=float(np.mean([r['output_tokens'] for r in raw]));flat.append(row)
                pretty.append({'Suite':suite,'Type':group,'Image':LABELS[c],'Questions':data['questions'],
                  'Accuracy % [95% CI]':cell(metrics['accuracy'],True),'Pair accuracy % [95% CI]':cell(metrics['pair_accuracy'],True),
                  'Identical % [95% CI]':cell(metrics['identical_answer_rate'],True),'Empty % [95% CI]':cell(metrics['empty_answer_rate'],True),
                  'Reference NLL [95% CI]':cell(metrics['reference_nll']),
                  'Generation ms/answer':f"{metrics['generation_seconds_per_answer']['mean']*1000:.2f}",
                  'Capped %':f"{row['truncation_rate']*100:.2f}"})
            for metric,value in data['correct_image_advantages'].items():
                adv.append({'suite':suite,'question_type':group,'metric':metric,'mean':value['mean'],'ci95_low':value['ci95'][0],'ci95_high':value['ci95'][1]})
            advpretty.append({'Suite':suite,'Type':group,'Accuracy advantage pp [95% CI]':cell(data['correct_image_advantages']['accuracy_correct_minus_complementary'],True),
                             'NLL advantage [95% CI]':cell(data['correct_image_advantages']['nll_complementary_minus_correct'])})
    csv_write(out/'final_metrics.csv',flat);csv_write(out/'correct_image_advantages.csv',adv)
    (out/'final_results.md').write_text('# Final results\n\nGrounding uses normalized exact match; natural VQA uses the ten-answer leave-one-out soft score. Percentages are on a 0–100 scale. Intervals use 4000 complete-pair bootstrap resamples.\n\n'+markdown_table([p for p in pretty if p['Type']=='overall'],list(pretty[0]))+'\n## By question type\n\n'+markdown_table([p for p in pretty if p['Type']!='overall'],list(pretty[0]))+'\n## Correct minus complementary advantages\n\nAccuracy is in percentage points; NLL is complementary minus correct.\n\n'+markdown_table(advpretty,list(advpretty[0])))
    devrows=[]
    for d in dev:
        row={'update':d['update'],'caption_loss':d['caption_reference_loss']}
        for suite in ('grounding','natural'):
            for c,r in d['suites'][suite]['conditions'].items():row[f'{suite}_{c}_nll']=r['mean_answer_nll']
            row[suite+'_swap_penalty']=d['suites'][suite]['image_swap_penalty']
        devrows.append(row)
    csv_write(out/'development_metrics.csv',devrows)
    behavior=[]
    for suite in ('grounding','natural'):
        base={r['example_id']:r for r in records[suite]['correct']};swapped=records[suite]['complementary']
        transitions=collections.Counter('original' if r['canonical_exact_match'] else ('partner' if r['partner_exact_match'] else 'neither') for r in swapped)
        changed=[r for r in swapped if r['normalized_prediction']!=base[r['example_id']]['normalized_prediction']]
        behavior.append({'suite':suite,'questions':len(swapped),'prediction_changed_rate':len(changed)/len(swapped),
          'swap_original_canonical_rate':transitions['original']/len(swapped),'swap_partner_canonical_rate':transitions['partner']/len(swapped),
          'swap_neither_canonical_rate':transitions['neither']/len(swapped),
          'changed_and_partner_rate':sum(r['partner_exact_match'] for r in changed)/len(swapped)})
    csv_write(out/'swap_behavior.csv',behavior)
    finished=read(task/'checkpoints/main/finished.json');resources=[]
    feature=read(task/'results/feature_caching.json')
    resources.append({'phase':'Feature caching','updates':0,'presentations':6528,'trainable_parameters':0,
      'measured_seconds':feature['total_seconds_this_session'],'peak_allocated_GiB':feature['peak_gpu_allocated_bytes']/2**30,
      'peak_reserved_GiB':'','timing_scope':'Feature-cache process session; staging, preprocessing, forward and persistence. Excludes smoke and main training.'})
    for n,stage in finished['stages'].items():
        updates={'1':2250,'2':1500,'3':750}[n]
        resources.append({'phase':'Stage '+n,'updates':updates,'presentations':updates*16,'trainable_parameters':stage['trainable_parameters'],
          'measured_seconds':stage['training_seconds'],'peak_allocated_GiB':stage['peak_gpu_allocated_bytes']/2**30,
          'peak_reserved_GiB':stage['peak_gpu_reserved_bytes']/2**30,'timing_scope':'Sum of update calls, including overflow retries; excludes development, checkpoint saves and model loading.'})
    f=read(task/'results/final/finished.json')
    resources.append({'phase':'Final evaluation','updates':0,'presentations':1280,'trainable_parameters':0,
      'measured_seconds':f['seconds_this_session'],'peak_allocated_GiB':f['peak_gpu_allocated_bytes_all_conditions']/2**30,
      'peak_reserved_GiB':'','timing_scope':'Final evaluation session, including reference NLL, generation, bootstrap and output writes; excludes model loading.'})
    csv_write(out/'resources.csv',resources)
    (out/'resources.md').write_text('# Recorded resources\n\n'+markdown_table(resources,list(resources[0]))+
      f"\nTotal main-update time: {sum(x['training_seconds'] for x in finished['stages'].values()):.2f} s. Development evaluations: {sum(d['seconds'] for d in dev):.2f} s. Recovery-state writes: {finished['checkpoint_save_seconds']:.2f} s. Generation only (all 1280 outputs): {f['generation_seconds_all_conditions']:.2f} s. The logs do not record one complete queue wall-clock duration.\n\nFeature forwards alone took {feature['vision_forward_seconds_all_completed_shards']:.3f} s; this is not the feature-caching wall time. Five initial AMP overflow attempts were retried on the same sampled batch before any successful update. Frozen weights and inference-only arrays are excluded from trainable counts.\n")
    write(out/'recomputed_summaries.json',summaries)
    return behavior

def figures(task, log, dev, summaries, records):
    out=task/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})
    def boundaries(ax):
        for n in (2250,3750):ax.axvline(n,color='#555555',ls='--',lw=.8)
        ax.set_xlim(0,4500);ax.set_xlabel('Successful update');ax.grid(alpha=.15)
    def save(fig,name):
        fig.savefig(out/(name+'.png'),bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');xs=np.array([d['update'] for d in dev])
    ax=axes[0,0];ax.plot([r['update'] for r in log],[r['loss'] for r in log],alpha=.14,color='#1976a3',lw=.5)
    # Smooth each stage separately so the caption/VQA scale changes do not bleed across boundaries.
    for stage in (1,2,3):
        rows=[r for r in log if r['stage']==stage]
        ax.plot([np.mean([r['update'] for r in rows[i:i+25]]) for i in range(0,len(rows),25)],
                [np.mean([r['loss'] for r in rows[i:i+25]]) for i in range(0,len(rows),25)],color='#1976a3')
    ax.set_title('Training loss (raw and 25-update means)');ax.set_ylabel('Response + EOT cross-entropy')
    ax=axes[0,1];ax.plot(xs,[d['caption_reference_loss'] for d in dev],'o-',color='#1976a3');ax.set_title('Caption development loss');ax.set_ylabel('Reference loss, including EOT')
    for ax,suite in zip(axes[1],('grounding','natural')):
        for c in ('correct','complementary','no_image'):
            ax.plot(xs,[d['suites'][suite]['conditions'][c]['mean_answer_nll'] for d in dev],'o-',color=COLORS[c],label=LABELS[c],ms=3)
        ax.set_title(suite.capitalize()+' development answer NLL');ax.set_ylabel('Content NLL (EOT excluded)');ax.legend(fontsize=8)
    for ax in axes.flat:boundaries(ax)
    fig.suptitle('Three-stage curriculum: caption alignment → visual instruction → natural VQA')
    save(fig,'learning_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,suite in zip(axes,('grounding','natural')):
        ax.plot(xs,[d['suites'][suite]['image_swap_penalty'] for d in dev],'o-',color='#1976a3',ms=4)
        ax.axhline(0,color='#777',lw=.6);ax.set_title(suite.capitalize()+' development image-swap penalty')
        ax.set_ylabel('NLL(partner) − NLL(correct)');boundaries(ax)
    save(fig,'image_swap_penalties')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,suite in zip(axes,('grounding','natural')):
        data=summaries[suite]['groups']['overall']['conditions'];x=np.arange(4)
        means=np.array([100*data[c]['accuracy']['mean'] for c in CONDITIONS])
        lo=np.array([100*data[c]['accuracy']['ci95'][0] for c in CONDITIONS]);hi=np.array([100*data[c]['accuracy']['ci95'][1] for c in CONDITIONS])
        ax.bar(x,means,color=[COLORS[c] for c in CONDITIONS],width=.65)
        ax.errorbar(x,means,yerr=[means-lo,hi-means],fmt='none',color='black',capsize=4,lw=1)
        for i,m in enumerate(means):ax.text(i,hi[i]+3,f'{m:.2f}',ha='center',fontsize=9)
        ax.set_xticks(x,[LABELS[c] for c in CONDITIONS],rotation=16);ax.set_ylim(0,100)
        ax.set_ylabel('Exact match (%)' if suite=='grounding' else 'VQA soft accuracy (%)');ax.set_title(suite.capitalize());ax.grid(axis='y',alpha=.15)
    fig.suptitle('Final Stage 3 model: complete-pair 95% bootstrap intervals');save(fig,'final_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,suite in zip(axes,('grounding','natural')):
        groups=[g for g in summaries[suite]['groups'] if g!='overall'];x=np.arange(len(groups));width=.18
        for i,c in enumerate(CONDITIONS):
            vals=[summaries[suite]['groups'][g]['conditions'][c]['accuracy'] for g in groups]
            m=np.array([v['mean']*100 for v in vals]);lo=np.array([v['ci95'][0]*100 for v in vals]);hi=np.array([v['ci95'][1]*100 for v in vals])
            ax.bar(x+(i-1.5)*width,m,width,label=LABELS[c],color=COLORS[c]);ax.errorbar(x+(i-1.5)*width,m,yerr=[m-lo,hi-m],fmt='none',color='black',capsize=2,lw=.7)
        ax.set_xticks(x,groups);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(suite.capitalize());ax.grid(axis='y',alpha=.15)
    axes[1].legend(fontsize=8);save(fig,'accuracy_by_type')
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,suite in zip(axes,('grounding','natural')):
        rows=records[suite]['complementary'];cats=['Original canonical','Partner canonical','Neither canonical']
        vals=[sum(r['canonical_exact_match'] for r in rows),sum(r['partner_exact_match'] for r in rows),sum(not r['canonical_exact_match'] and not r['partner_exact_match'] for r in rows)]
        ax.bar(cats,np.array(vals)/len(rows)*100,color=['#737b83','#1976a3','#d07728']);ax.set_ylim(0,100)
        ax.set_title(suite.capitalize()+' after complementary swap');ax.set_ylabel('Predictions (%)');ax.tick_params(axis='x',rotation=12)
    save(fig,'swap_answer_behavior')

def render_pair_panel(pair,panels,images,destination,suite):
    """Render at fixed pixel margins; text layout never changes image axes.

    Pixel-measured wrapping prevents long quoted responses from pushing labels
    beyond the canvas, regardless of image aspect ratio or plotting version.
    """
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    from matplotlib import font_manager
    font_path=font_manager.findfont('DejaVu Sans')
    font=ImageFont.truetype(font_path,25);heading=ImageFont.truetype(font_path,31)
    width=1800;margin=40;gap=60;column=(width-2*margin-gap)//2
    measure=ImageDraw.Draw(Image.new('RGB',(1,1)))
    def wrap(text,face,max_width):
        lines=[];line=''
        for word in text.split():
            candidate=(line+' '+word).strip()
            if measure.textlength(candidate,font=face)<=max_width:line=candidate
            else:
                if line:lines.append(line)
                line=''
                for char in word:
                    if line and measure.textlength(line+char,font=face)>max_width:lines.append(line);line=''
                    line+=char
        if line:lines.append(line)
        return '\n'.join(lines)
    title=suite.capitalize()+' / '+pair['question_type']+'\n'+wrap(pair['question'],heading,width-2*margin)
    title_height=measure.multiline_textbbox((0,0),title,font=heading,spacing=8)[3]
    image_y=30+title_height+65;image_height=530;text_y=image_y+image_height+25
    blocks=[];block_heights=[]
    for e,row in panels:
        strings=['References: '+str(dict(collections.Counter(e['answers'])))]
        strings += [LABELS[c]+': '+repr(row['responses'][c])+f"  [score {row['scores'][c]:.2f}]" for c in CONDITIONS]
        strings=[wrap(s,font,column) for s in strings];blocks.append(strings)
        block_heights.append(sum(measure.multiline_textbbox((0,0),s,font=font,spacing=7)[3]+23 for s in strings))
    canvas=Image.new('RGB',(width,text_y+max(block_heights)+35),'white');draw=ImageDraw.Draw(canvas)
    draw.multiline_text((width//2,20),title,font=heading,fill='black',anchor='ma',align='center',spacing=8)
    for i,((e,row),image,strings) in enumerate(zip(panels,images,blocks)):
        x=margin+i*(column+gap)
        caption=f"Image {e['image_id']} · canonical: {e['canonical_answer']}"
        draw.text((x+column//2,image_y-42),caption,font=font,fill='black',anchor='ma')
        fitted=ImageOps.contain(image.convert('RGB'),(column,image_height),method=Image.Resampling.LANCZOS)
        canvas.paste(fitted,(x+(column-fitted.width)//2,image_y+(image_height-fitted.height)//2))
        y=text_y
        for text in strings:
            draw.multiline_text((x,y),text,font=font,fill='black',spacing=7)
            y+=draw.multiline_textbbox((0,0),text,font=font,spacing=7)[3]+23
    canvas.save(destination)

def qualitative(task,records,require_images=False):
    rule=read(task/'splits/qualitative_example_ids.json');pages=[];rows=[]
    image_manifest=task/'data/images/manifest.json';reader=None
    if image_manifest.exists() and all((image_manifest.parent/s['file_name']).is_file() for s in read(image_manifest)['shards']):
        image_module=load_module('task2_image_report',task/'code/task2_prepare_images.py')
        reader=image_module.ImageArchiveReader(image_manifest.parent,read(image_manifest))
    if require_images:assert reader is not None,'Run against the original Task 2 Drive folder to render the saved image pairs.'
    out=task/'figures/qualitative';out.mkdir(parents=True,exist_ok=True)
    try:
        for suite,pair_ids in rule['pairs'].items():
            pairs={p['pair_id']:p for p in read(task/f'splits/{suite}_test.json')}
            lookups={c:{r['example_id']:r for r in records[suite][c]} for c in CONDITIONS}
            for i,pair_id in enumerate(pair_ids,1):
                pair=pairs[pair_id];panels=[]
                for e in pair['examples']:
                    row={'suite':suite,'pair_id':pair_id,'question_type':pair['question_type'],'example_id':e['example_id'],
                         'image_id':e['image_id'],'question':e['question'],'canonical_answer':e['canonical_answer'],'references':e['answers'],
                         'responses':{c:lookups[c][e['example_id']]['response'] for c in CONDITIONS},
                         'scores':{c:lookups[c][e['example_id']]['score'] for c in CONDITIONS}}
                    rows.append(row);panels.append((e,row))
                name=f'{suite}_{i:02d}.png';pages.append(str(Path('figures/qualitative')/name))
                if reader is not None:
                    render_pair_panel(pair,panels,[reader.get(e) for e,row in panels],out/name,suite)
        write(task/'analysis/qualitative_examples.json',{'selection_rule':rule['rule'],'examples':rows})
        write(task/'analysis/qualitative_status.json',{'selection_rule':rule['rule'],'pairs':5,
          'rendered_images':all((task/p).is_file() for p in pages),'rendered_this_run':reader is not None,'pages':pages})
    finally:
        if reader is not None:reader.close()
    return reader is not None

def findings(task,report,summaries,behavior):
    g=summaries['grounding']['groups'];n=summaries['natural']['groups']
    lines=['# Task 2 results and interpretation','',
      'The final Stage 3 checkpoint was evaluated on fixed, image-disjoint local subsets. No final-test score selected the checkpoint. Grounding uses normalized exact match and natural VQA uses the ten-answer soft consensus score. Intervals resample complete pairs 4000 times, reusing pair indices across image conditions.','',
      '## Curriculum','',
      'Caption development loss decreased from 5.8840 before training to 2.5623 after caption alignment and 2.5400 after Stage 3. This is reference-likelihood evidence for caption alignment; no caption-generation evaluation was performed. Grounding development swap penalties were near zero after Stage 1 (0.0179), reached 3.1992 after Stage 2, and ended at 2.1299 after natural VQA specialization. Natural VQA penalties grew from 0.1334 after Stage 1 to 1.1892 after Stage 2 and ended at 1.1419. Visual instruction training introduced substantially stronger image-conditioned answer likelihoods. Later natural specialization did not improve every development measure monotonically. Training losses across stages mix different target lengths and tasks, so their absolute levels are not directly comparable.','',
      '## Final accuracy and grounding','',
      f"Grounding correct-image exact match was {cell(g['overall']['conditions']['correct']['accuracy'],True)}%; complementary swaps scored {cell(g['overall']['conditions']['complementary']['accuracy'],True)}%. The paired advantage was {cell(g['overall']['correct_image_advantages']['accuracy_correct_minus_complementary'],True)} percentage points. Presence was easier than counting: {g['presence']['conditions']['correct']['accuracy']['mean']*100:.2f}% versus {g['counting']['conditions']['correct']['accuracy']['mean']*100:.2f}%.",'',
      f"Natural VQA correct-image soft accuracy was {cell(n['overall']['conditions']['correct']['accuracy'],True)}%; complementary swaps scored {cell(n['overall']['conditions']['complementary']['accuracy'],True)}%. Its paired advantage was {cell(n['overall']['correct_image_advantages']['accuracy_correct_minus_complementary'],True)} percentage points. Correct-image scores were {n['yes/no']['conditions']['correct']['accuracy']['mean']*100:.2f}% yes/no, {n['number']['conditions']['correct']['accuracy']['mean']*100:.2f}% number, and {n['other']['conditions']['correct']['accuracy']['mean']*100:.2f}% other.",'',
      '## What the controls measure','',
      'Correct images improve both generated accuracy and teacher-forced original-reference likelihood. Complementary swaps preserve the question and visual-token count, making their paired comparison stronger evidence of image use than the no-image contrast alone. Mismatched inputs use an unrelated image from the same suite and keep original references fixed. Swap correctness is always scored against the original image references; partner-answer following is reported separately.','']
    for b in behavior:
        lines.extend([f"For {b['suite']}, {100*b['prediction_changed_rate']:.2f}% of answers changed after swapping. Among all swapped predictions, {100*b['swap_partner_canonical_rate']:.2f}% matched the partner canonical answer, {100*b['swap_original_canonical_rate']:.2f}% matched the original canonical answer, and {100*b['swap_neither_canonical_rate']:.2f}% matched neither. These are canonical-string diagnostics; natural VQA soft credit can differ.",''])
    lines.extend(['Both no-image suites scored zero under the required exact/soft answer scoring, with no empty outputs. This does not mean the model produced no relevant words: many responses became explanatory sentences rather than canonical short answers. Grounding had 50/128 capped outputs and natural VQA had 114/192. Omitting 196 visual tokens also shortens the prefix and shifts the model away from its trained image-conditioned input format. The zero score therefore combines lost visual information with format and distribution changes; it is not a clean estimate of language-only knowledge. No substring extraction or post-hoc answer rewriting was applied.','',
      '## Pair-level and qualitative evidence','',
      'Pair accuracy requires both responses to match their original canonical targets. Identical-answer rates count equal normalized responses, including empty responses if present. The complete tables report both metrics alongside reference NLL and generation time. Five illustrative pairs were fixed before predictions: the first sorted pair per question type, excluding grounding pairs flagged in the pre-training review. They illustrate behavior and are not a random estimate of success frequency. Consult the aligned original-image panels and raw responses rather than inferring visual correctness from answer changes alone.','',
      'In the selected fire-hydrant presence pair, correct-image outputs are yes/no and complementary outputs reverse to no/yes, matching the partner-image answers. In the stop-sign counting pair, the model predicts 1/2 against annotation targets 3/4; swapping reverses its predictions to 2/1 but neither original target is recovered. This is image sensitivity without accurate counting.','',
      'The selected natural pairs show failures that aggregate accuracy alone hides. The model answers 2 for both bird-eye images (targets 2/1), closed for both shower curtains (closed/open), and yes for both shorts questions (yes/no). These answers remain unchanged under complementary swaps, so each pair has one canonical hit but zero pair accuracy. The no-image bird response contains the numeral 2 inside a full sentence and receives zero under the required whole-answer scoring; this illustrates the format limitation without changing the scorer.','',
      '## Limits','',*['- '+x for x in report['limits']],
      '- Annotation-derived absence means no recorded object annotation, and counts depend on COCO object instances, occlusion and category conventions. Thirty of 88 reviewed development/test grounding pairs were flagged for possible ambiguity; this is not evidence that all 30 labels are wrong. All examples remain in primary scores.',
      '- A 16-new-token cap can affect verbose outputs, particularly the no-image condition. Timing reflects this A100 run, batch size and implementation; per-answer timings are amortized over generation batches.',
      '- Only the required 196-token recipe was run; the optional 49-token experiment was not performed.',
      '- These notes are an analysis draft for student review and the combined report, not an independently adjudicated annotation study.',''])
    (task/'analysis/RESULTS.md').write_text('\n'.join(lines))

def analyze(task,require_images=False):
    task=Path(task).resolve();(task/'analysis').mkdir(exist_ok=True)
    report,log,dev,summaries,records=audit(task)
    write(task/'analysis/audit.json',report)
    behavior=tables(task,summaries,records,dev);figures(task,log,dev,summaries,records)
    rendered=qualitative(task,records,require_images);findings(task,report,summaries,behavior)
    print(json.dumps({'passed':True,'updates':4500,'presentations':72000,'final_answers':1280,'development_evaluations':12,
      'qualitative_images_rendered':rendered,'analysis':str(task/'analysis')},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True);parser.add_argument('--require-images',action='store_true')
    args=parser.parse_args();analyze(args.task_dir,args.require_images)
