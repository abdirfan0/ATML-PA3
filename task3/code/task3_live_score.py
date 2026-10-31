"""Audit and grade the frozen live queue on CPU, without loading torch/models."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.metadata as metadata, json, math
from pathlib import Path
import numpy as np
from task3_answers import last_boxed_answer, math_equivalent, math_groups
from task3_live_spec import CALLS, STRATEGIES, MODEL_ID, MODEL_REVISION, messages, seed_for

VERSION = 'pa3-task3-live-scoring-v1'

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, d):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp'); temp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n'); temp.replace(p)

def select_parallel(answers):
    """Missing answers remain voting candidates; ties use earliest sample."""
    groups=math_groups(answers)
    counts={g:groups.count(g) for g in groups}
    winner=max(counts,key=counts.get)
    return groups.index(winner), groups

def interval(values, draws):
    a=np.asarray(values,dtype=float)
    lo,hi=np.percentile(a[draws].mean(axis=1),[2.5,97.5])
    return {'mean':float(a.mean()),'ci95':[float(lo),float(hi)]}

def audit(task):
    task=Path(task); root=task/'results/live'; finish=read(root/'finished.json'); assert finish['passed']
    identity=finish['identity']; assets=read(task/'configs/live_assets.json')
    for key,path in [('runner_sha256','configs/live_runner_frozen.json'),('gpu_environment_sha256','configs/live_gpu_environment.json'),('assets_sha256','configs/live_assets.json')]:
        assert sha(task/path)==identity[key],path
    for cfg in ['configs/live_assets.json','configs/live_runner_frozen.json','configs/scorer_frozen.json']:
        for name,digest in read(task/cfg)['source_sha256'].items(): assert sha(task/'code'/name)==digest,name
    for key,path in [('scorer_sha256','configs/scorer_frozen.json'),('references_sha256','data/live_math_references.json'),('prompts_sha256','data/live_prompts.json'),('recipe_sha256','configs/live_recipe.json'),('question_splits_sha256','splits/question_splits.json'),('selected_questions_sha256','splits/live_math_questions.json'),('offline_finished_sha256','results/offline/finished.json')]:
        assert sha(task/path)==assets[key],path
    manifest=read(root/'call_manifest.json'); assert manifest['identity']==identity
    for key,name in [('call_manifest_sha256','call_manifest.json'),('strategy_costs_sha256','strategy_costs.json')]: assert sha(root/name)==finish[key]
    assert len(manifest['files'])==250
    questions=read(task/'data/live_prompts.json')['questions']; assert len(questions)==50
    expected_ids=read(task/'splits/question_splits.json')['datasets']['math']['live_math']
    assert [q['sample_id'] for q in questions]==expected_ids
    references=read(task/'data/live_math_references.json')['references']; assert [r['sample_id'] for r in references]==expected_ids
    saved_costs=read(root/'strategy_costs.json'); assert saved_costs['identity']==identity
    all_calls=[]; costs=[]; seeds=set(); seen=set()
    for index,q in enumerate(questions):
        calls={}
        for call,cap in CALLS.items():
            rel=f'results/live/calls/{index:03d}_{call}.json'; p=task/rel
            assert sha(p)==manifest['files'][rel],rel
            marker=read(p.with_name(p.name+'.complete.json'))
            assert marker=={'identity':identity,'sha256':sha(p)}
            c=read(p); calls[call]=c; seen.add(rel)
            assert c['identity']==identity and not c['smoke']
            assert c['sample_id']==q['sample_id'] and c['call']==call
            assert c['model_id']==MODEL_ID and c['model_revision']==MODEL_REVISION
            assert c['seed']==seed_for(q['sample_id'],call) and c['seed'] not in seeds; seeds.add(c['seed'])
            assert c['max_new_tokens']==cap and 0<c['output_tokens']==len(c['generated_token_ids'])<=cap
            cfg=c['generation_settings']
            for k,v in {'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.,'num_beams':1,'num_return_sequences':1,'use_cache':True,'max_new_tokens':cap}.items(): assert cfg[k]==v,(rel,k)
            assert cfg['eos_token_id']==[151643,151645] and cfg['pad_token_id']==151643
            ended=c['generated_token_ids'][-1] in cfg['eos_token_id']
            retained=c['generated_token_ids'][:-1] if ended else c['generated_token_ids']
            assert c['ended_by_eos']==ended and c['retained_token_ids']==retained and c['retained_output_tokens']==len(retained)
            assert c['reached_token_cap']==(c['output_tokens']==cap)
            assert c['truncated']==(c['output_tokens']==cap and not ended)
            assert c['extracted_answer']==last_boxed_answer(c['response'])
            assert c['extraction_failed']==(c['extracted_answer'] is None)
            req=c['request']; assert req['sample_id']==q['sample_id'] and req['problem']==q['problem']
            assert c['input_tokens']==req['input_tokens']==len(req['input_ids'])
            assert math.isfinite(c['seconds']) and c['seconds']>0
            if call!='revision':
                assert req==q and c.get('initial_call_sha256') is None
        initial=calls['parallel_0']; revision=calls['revision']
        assert revision['request']['messages']==messages(q['problem'],initial['response'])
        assert revision['initial_call_sha256']==manifest['files'][f'results/live/calls/{index:03d}_parallel_0.json']
        assert q['messages']==messages(q['problem'])
        computed={s:{'calls':keys,'input_tokens':sum(calls[k]['input_tokens'] for k in keys),'output_tokens':sum(calls[k]['output_tokens'] for k in keys),'retained_output_tokens':sum(calls[k]['retained_output_tokens'] for k in keys),'seconds':sum(calls[k]['seconds'] for k in keys),'peak_gpu_allocated_bytes':max(calls[k]['peak_gpu_allocated_bytes'] for k in keys)} for s,keys in STRATEGIES.items()}
        assert saved_costs['questions'][index]=={'sample_id':q['sample_id'],'strategies':computed}
        all_calls.append(calls); costs.append(computed)
    assert seen==set(manifest['files'])
    flat=[c for calls in all_calls for c in calls.values()]
    for field,total in [('output_tokens','physical_output_tokens'),('retained_output_tokens','physical_retained_output_tokens')]: assert sum(c[field] for c in flat)==finish[total]
    assert abs(sum(c['seconds'] for c in flat)-finish['physical_generation_seconds'])<1e-6
    assert max(c['peak_gpu_allocated_bytes'] for c in flat)==finish['peak_gpu_allocated_bytes']
    assert finish['physical_calls']==250 and finish['questions']==50
    return questions,references,all_calls,costs,finish

def score(task):
    task=Path(task)
    for name,version in read(task/'configs/scorer_frozen.json')['packages'].items(): assert metadata.version(name)==version,(name,metadata.version(name),version)
    questions,refs,calls,costs,finish=audit(task)
    # All selections are made without references, before any correctness check.
    choices=[]
    for c in calls:
        rid,groups=select_parallel([c[f'parallel_{i}']['extracted_answer'] for i in range(3)])
        choices.append({'single_long':'single_long','parallel':f'parallel_{rid}','sequential':'revision'})
    rows=[]
    for i,(q,ref,c) in enumerate(zip(questions,refs,calls)):
        correct={k:math_equivalent(v['extracted_answer'],ref['gold_answer']) for k,v in c.items()}
        rows.append({'sample_id':q['sample_id'],'gold_answer':ref['gold_answer'],'call_correct':correct,
            'call_extracted_answers':{k:v['extracted_answer'] for k,v in c.items()},
            'parallel_groups':math_groups([c[f'parallel_{j}']['extracted_answer'] for j in range(3)]),
            'selected_calls':choices[i], 'strategies':{s:{'correct':correct[k],'answer':c[k]['extracted_answer'],
                'extraction_failed':c[k]['extraction_failed'],'selected_truncated':c[k]['truncated'],
                'truncated_calls':sum(c[x]['truncated'] for x in STRATEGIES[s]),**costs[i][s]} for s,k in choices[i].items()}})
    draws=np.random.default_rng(6304).integers(0,50,size=(1000,50))
    strategies={}
    for s in STRATEGIES:
        strategies[s]={k:interval([r['strategies'][s][k] for r in rows],draws) for k in ['correct','input_tokens','output_tokens','retained_output_tokens','seconds','peak_gpu_allocated_bytes','extraction_failed','selected_truncated']}
        strategies[s]['correct_questions']=sum(r['strategies'][s]['correct'] for r in rows)
        strategies[s]['total_truncated_calls']=sum(r['strategies'][s]['truncated_calls'] for r in rows)
    differences={}
    for a,b in [('parallel','single_long'),('sequential','single_long'),('sequential','parallel')]:
        differences[f'{a}_minus_{b}']={k:interval([float(r['strategies'][a][k])-float(r['strategies'][b][k]) for r in rows],draws) for k in ['correct','input_tokens','output_tokens','seconds']}
    transition={f'{a}_to_{b}':sum(r['call_correct']['parallel_0']==a and r['call_correct']['revision']==b for r in rows) for a in [False,True] for b in [False,True]}
    summary={'version':VERSION,'passed':True,'questions':50,'physical_calls':250,'gpu_used_for_scoring':False,
      'generation_identity':finish['identity'],'bootstrap':{'unit':'question','resamples':1000,'seed':6304,'paired':True,'interval':'percentile 95%'},
      'strategies':strategies,'paired_differences':differences,'revision_transitions':transition,
      'cost_notes':['Generated output cost includes final EOS; retained output is reported separately.',
        'Parallel means three independent batch-one samples executed serially; time is their summed runtime.',
        'Sequential includes the shared initial call and the full revision prompt/output; the final revision is always selected.',
        'Peak GPU memory is the measured allocated peak per strategy, not the sum across its calls.',
        'Fifty questions give limited precision; intervals describe sampling uncertainty for this fixed run.'],
      'physical_generation_seconds':finish['physical_generation_seconds'],'source_sha256':sha(Path(__file__)),
      'call_manifest_sha256':finish['call_manifest_sha256'],'references_sha256':sha(task/'data/live_math_references.json')}
    out=task/'results/live_scoring';write(out/'per_question.json',rows);write(out/'summary.json',summary)
    with (out/'strategies.csv').open('w',newline='') as f:
        fields=['sample_id','strategy','correct','answer','input_tokens','output_tokens','retained_output_tokens','seconds','peak_gpu_allocated_bytes','extraction_failed','selected_truncated']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for s,d in r['strategies'].items():w.writerow({'sample_id':r['sample_id'],'strategy':s,**{k:d[k] for k in fields[2:]}})
    # Fixed lexicographic examples in each transition, without manual selection.
    examples={}
    for a,b in [(False,True),(True,False),(False,False),(True,True)]:
        ids=sorted(r['sample_id'] for r in rows if r['call_correct']['parallel_0']==a and r['call_correct']['revision']==b)[:2]
        examples[f'{a}_to_{b}']=[{'sample_id':sid,'problem':questions[i]['problem'],'gold_answer':refs[i]['gold_answer'],
            'initial_response':calls[i]['parallel_0']['response'],'revision_response':calls[i]['revision']['response']} for sid in ids for i in [next(j for j,q in enumerate(questions) if q['sample_id']==sid)]]
    write(out/'revision_examples.json',examples)
    write(out/'finished.json',{'passed':True,'version':VERSION,'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='finished.json'}})
    print(json.dumps({'passed':True,'questions':50,'correct_out_of_50':{s:d['correct_questions'] for s,d in strategies.items()},'revision_transitions':transition,'gpu_used':False},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-dir',type=Path,required=True);score(p.parse_args().task_dir)
