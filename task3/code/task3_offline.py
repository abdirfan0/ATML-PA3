"""CPU selection, signal diagnostics, development-only stopping and held-out reports."""
from __future__ import annotations
import argparse,collections,concurrent.futures,csv,gzip,hashlib,importlib.metadata as metadata,json,multiprocessing as mp,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from task3_answers import clean,math_groups,math_equivalent,parse_math,mathvista_answer,mathvista_reference,NO_ANSWER
from task3_pool import PreparedPool,sha256
from task3_selection import BUDGETS,select_prefix,visual_scores,stop_at,policy_grid,choose_policy

VERSION='pa3-task3-offline-v1'
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def write_gzip(p,value):
    temp=p.with_name(p.name+'.tmp')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'),allow_nan=False)
    temp.replace(p)
def load_gzip(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return json.load(f)
def correlation(a,b,rank=False):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if rank:a=rankdata(a);b=rankdata(b)
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den>0 else None

def question_analysis(payload):
    label,question,orders,meta,trajectory_ids=payload;info=question['metadata'];rows=info['rollouts']
    assert [r['rollout_id'] for r in rows]==list(range(32))
    answers=[r['extracted_answer'] for r in rows];gold=info['gold_answer'];lengths=np.array([r['n_generated_text_tokens'] for r in rows])
    assert (lengths>0).all(),'Zero-token policy must be specified before processing a different export.'
    means_logp=np.array([r['mean_logp'] for r in rows]);means_kl=np.array([r['mean_kl'] for r in rows])
    offsets=question['offsets'];kl_tokens=[question['kl'][offsets[i]:offsets[i+1]] for i in range(32)]
    vision_tokens=[question['vision'][offsets[i]:offsets[i+1]] for i in range(32)] if label=='mathvista' else None
    if label=='math':
        normalized=None
    else:
        normalized=[mathvista_answer(a,meta) for a in answers]
    group_cache={};all_prefixes=[];visual32=None
    for order in orders:
        prefixes={}
        for n in BUDGETS:
            ids=order[:n];idx=np.array(ids)
            if label=='math':
                subset=[clean(answers[i]) for i in ids];cache_key=tuple(sorted(set(subset)-{None}))
                if cache_key not in group_cache:
                    labels=math_groups(list(cache_key));group_cache[cache_key]=dict(zip(cache_key,labels))
                groups=[NO_ANSWER if a is None else group_cache[cache_key][a] for a in subset]
            else:groups=[normalized[i] for i in ids]
            visual=visual_scores([kl_tokens[i] for i in ids],[vision_tokens[i] for i in ids]) if vision_tokens is not None else None
            selected=select_prefix(groups,lengths[idx],means_logp[idx],means_kl[idx],visual)
            choices={method:int(ids[i]) for method,i in selected['choices'].items()}
            selected.pop('choices')
            selected.update({'candidate_ids':ids,'choices':choices,'token_cost':int(lengths[idx].sum())})
            if visual is not None:selected['visconf_percentiles']=visual['percentiles']
            prefixes[str(n)]=selected
        all_prefixes.append(prefixes)
    # Diagnostics use the complete pool retrospectively. This information is not
    # passed into select_prefix/stop_at or any score normalization at smaller N.
    # Grade only after all selection/stopping inputs have been constructed.
    if label=='math':
        correct=[math_equivalent(a,gold) for a in answers]
        parse_failures=[a for a in sorted({clean(a) for a in answers}-{None}) if not parse_math(a)]
        gold_parseable=bool(parse_math(gold));canonical_gold=None
    else:
        canonical_gold=mathvista_reference(gold,meta)
        correct=[a!=NO_ANSWER and a==canonical_gold for a in normalized];parse_failures=[];gold_parseable=None
    scores={'length':lengths.astype(float),'logp':means_logp,'self_certainty':means_kl}
    if label=='mathvista':
        visual32=visual_scores(kl_tokens,vision_tokens)
        scores.update({k:visual32[v] for k,v in [('vision','vision'),('confidence','confidence'),('visconf','visconf'),('tokenwise','tokenwise')]})
    order0=orders[0];quartiles={};associations={}
    for name,values in scores.items():
        top=sorted(order0,key=lambda i:-values[i])[:8];bottom=sorted(order0,key=lambda i:values[i])[:8]
        quartiles[name]={'top_accuracy':float(np.mean([correct[i] for i in top])),'bottom_accuracy':float(np.mean([correct[i] for i in bottom])),
          'accuracy_gap':float(np.mean([correct[i] for i in top])-np.mean([correct[i] for i in bottom])),
          'top_mean_length':float(lengths[top].mean()),'bottom_mean_length':float(lengths[bottom].mean()),'top_rollout_ids':top,'bottom_rollout_ids':bottom}
        associations[name]={'spearman_correctness':correlation(values,correct,True),'spearman_length':correlation(values,lengths,True)}
    diagnostics={'quartiles':quartiles,'associations':associations,'math_parse_failed_unique_answers':parse_failures,'math_gold_parseable':gold_parseable}
    trajectories=[]
    if visual32 is not None:
        correlations=[correlation(c,v) for c,v in zip(visual32['calibrated_tokens'],vision_tokens)]
        raw_correlations=[correlation(k,v) for k,v in zip(kl_tokens,vision_tokens)]
        valid=[v for v in correlations if v is not None];raw_valid=[v for v in raw_correlations if v is not None]
        diagnostics.update({'confidence_vision_correlation_mean':float(np.mean(valid)) if valid else None,
          'confidence_vision_defined_rollouts':len(valid),'raw_kl_vision_correlation_mean':float(np.mean(raw_valid)) if raw_valid else None,
          'mean_tokenwise_minus_product':float(np.mean(visual32['tokenwise']-visual32['visconf'])),
          'mean_token_covariance':float(visual32['covariance'].mean()),
          'per_rollout_confidence_vision_correlation':correlations})
    for rid in trajectory_ids:
        record={'rollout_id':rid,'generated_text':rows[rid]['generated_text'],'extracted_answer':answers[rid],
          'correct':correct[rid],'kl_u_p':kl_tokens[rid].tolist()}
        if visual32 is not None:record.update({'confidence':visual32['calibrated_tokens'][rid].tolist(),'vision':vision_tokens[rid].tolist()})
        trajectories.append(record)
    candidates=[{'rollout_id':i,'extracted_answer':answers[i],'normalized_answer':normalized[i] if normalized is not None else None,
      'correct':bool(correct[i]),'tokens':int(lengths[i]),'capped':rows[i]['hit_max_new_tokens'],
      'mean_logp':float(means_logp[i]),'mean_kl':float(means_kl[i])} for i in range(32)]
    return {'sample_id':info['sample_id'],'dataset':label,'question_text':info['question_text'],'gold_answer':gold,
      'canonical_gold':canonical_gold,'candidates':candidates,'orders':all_prefixes,'diagnostics':diagnostics,'trajectories':trajectories}

def outcome(q,prefix,method):
    rid=prefix['choices'][method];candidate=q['candidates'][rid]
    first=method=='FirstAns';pool_oracle=any(q['candidates'][i]['correct'] for i in prefix['candidate_ids'])
    oracle=candidate['correct'] if first else pool_oracle
    return {'accuracy':float(candidate['correct']),'token_cost':candidate['tokens'] if first else prefix['token_cost'],
      'rollouts':1 if first else len(prefix['candidate_ids']),'oracle_coverage':float(oracle),
      'pool_oracle_coverage':float(pool_oracle),'distinct_groups':prefix['distinct_groups'],
      'sampling_failure':float(not oracle),'selection_failure':float(oracle and not candidate['correct']),
      'selected_rollout_id':rid}

def policy_outcomes(q,setting):
    result=[]
    for order_index,prefixes in enumerate(q['orders']):
        n=stop_at(prefixes,setting['family'],setting['tau'],setting['gamma'] or 0.)
        row=outcome(q,prefixes[str(n)],'SC');row['stop_n']=n;row['order_index']=order_index
        row['incorrect_early_stop']=bool(n<32 and not row['accuracy'])
        row['fixed32_would_recover']=bool(row['incorrect_early_stop'] and outcome(q,prefixes['32'],'SC')['accuracy'])
        result.append(row)
    return result

def select_policies(dev_questions):
    baseline=float(np.mean([np.mean([outcome(q,o['32'],'SC')['accuracy'] for o in q['orders']]) for q in dev_questions]))
    result={'fixed32_sc_development_accuracy':baseline,'families':{}}
    for family in ('agreement','quality'):
        evaluated=[]
        for setting in policy_grid(family):
            outcomes=[policy_outcomes(q,setting) for q in dev_questions]
            evaluated.append({'setting':setting,'accuracy':float(np.mean([np.mean([r['accuracy'] for r in rows]) for rows in outcomes])),
              'token_cost':float(np.mean([np.mean([r['token_cost'] for r in rows]) for rows in outcomes]))})
        result['families'][family]={'selected':choose_policy(evaluated,baseline),'grid':evaluated}
    return result

def interval(values,indices):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'ci95':np.percentile(a[indices].mean(axis=1),[2.5,97.5]).tolist()}

def summarize(label,questions,policy):
    # One row per question after averaging its ten saved candidate orders.
    indices=np.random.default_rng(6304).integers(len(questions),size=(1000,len(questions)))
    fixed=[];vectors={}
    methods=list(questions[0]['orders'][0]['1']['choices'])
    metric_names=['accuracy','token_cost','rollouts','oracle_coverage','pool_oracle_coverage','distinct_groups','sampling_failure','selection_failure']
    for n in BUDGETS:
        for method in methods:
            rows=[[outcome(q,o[str(n)],method) for o in q['orders']] for q in questions]
            means={metric:np.array([np.mean([r[metric] for r in values]) for values in rows]) for metric in metric_names}
            vectors[(n,method)]=means['accuracy'];record={'dataset':label,'budget':n,'method':method,'questions':len(questions)}
            record.update({k:float(v.mean()) for k,v in means.items()});record['accuracy_ci95']=interval(means['accuracy'],indices)['ci95'];fixed.append(record)
    for record in fixed:
        n=record['budget'];method=record['method']
        record['difference_vs_SC']=interval(vectors[(n,method)]-vectors[(n,'SC')],indices)
        record['difference_vs_FirstAns']=interval(vectors[(n,method)]-vectors[(n,'FirstAns')],indices)
    adaptive=[];raw_adaptive={};cases=[]
    for family in ('agreement','quality'):
        setting=policy['families'][family]['selected']['setting'];rows=[policy_outcomes(q,setting) for q in questions]
        accuracy=np.array([np.mean([r['accuracy'] for r in values]) for values in rows])
        distribution=collections.Counter(r['stop_n'] for values in rows for r in values)
        record={'dataset':label,'family':family,'setting':setting,'questions':len(questions),
          'accuracy':interval(accuracy,indices),'difference_vs_fixed32_SC':interval(accuracy-vectors[(32,'SC')],indices),
          'token_cost':float(np.mean([np.mean([r['token_cost'] for r in values]) for values in rows])),
          'mean_rollouts':float(np.mean([np.mean([r['stop_n'] for r in values]) for values in rows])),
          'stop_counts':{str(n):distribution[n] for n in (4,8,16,32)},
          'stop_fractions':{str(n):distribution[n]/(len(questions)*10) for n in (4,8,16,32)},
          'incorrect_early_stop_rate':float(np.mean([r['incorrect_early_stop'] for values in rows for r in values])),
          'fixed32_recoverable_early_stop_rate':float(np.mean([r['fixed32_would_recover'] for values in rows for r in values]))}
        buckets={}
        for group,low,high in [('none',0,0),('low',0,.25),('middle',.25,.75),('high',.75,1.)]:
            kept=[values for q,values in zip(questions,rows) if (sum(r['correct'] for r in q['candidates'])/32==0 if group=='none' else low<sum(r['correct'] for r in q['candidates'])/32<=high)]
            buckets[group]={'questions':len(kept),'mean_rollouts':float(np.mean([r['stop_n'] for values in kept for r in values])) if kept else None,
              'accuracy':float(np.mean([r['accuracy'] for values in kept for r in values])) if kept else None}
        record['retrospective_full_pool_correct_fraction_buckets']=buckets;adaptive.append(record)
        raw_adaptive[family]=[{'sample_id':q['sample_id'],'outcomes':values} for q,values in zip(questions,rows)]
        for q,values in sorted(zip(questions,rows),key=lambda x:x[0]['sample_id']):
            found=next((r for r in values if r['fixed32_would_recover']),None)
            if found:cases.append({'kind':'premature_stop_recoverable','family':family,'sample_id':q['sample_id'],**found});break
    for kind,check in [('SC_wrong_SelfCertBest_correct',lambda a,b:not a and b),('SC_correct_SelfCertBest_wrong',lambda a,b:a and not b)]:
        for q in sorted(questions,key=lambda x:x['sample_id']):
            p=q['orders'][0]['32'];sc=outcome(q,p,'SC');best=outcome(q,p,'SelfCertBest')
            if check(sc['accuracy'],best['accuracy']):
                cases.append({'kind':kind,'sample_id':q['sample_id'],'order_index':0,'SC_rollout':sc['selected_rollout_id'],'SelfCertBest_rollout':best['selected_rollout_id']});break
    signal={}
    for score in questions[0]['diagnostics']['quartiles']:
        gaps=[q['diagnostics']['quartiles'][score]['accuracy_gap'] for q in questions]
        assoc={}
        for kind in ('spearman_correctness','spearman_length'):
            values=[q['diagnostics']['associations'][score][kind] for q in questions];valid=[v for v in values if v is not None]
            assoc[kind]={'mean':float(np.mean(valid)) if valid else None,'defined_questions':len(valid),'undefined_questions':len(values)-len(valid)}
        signal[score]={'quartile_accuracy_gap':interval(gaps,indices),**assoc}
    visual={}
    if label=='mathvista':
        for key in ['confidence_vision_correlation_mean','raw_kl_vision_correlation_mean','mean_tokenwise_minus_product','mean_token_covariance']:
            values=[q['diagnostics'][key] for q in questions];valid=[v for v in values if v is not None]
            visual[key]={'mean':float(np.mean(valid)) if valid else None,'defined_questions':len(valid)}
    summary={'dataset':label,'held_out_questions':len(questions),'orders_per_question':10,'bootstrap_resamples':1000,
      'bootstrap_unit':'question after averaging its ten orders','fixed':fixed,'adaptive':adaptive,'signals':signal,'visual_signals':visual,
      'math_parse_audit':{'questions_with_unparseable_gold':sum(q['diagnostics']['math_gold_parseable'] is False for q in questions),
        'unique_per_question_unparseable_predictions':sum(len(q['diagnostics']['math_parse_failed_unique_answers']) for q in questions)},
      'normalization_audit':{'candidate_rows':32*len(questions),
        'raw_missing_extractions':sum(clean(c['extracted_answer']) is None for q in questions for c in q['candidates']),
        'normalized_no_answer':sum((c['normalized_answer']==NO_ANSWER if label=='mathvista' else clean(c['extracted_answer']) is None) for q in questions for c in q['candidates']),
        'capped_rollouts':sum(c['capped'] for q in questions for c in q['candidates'])},
      'case_selection_rule':'Lexicographically first held-out question meeting each stated failure/disagreement condition; earliest qualifying saved order for stopping; descriptive post-hoc examples.',
      'cases':cases}
    return summary,raw_adaptive

def get_identity(task):
    scorer=read(task/'configs/scorer_frozen.json');preparation=read(task/'configs/preparation_frozen.json')
    assert read(task/'results/data_preparation.json')['passed']
    assert sha256(task/'configs/scorer_frozen.json')==preparation['scorer_sha256']
    assert read(task/'data/prepared/manifest.json')['identity']==preparation
    assert sha256(task/'configs/supplied_data_inventory.json')==preparation['inventory_sha256']
    for name,version in scorer['packages'].items():assert metadata.version(name)==version,(name,metadata.version(name),version)
    for record in (scorer,preparation):
        for name,value in record['source_sha256'].items():assert sha256(task/'code'/name)==value,name
    assert sha256(task/'splits/question_splits.json')==preparation['question_splits_sha256']
    assert sha256(task/'splits/candidate_orders.json')==preparation['candidate_orders_sha256']
    asset=task/'assets/mathvista'/scorer['mathvista_revision']/'testmini.json';assert sha256(asset)==scorer['mathvista_metadata_sha256']
    manifest=read(task/'data/prepared/manifest.json')
    for label,record in manifest['datasets'].items():assert sha256(task/'data/prepared'/label/'manifest.json')==record['manifest_sha256']
    return {'version':VERSION,'scorer_sha256':sha256(task/'configs/scorer_frozen.json'),
      'preparation_sha256':sha256(task/'configs/preparation_frozen.json'),'prepared_manifest_sha256':sha256(task/'data/prepared/manifest.json'),
      'source_sha256':{n:sha256(task/'code'/n) for n in ['task3_selection.py','task3_offline.py','test_task3_offline_cpu.py']},
      'quartile_policy':'8 highest/lowest candidates per question, ties by saved order 0; full-pool retrospective diagnostics',
      'correlations':'Spearman across candidates within question; Pearson confidence/vision across tokens within rollout then mean within question; undefined correlations retained as null and counted',
      'rwc':'Exact integer units proportional to prescribed descending-average-rank weights; vote ties by earliest supporting candidate',
      'zero_tokens':'This export was audited to have none; fail if that assumption changes.'}

def main(task,workers):
    started=time.perf_counter();identity=get_identity(task);root=task/'results/offline';root.mkdir(parents=True,exist_ok=True)
    frozen_path=task/'configs/offline_frozen.json'
    if frozen_path.exists():assert read(frozen_path)==identity,'Offline code or inputs changed.'
    else:write(frozen_path,identity)
    orders=read(task/'splits/candidate_orders.json')['datasets'];splits=read(task/'splits/question_splits.json')['datasets']
    examples=read(task/'splits/trajectory_examples.json')['datasets'];scorer=read(task/'configs/scorer_frozen.json')
    mv=read(task/'assets/mathvista'/scorer['mathvista_revision']/'testmini.json');all_questions={}
    for label in ('math','mathvista'):
        directory=root/label;directory.mkdir(exist_ok=True);questions=[];traj={r['sample_id']:r['rollout_ids'] for r in examples[label]}
        ids=sorted(orders[label]);print(f'Processing {label}: {len(ids)} questions with {workers} CPU workers.',flush=True)
        with PreparedPool(task,label) as pool,concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as executor:
            for start in range(0,len(ids),50):
                path=directory/f'questions_{start//50:04d}.json.gz';marker=path.with_name(path.name+'.complete.json')
                if marker.exists():
                    mark=read(marker);assert mark['identity']==identity and sha256(path)==mark['sha256']
                    saved=load_gzip(path);assert [q['sample_id'] for q in saved['questions']]==ids[start:start+50]
                    questions.extend(saved['questions']);print(f'{label}: reused {start+50}/{len(ids)}',flush=True);continue
                payloads=[(label,pool.get(qid),orders[label][qid],mv[qid] if label=='mathvista' else None,traj.get(qid,[])) for qid in ids[start:start+50]]
                completed=[];chunk_start=time.perf_counter()
                for row in executor.map(question_analysis,payloads,chunksize=1):
                    completed.append(row)
                    if len(completed)%10==0:print(f'{label}: {start+len(completed)}/{len(ids)} analyzed',flush=True)
                write_gzip(path,{'identity':identity,'questions':completed})
                write(marker,{'identity':identity,'sha256':sha256(path),'seconds':time.perf_counter()-chunk_start,'questions':len(completed)})
                questions.extend(completed)
        all_questions[label]={q['sample_id']:q for q in questions}
    # Only development labels enter threshold selection. The chosen settings are
    # persisted before any held-out summary or figure is produced.
    selected={label:select_policies([all_questions[label][qid] for qid in splits[label]['development']]) for label in ('math','mathvista')}
    selection={'identity':identity,'rule':'Lowest development mean token cost within 1 percentage point of fixed-32 SC; ties higher accuracy then smaller tau/gamma; always-32 fallback.',
      'datasets':selected,'development_ids_sha256':hashlib.sha256(json.dumps({l:splits[l]['development'] for l in splits},sort_keys=True).encode()).hexdigest()}
    policy_path=task/'configs/adaptive_policies_frozen.json'
    if policy_path.exists():assert read(policy_path)==selection,'Selected policy changed; refusing to overwrite.'
    else:write(policy_path,selection)
    for label in ('math','mathvista'):
        heldout=[all_questions[label][qid] for qid in splits[label]['evaluation']]
        summary,raw=summarize(label,heldout,selected[label]);summary['identity']=identity;summary['frozen_policy_sha256']=sha256(policy_path)
        write(root/f'{label}_summary.json',summary);write_gzip(root/f'{label}_adaptive_per_question.json.gz',raw)
        columns=[k for k in summary['fixed'][0] if k not in ('accuracy_ci95','difference_vs_SC','difference_vs_FirstAns')]
        with (root/f'{label}_fixed.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in summary['fixed'])
        print(label+': held-out summary saved.',flush=True)
    result={'passed':True,'identity':identity,'datasets':2,'questions_each':1000,'rollouts_each':32000,'held_out_questions_each':800,
      'orders_per_question':10,'bootstrap_resamples':1000,'policy_sha256':sha256(policy_path),'gpu_used':False,
      'elapsed_seconds_this_session':time.perf_counter()-started,
      'note':'CPU selection/signal/adaptive analysis complete. Live 50-question generation, final plots/notebook and repository organization remain.'}
    write(root/'finished.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();assert 1<=args.workers<=4;main(args.task_dir,args.workers)
