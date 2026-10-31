"""Recheck saved Task 3 evidence and render CPU-only tables and figures."""
from __future__ import annotations
import argparse, csv, gzip, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from task3_live_score import read,write,sha,score
from task3_offline import select_policies,summarize

LABELS={'math':'MATH · Qwen2.5-3B','mathvista':'MathVista · Qwen2.5-VL-3B'}
COLORS=['#294c7a','#d67c34','#39866c','#9866a4','#c65058','#6694ae']
BUDGETS=[1,2,4,8,16,32]

def close(a,b,path='root'):
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:close(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):close(x,y,path+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-11),(path,a,b)
    else:assert a==b,(path,a,b)

def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in row)+' |' for row in rows)+'\n'

def csvfile(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def savefig(task,name,fig):
    fig.savefig(task/'figures'/f'{name}.png',dpi=180,bbox_inches='tight',facecolor='white');plt.close(fig)

def getfixed(summary,method):return sorted([r for r in summary['fixed'] if r['method']==method],key=lambda r:r['budget'])

def audit_offline(task):
    finished=read(task/'results/offline/finished.json');assert finished['passed']
    identity=finished['identity'];assert read(task/'configs/offline_frozen.json')==identity
    for name,digest in identity['source_sha256'].items():assert sha(task/'code'/name)==digest
    for key,path in [('scorer_sha256','configs/scorer_frozen.json'),('preparation_sha256','configs/preparation_frozen.json')]:assert sha(task/path)==identity[key]
    prep=read(task/'configs/preparation_frozen.json')
    for key,path in [('inventory_sha256','configs/supplied_data_inventory.json'),('question_splits_sha256','splits/question_splits.json'),('candidate_orders_sha256','splits/candidate_orders.json')]:assert sha(task/path)==prep[key]
    for name,digest in prep['source_sha256'].items():assert sha(task/'code'/name)==digest
    policies=read(task/'configs/adaptive_policies_frozen.json');assert sha(task/'configs/adaptive_policies_frozen.json')==finished['policy_sha256']
    orders=read(task/'splits/candidate_orders.json')['datasets'];splits=read(task/'splits/question_splits.json')['datasets']
    summaries={};all_questions={}
    for label in LABELS:
        qs=[]
        for path in sorted((task/'results/offline'/label).glob('questions_*.json.gz')):
            mark=read(path.with_name(path.name+'.complete.json'));assert mark['sha256']==sha(path) and mark['identity']==identity
            chunk=json.loads(gzip.decompress(path.read_bytes()));assert chunk['identity']==identity;qs.extend(chunk['questions'])
        assert len(qs)==1000 and len({q['sample_id'] for q in qs})==1000
        byid={q['sample_id']:q for q in qs};s=splits[label]
        expected=np.random.default_rng(6304).permutation(sorted(byid)).tolist()
        assert s['development']==expected[:200] and s['evaluation']==expected[200:]
        for q in qs:
            assert [c['rollout_id'] for c in q['candidates']]==list(range(32)) and len(q['orders'])==10
            for order_index,prefixes in enumerate(q['orders']):
                for n in BUDGETS:
                    p=prefixes[str(n)];assert p['candidate_ids']==orders[label][q['sample_id']][order_index][:n]
                    assert p['token_cost']==sum(q['candidates'][i]['tokens'] for i in p['candidate_ids'])
                    assert all(i in p['candidate_ids'] for i in p['choices'].values())
        selected=select_policies([byid[qid] for qid in s['development']]);close(selected,policies['datasets'][label])
        actual,raw=summarize(label,[byid[qid] for qid in s['evaluation']],selected)
        actual.update(identity=identity,frozen_policy_sha256=sha(task/'configs/adaptive_policies_frozen.json'))
        close(actual,read(task/f'results/offline/{label}_summary.json'))
        close(raw,json.loads(gzip.decompress((task/f'results/offline/{label}_adaptive_per_question.json.gz').read_bytes())))
        summaries[label]=actual;all_questions[label]=byid
        print(label+': 1000 question records, fixed/adaptive metrics and paired intervals verified.',flush=True)
    return summaries,all_questions

def figures(task,summaries,questions,live):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold','axes.grid':True,'grid.alpha':.18,'legend.frameon':False})
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(label,s) in enumerate(summaries.items()):
        sc=getfixed(s,'SC');first=getfixed(s,'FirstAns')
        ax=axes[row,0]
        for vals,key,name,color in [(first,'accuracy','First answer',COLORS[0]),(sc,'accuracy','Self-consistency',COLORS[1]),(sc,'oracle_coverage','Oracle coverage',COLORS[2])]:ax.plot(BUDGETS,[100*r[key] for r in vals],'-o',label=name,color=color,ms=4)
        ax.set(title=LABELS[label],ylabel='Accuracy / coverage (%)',xlabel='Candidate budget');ax.set_xscale('log',base=2);ax.set_xticks(BUDGETS,labels=BUDGETS);ax.legend(fontsize=8)
        axes[row,1].plot(BUDGETS,[r['distinct_groups'] for r in sc],'-o',color=COLORS[0]);axes[row,1].set(xlabel='Candidate budget',ylabel='Mean distinct answer groups',title='Diversity of revealed answers');axes[row,1].set_xscale('log',base=2);axes[row,1].set_xticks(BUDGETS,labels=BUDGETS)
        axes[row,2].stackplot(BUDGETS,[100*r['sampling_failure'] for r in sc],[100*r['selection_failure'] for r in sc],labels=['No correct candidate','Correct available, missed'],colors=[COLORS[0],COLORS[1]],alpha=.8);axes[row,2].set(xlabel='Candidate budget',ylabel='Fraction of questions (%)',title='Self-consistency errors');axes[row,2].set_xscale('log',base=2);axes[row,2].set_xticks(BUDGETS,labels=BUDGETS);axes[row,2].legend(fontsize=8)
    savefig(task,'sampling_selection',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,(label,s) in zip(axes,summaries.items()):
        records=[r for r in s['fixed'] if r['budget']==32];y=np.arange(len(records));v=np.array([r['accuracy'] for r in records]);ci=np.array([r['accuracy_ci95'] for r in records]);ax.errorbar(v*100,y,xerr=np.array([v-ci[:,0],ci[:,1]-v])*100,fmt='o',capsize=3,color=COLORS[0]);ax.set_yticks(y,[r['method'] for r in records]);ax.invert_yaxis();ax.set(title=LABELS[label]+' · N = 32',xlabel='Accuracy (%) · question-bootstrap 95% CI')
    savefig(task,'quality_comparison',fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,(label,s) in enumerate(summaries.items()):
        names=list(s['signals']);y=np.arange(len(names));g=np.array([s['signals'][n]['quartile_accuracy_gap']['mean'] for n in names]);ci=np.array([s['signals'][n]['quartile_accuracy_gap']['ci95'] for n in names])
        axes[row,0].errorbar(g*100,y,xerr=np.maximum(0,np.array([g-ci[:,0],ci[:,1]-g]))*100,fmt='o',color=COLORS[0],capsize=3)
        for col,key in [(1,'spearman_correctness'),(2,'spearman_length')]:axes[row,col].barh(y,[s['signals'][n][key]['mean'] for n in names],color=COLORS[col])
        for col in range(3):axes[row,col].set_yticks(y,names);axes[row,col].invert_yaxis();axes[row,col].axvline(0,color='gray',lw=.7);axes[row,col].set_title(LABELS[label])
        axes[row,0].set_xlabel('Top − bottom quartile accuracy (pp)');axes[row,1].set_xlabel('Mean within-question Spearman ρ\nwith correctness (defined questions)');axes[row,2].set_xlabel('Mean within-question Spearman ρ\nwith length (defined questions)')
    savefig(task,'signal_associations',fig)
    s=summaries['mathvista'];held=[questions['mathvista'][qid] for qid in read(task/'splits/question_splits.json')['datasets']['mathvista']['evaluation']]
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for m,c in [('VisConfBest',COLORS[0]),('TokenwiseBest',COLORS[1]),('VisConfRWC',COLORS[2]),('TokenwiseRWC',COLORS[3])]:axes[0].plot(BUDGETS,[100*r['accuracy'] for r in getfixed(s,m)],'-o',label=m,color=c,ms=3)
    axes[0].set(xlabel='Candidate budget',ylabel='Accuracy (%)',title='Rollout product vs tokenwise fusion');axes[0].set_xscale('log',base=2);axes[0].set_xticks(BUDGETS,labels=BUDGETS);axes[0].legend(fontsize=7)
    for ax,key,title in [(axes[1],'confidence_vision_correlation_mean','Confidence–vision relationship'),(axes[2],'mean_token_covariance','Tokenwise minus rollout product')]:
        values=[q['diagnostics'][key] for q in held if q['diagnostics'][key] is not None];ax.hist(values,bins=30,color=COLORS[0],alpha=.85);ax.axvline(np.mean(values),color=COLORS[1],label=f'Mean {np.mean(values):.4f}');ax.set(title=title,xlabel='Per-question mean across rollouts',ylabel='Questions');ax.legend(fontsize=8)
    savefig(task,'visual_fusion',fig)
    trajectories=[(q,t) for q in held for t in q['trajectories']]
    fig,axes=plt.subplots(3,2,figsize=(12,8),layout='constrained')
    assert len(trajectories)==6
    for ax,(q,t) in zip(axes.flat,trajectories):
        x=np.arange(1,len(t['confidence'])+1);ax.plot(x,t['confidence'],label='Calibrated confidence',color=COLORS[0],lw=.9);ax.plot(x,t['vision'],label='Visual engagement',color=COLORS[1],lw=.9);ax.set(title=f"Question {q['sample_id']} · rollout {t['rollout_id']} · correct={t['correct']}",xlabel='Retained generated token',ylabel='Signal value',ylim=(-.03,1.03));ax.legend(fontsize=7)
    savefig(task,'fixed_trajectories',fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,(label,s) in enumerate(summaries.items()):
        sc=getfixed(s,'SC');axes[row,0].plot([r['token_cost'] for r in sc],[100*r['accuracy'] for r in sc],'-o',label='Fixed-N SC',color=COLORS[0]);
        for r in sc:axes[row,0].annotate(str(r['budget']),(r['token_cost'],100*r['accuracy']),xytext=(4,5),textcoords='offset points',fontsize=7)
        for i,r in enumerate(s['adaptive']):
            axes[row,0].errorbar(r['token_cost'],r['accuracy']['mean']*100,yerr=[[100*(r['accuracy']['mean']-r['accuracy']['ci95'][0])],[100*(r['accuracy']['ci95'][1]-r['accuracy']['mean'])]],fmt='s',color=COLORS[i+1],capsize=3,label=r['family'])
            axes[row,1].bar(np.arange(4)+(i-.5)*.32,[100*r['stop_fractions'][str(n)] for n in [4,8,16,32]],width=.32,label=r['family'],color=COLORS[i+1])
        axes[row,0].set(title=LABELS[label],xlabel='Mean retained generated-token cost',ylabel='Accuracy (%)');axes[row,0].legend(fontsize=8)
        axes[row,1].set_xticks(range(4),[4,8,16,32]);axes[row,1].set(title='Frozen policies · stopping distribution',xlabel='Completed rollouts at stop',ylabel='Question-order outcomes (%)');axes[row,1].legend(fontsize=8)
    savefig(task,'adaptive_inference',fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,(label,s) in zip(axes,summaries.items()):
        for i,r in enumerate(s['adaptive']):
            buckets=r['retrospective_full_pool_correct_fraction_buckets'];ax.plot(range(4),[buckets[k]['mean_rollouts'] for k in ['none','low','middle','high']],'-o',color=COLORS[i+1],label=r['family'])
        ax.set_xticks(range(4),['0','(0, .25]','(.25, .75]','(.75, 1]']);ax.set(title=LABELS[label],xlabel='Correct fraction of full pool · retrospective only',ylabel='Mean consumed rollouts',ylim=(0,33));ax.legend()
    savefig(task,'adaptive_retrospective',fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');names=list(STRATEGY_NAMES)
    specs=[('correct',100,'Accuracy (%)'),('input_tokens',1,'Mean input tokens'),('output_tokens',1,'Mean generated output tokens'),('seconds',1,'Mean inference time (s)'),('extraction_failed',100,'Selected-answer extraction failure (%)'),('peak_gpu_allocated_bytes',1/2**30,'Mean allocated GPU peak (GiB)')]
    for ax,(key,scale,title) in zip(axes.flat,specs):
        v=np.array([live['strategies'][s][key]['mean'] for s in names])*scale;ci=np.array([live['strategies'][s][key]['ci95'] for s in names])*scale
        ax.bar(range(3),v,color=COLORS[:3],yerr=np.maximum(0,np.array([v-ci[:,0],ci[:,1]-v])),capsize=3);ax.set_xticks(range(3),[STRATEGY_NAMES[s] for s in names],fontsize=8);ax.set_title(title);ax.set_ylim(bottom=0)
    fig.suptitle('Live MATH · 50 questions · FP16 · batch 1 · paired question-bootstrap 95% CI',fontsize=11)
    savefig(task,'live_strategies',fig)

STRATEGY_NAMES={'single_long':'Single long','parallel':'Parallel voting','sequential':'Sequential revision'}

def tables(task,summaries,questions,live):
    analysis=task/'analysis';allfixed=[];adaptive=[];matched=[];signal=[]
    for label,s in summaries.items():
        allfixed.extend({k:(json.dumps(v) if isinstance(v,(dict,list)) else v) for k,v in r.items()} for r in s['fixed'])
        for a in s['adaptive']:
            setting=a['setting'];adaptive.append({'dataset':label,'family':a['family'],'tau':setting['tau'],'gamma':setting['gamma'],'accuracy':a['accuracy']['mean'],'ci95':json.dumps(a['accuracy']['ci95']),'token_cost':a['token_cost'],'mean_rollouts':a['mean_rollouts'],'incorrect_early_stop_rate':a['incorrect_early_stop_rate'],'fixed32_recoverable_early_stop_rate':a['fixed32_recoverable_early_stop_rate']})
            fixed=getfixed(s,'SC');near=min(fixed,key=lambda r:abs(r['token_cost']-a['token_cost']));matched.append({'dataset':label,'family':a['family'],'adaptive_accuracy':a['accuracy']['mean'],'adaptive_cost':a['token_cost'],'nearest_fixed_N':near['budget'],'fixed_accuracy':near['accuracy'],'fixed_cost':near['token_cost'],'cost_difference':a['token_cost']-near['token_cost'],'note':'Nearest observed mean cost; not exactly matched compute.'})
        for name,v in s['signals'].items():signal.append({'dataset':label,'signal':name,'quartile_accuracy_gap':v['quartile_accuracy_gap']['mean'],'gap_ci95':json.dumps(v['quartile_accuracy_gap']['ci95']),**{f'{k}_{j}':z for k in ['spearman_correctness','spearman_length'] for j,z in v[k].items()}})
    csvfile(analysis/'fixed_methods.csv',allfixed);csvfile(analysis/'adaptive.csv',adaptive);csvfile(analysis/'matched_cost.csv',matched);csvfile(analysis/'signal_associations.csv',signal)
    call_rows=[]
    for call in ['single_long','parallel_0','parallel_1','parallel_2','revision']:
        rows=[read(task/'results/live/calls'/f'{i:03d}_{call}.json') for i in range(50)]
        call_rows.append({'call':call,'questions':50,'extraction_failed':sum(r['extraction_failed'] for r in rows),'truncated':sum(r['truncated'] for r in rows),'reached_cap':sum(r['reached_token_cap'] for r in rows),'mean_input_tokens':float(np.mean([r['input_tokens'] for r in rows])),'mean_output_tokens':float(np.mean([r['output_tokens'] for r in rows])),'mean_seconds':float(np.mean([r['seconds'] for r in rows]))})
    csvfile(analysis/'live_call_rates.csv',call_rows)
    finish=read(task/'results/live/finished.json')
    resources={'gpu_environment':read(task/'configs/live_gpu_environment.json'),'physical_calls':finish['physical_calls'],'physical_output_tokens':finish['physical_output_tokens'],'physical_generation_seconds':finish['physical_generation_seconds'],'peak_gpu_allocated_GiB':finish['peak_gpu_allocated_bytes']/2**30,'peak_gpu_reserved_GiB':finish['peak_gpu_reserved_bytes']/2**30,'session':finish['session'],'timing_scope':'Physical generation excludes loading/smoke/disk persistence; session elapsed includes those. Strategy costs reuse and charge the initial call twice across comparisons.'}
    write(analysis/'resources.json',resources)
    text='# Task 3 results\n\nHeld-out evaluation: 800 questions per dataset, ten nested candidate orders per question. Intervals bootstrap questions after averaging orders (1,000 paired resamples, seed 6304). The two model–dataset combinations do not isolate a causal effect of vision.\n\n## Fixed budget: N = 32\n\n'
    text+=mdtable(['Dataset','Method','Accuracy (%)','95% CI','Δ vs SC (pp)','Δ 95% CI (pp)','Token cost'],[(LABELS[l],r['method'],f"{100*r['accuracy']:.3f}",f"[{100*r['accuracy_ci95'][0]:.2f}, {100*r['accuracy_ci95'][1]:.2f}]",f"{100*r['difference_vs_SC']['mean']:+.3f}",str([round(100*x,3) for x in r['difference_vs_SC']['ci95']]),f"{r['token_cost']:.1f}") for l,s in summaries.items() for r in s['fixed'] if r['budget']==32])
    text+='\nSelf-certainty RWC reaches 75.55% on MATH versus 73.7875% for SC. MathVista VisConf RWC reaches 63.125% versus 61.675% for SC. These compare selectors on identical revealed candidate sets. Oracle coverage is a retrospective upper bound, not an implementable selector. Shortest-rollout control and score–length associations prevent treating every quality gain as evidence of calibrated correctness.\n\n## Adaptive sampling\n\n'
    text+=mdtable(['Dataset','Controller','τ','γ','Accuracy (%)','Mean token cost','Mean rollouts','Tokens saved vs SC32 (%)'],[(r['dataset'],r['family'],r['tau'],r['gamma'],f"{r['accuracy']*100:.3f}",f"{r['token_cost']:.1f}",f"{r['mean_rollouts']:.3f}",f"{100*(1-r['token_cost']/getfixed(summaries[r['dataset']],'SC')[-1]['token_cost']):.2f}") for r in adaptive])
    text+='\nThresholds were selected on 200 development questions per dataset and frozen before held-out summaries. Both controllers return current SC. The 1 pp tolerance constrains development accuracy, not a guaranteed held-out loss. Compare the full fixed-SC cost curve in the figure; nearest-cost rows in matched_cost.csv are approximate observed comparisons, not exactly equal-cost trials. Incorrect and recoverable early-stop rates use all question-order outcomes as denominator. Full-pool correctness buckets are retrospective diagnostics only.\n\n## Signal and fusion findings\n\nMathVista calibrated confidence and visual engagement have mean within-rollout correlation −0.20934 (rollouts averaged within each question). Tokenwise fusion minus the rollout product averages −0.00224338, exactly matching the saved population covariance. This negative relationship makes the two fusion definitions different. The fixed six MathVista trajectories were selected before metrics; their confidence uses the complete N=32 set, appropriate only for this retrospective display. Attention associations do not prove causal grounding.\n\n## Live comparison\n\n'
    text+=mdtable(['Strategy','Correct / 50','Accuracy (%)','Input tokens','Output tokens','Time (s)','Selected truncation (%)','Extraction failure (%)'],[(STRATEGY_NAMES[s],live['strategies'][s]['correct_questions'],f"{100*live['strategies'][s]['correct']['mean']:.1f}",f"{live['strategies'][s]['input_tokens']['mean']:.2f}",f"{live['strategies'][s]['output_tokens']['mean']:.2f}",f"{live['strategies'][s]['seconds']['mean']:.2f}",f"{100*live['strategies'][s]['selected_truncated']['mean']:.1f}",f"{100*live['strategies'][s]['extraction_failed']['mean']:.1f}") for s in STRATEGY_NAMES])
    text+='\nAll three have the same 3,072-token output allowance; actual use differs. Output cost includes EOS; retained-token counts are separate in the CSV. Parallel samples were executed serially, so runtime is summed; the shared initial call is charged to both parallel and sequential. Sequential includes all critique tokens and the longer input prompt. Loading, smoke and disk writes are outside inference timing. The shared initial answer was correct on 28/50: revision corrected one error, spoiled none, and retained 28 correct answers. Long/parallel each score 30/50; revision scores 29/50. A 50-question comparison supports limited precision rather than a universal strategy ranking.\n\n'
    text+=mdtable(['Paired accuracy difference','Mean (pp)','95% CI (pp)'],[(k,f"{100*v['correct']['mean']:+.2f}",str([round(100*x,2) for x in v['correct']['ci95']])) for k,v in live['paired_differences'].items()])
    text+='\n## Audit scope and limitations\n\nAll stored chunk and live-call hashes, split/order IDs, selection membership, retained-token costs, development grid selection, fixed/adaptive metrics, and saved confidence intervals are rechecked. Live parsing, grading, revision messages, seeds, limits and costs are independently rechecked on CPU. This report export excludes raw parquet and prepared NPZ arrays; it cannot rerun the original full-distribution token alignment or recompute every VisConf value from raw arrays. Original preparation audits and hashes are retained. Model tensor bytes and GPU timings are recorded evidence, not re-measured by CPU report mode. No gold/future candidates enter selection or stopping.\n'
    (analysis/'RESULTS.md').write_text(text)
    examples=[];md='# Aligned qualitative examples\n\nOffline selection: first lexicographic held-out question satisfying each saved failure/disagreement condition; earliest qualifying order. Examples are descriptive and selected after outcomes, not prevalence estimates. The underlying generation text remains in the supplied parquet; these panels align the problem, reference, extracted candidates and decisions. Live revision examples use the first two lexicographic IDs per transition.\n\n'
    for label,s in summaries.items():
        for case in s['cases']:
            q=questions[label][case['sample_id']];order=q['orders'][case['order_index']];p=order[str(case.get('stop_n',32))]
            ids=([case['SC_rollout'],case['SelfCertBest_rollout']] if 'SC_rollout' in case else [case['selected_rollout_id'],order['32']['choices']['SC']])
            record={'dataset':label,'case':case,'problem':q['question_text'],'gold_answer':q['gold_answer'],'revealed_ids':p['candidate_ids'],'SC_votes':p['sc']['votes'],'RWC_votes':p['rwc']['votes'],'aligned_candidates':[q['candidates'][rid] for rid in ids]};examples.append(record)
            md+=f"## {label} · {case['kind']} · {q['sample_id']}\n\n{q['question_text']}\n\nReference: `{q['gold_answer']}`. Order {case['order_index']}; revealed {len(p['candidate_ids'])} candidates.\n\n"
            md+=mdtable(['Rollout','Extracted answer','Correct','Retained tokens','Mean logp','Self-certainty'],[(c['rollout_id'],c['extracted_answer'],c['correct'],c['tokens'],round(c['mean_logp'],4),round(c['mean_kl'],4)) for c in record['aligned_candidates']])
            md+='\nDecision and prefix votes:\n\n```json\n'+json.dumps({'case':case,'SC_votes':p['sc']['votes'],'RWC_votes':p['rwc']['votes']},indent=2)+'\n```\n\n'
    live_examples=read(task/'results/live_scoring/revision_examples.json')
    for transition,rows in live_examples.items():
        for r in rows:
            md+=f"## Live · {transition} · {r['sample_id']}\n\n{r['problem']}\n\nReference: `{r['gold_answer']}`\n\n### Shared initial response\n\n```text\n{r['initial_response']}\n```\n\n### Critique and revision\n\n```text\n{r['revision_response']}\n```\n\n"
    write(analysis/'qualitative_examples.json',{'offline':examples,'live':live_examples});(analysis/'QUALITATIVE.md').write_text(md)

def analyze(task):
    task=Path(task);(task/'analysis').mkdir(exist_ok=True);(task/'figures').mkdir(exist_ok=True)
    live=score(task);summaries,questions=audit_offline(task)
    figures(task,summaries,questions,live);tables(task,summaries,questions,live)
    checks=['Frozen source and input identities','All 40 offline chunk hashes; 2000 questions and 64000 candidates','Saved splits, nested candidate orders, selected IDs and retained-token costs','Development-only adaptive grid selection and frozen thresholds','All held-out fixed/adaptive summaries and paired bootstrap intervals','All 250 live call hashes, prompt/seed/cap/EOS/revision/cost records','Live gold-free voting, frozen grading, paired intervals and revision transitions']
    audit={'passed':True,'gpu_used':False,'checks':checks,'limits':'Raw parquet/NPZ/model bytes are external and not re-read by this report audit. Token-level measurements retain original preparation hashes and audit receipts.','live_manifest_sha256':live['call_manifest_sha256'],'offline_finished_sha256':sha(task/'results/offline/finished.json'),'source_sha256':sha(Path(__file__))}
    write(task/'analysis/audit.json',audit)
    print(json.dumps({'passed':True,'gpu_used':False,'figures':len(list((task/'figures').glob('*.png'))),'checks':checks},indent=2))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-dir',type=Path,required=True);analyze(p.parse_args().task_dir)
