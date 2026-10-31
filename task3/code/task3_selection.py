"""Task 3 selection/stopping: inputs contain no gold or correctness labels."""
import collections
import numpy as np
from scipy.stats import rankdata
from task3_answers import NO_ANSWER

BUDGETS=(1,2,4,8,16,32)
TAUS=(.5,.625,.75,.875,1.)
GAMMAS=(0.,.1,.25,.5)

def vote(groups,units=None):
    if units is None:units=np.ones(len(groups),dtype=np.int64)
    totals={};first={}
    for i,(group,weight) in enumerate(zip(groups,units)):
        totals[group]=totals.get(group,0)+int(weight);first.setdefault(group,i)
    winner=max(totals,key=lambda g:totals[g]) # insertion order = earliest support
    ranked=sorted(totals.values(),reverse=True);runner=ranked[1] if len(ranked)>1 else 0
    return {'index':first[winner],'group':winner,'unique':ranked.count(ranked[0])==1,
      'support':groups.count(winner)/len(groups),'margin':(ranked[0]-runner)/sum(ranked),
      'votes':totals}

def rwc(groups,scores):
    n=len(groups)
    # Twice the descending average ranks are integers. Use integer vote units
    # so mathematically tied votes do not differ due to float accumulation.
    units=np.ones(1,dtype=np.int64) if n==1 else np.rint(2*(n-rankdata(-np.asarray(scores),method='average'))).astype(np.int64)
    return vote(groups,units)

def visual_scores(kl,vision):
    assert len(kl)==len(vision) and all(len(a)>0 and len(a)==len(b) for a,b in zip(kl,vision))
    lo,hi=np.percentile(np.concatenate(kl),[5,95],method='linear')
    c=[np.clip((np.asarray(k,dtype=np.float64)-lo)/max(float(hi-lo),1e-8),0,1) for k in kl]
    confidence=np.array([x.mean() for x in c]);engagement=np.array([np.asarray(v,dtype=np.float64).mean() for v in vision])
    tokenwise=np.array([np.mean(x*np.asarray(v,dtype=np.float64)) for x,v in zip(c,vision)])
    product=confidence*engagement
    covariance=np.array([np.mean((x-x.mean())*(np.asarray(v,dtype=np.float64)-np.mean(v))) for x,v in zip(c,vision)])
    assert np.allclose(tokenwise-product,covariance,atol=1e-12,rtol=1e-10)
    return {'confidence':confidence,'vision':engagement,'visconf':product,'tokenwise':tokenwise,
            'calibrated_tokens':c,'covariance':covariance,'percentiles':[float(lo),float(hi)]}

def select_prefix(groups,lengths,logp,kl,visual=None):
    sc=vote(groups);quality=kl if visual is None else visual['visconf'];weighted=rwc(groups,quality)
    choices={'FirstAns':0,'SC':sc['index'],'LogPBest':int(np.argmax(logp)),
      'SelfCertBest':int(np.argmax(kl)),'SelfCertRWC':rwc(groups,kl)['index'],
      'Shortest':int(np.argmin(lengths))}
    if visual is not None:
        for stem,key in [('Vision','vision'),('VisConf','visconf'),('Tokenwise','tokenwise')]:
            choices[stem+'Best']=int(np.argmax(visual[key]));choices[stem+'RWC']=rwc(groups,visual[key])['index']
    if len(groups)==1:assert set(choices.values())=={0}
    return {'choices':choices,'sc':sc,'rwc':weighted,'distinct_groups':len(set(groups))}

def stop_at(prefixes,family,tau=None,gamma=0.):
    if family=='always_32':return 32
    for n in (4,8,16):
        p=prefixes[str(n)];sc=p['sc'];weighted=p['rwc']
        agreement=sc['unique'] and sc['group']!=NO_ANSWER and sc['support']>=tau
        if agreement and (family=='agreement' or (family=='quality' and sc['group']==weighted['group'] and weighted['margin']>=gamma)):
            return n
    return 32

def policy_grid(family):
    grid=[{'family':family,'tau':tau,'gamma':gamma} for tau in TAUS for gamma in (GAMMAS if family=='quality' else (0.,))]
    return grid+[{'family':'always_32','tau':None,'gamma':None}]

def choose_policy(evaluated,baseline):
    eligible=[row for row in evaluated if row['accuracy']>=baseline-.01-1e-12]
    assert eligible,'Always-32 fallback must be eligible.'
    return min(eligible,key=lambda r:(r['token_cost'],-r['accuracy'],r['setting']['tau'] if r['setting']['tau'] is not None else float('inf'),r['setting']['gamma'] if r['setting']['gamma'] is not None else float('inf')))
