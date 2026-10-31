"""CPU token audit, fixed candidate orders, and joined local question caches.

No model loading, math-equivalence grading, policy selection, or GPU operations.
"""
from __future__ import annotations
import argparse,collections,hashlib,importlib.metadata as metadata,json,shutil,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from task3_answers import clean,last_boxed_answer
from task3_pool import sha256,PreparedPool

VERSION='pa3-task3-prepared-v1'
EXPECTATIONS={'math':{'questions':1000,'rows':32000,'tokens':20742094,'capped':523,'missing':521},
              'mathvista':{'questions':1000,'rows':32000,'tokens':8961703,'capped':487,'missing':3032}}
SOURCE_FILES=['task3_prepare.py','task3_pool.py','test_task3_prepare_cpu.py']

def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v,compact=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temporary=p.with_name(p.name+'.tmp')
    temporary.write_text(json.dumps(v,ensure_ascii=False,allow_nan=False,**({'separators':(',',':')} if compact else {'indent':2}))+'\n');temporary.replace(p)

def candidate_orders(ids_by_dataset):
    result={'seed':6304,'orders_per_question':10,'rollouts_per_question':32,
      'rule':'Independent default_rng(6304) per dataset; sorted question IDs; ten consecutive permutations per question; reuse nested prefixes across all methods and policies.',
      'datasets':{}}
    for label in ('math','mathvista'):
        rng=np.random.default_rng(6304)
        result['datasets'][label]={qid:[rng.permutation(32).tolist() for _ in range(10)] for qid in sorted(ids_by_dataset[label])}
    return result

def index_table(table):
    columns=table.select(['dataset','sample_id','rollout_id']).to_pydict();index={}
    for i,(dataset,qid,rid) in enumerate(zip(columns['dataset'],columns['sample_id'],columns['rollout_id'])):
        assert isinstance(dataset,str) and isinstance(qid,str)
        key=(dataset,qid,int(rid));assert key not in index,('Duplicate join key',key)
        index[key]=i
    return index

def array(table,column,row,dtype):
    scalar=table[column][row];assert scalar.is_valid,(column,row)
    assert scalar.values.null_count==0,(column,row)
    value=np.asarray(scalar.values.to_numpy(zero_copy_only=False),dtype=dtype)
    assert value.ndim==1
    return value

def check_vectors(tokens,logp,kl,n,attention=None):
    assert n>=0 and len(tokens)==len(logp)==len(kl)==n,'Generated/probability token lengths differ.'
    assert np.isfinite(logp).all() and np.isfinite(kl).all(),'Non-finite probability measurement.'
    assert (logp<=1e-4).all() and (kl>=-1e-4).all(),'Invalid log probability or KL.'
    assert (tokens>=0).all() and (tokens<=np.iinfo(np.int32).max).all()
    if attention is None:return None
    image,prompt,generated,preceding=attention
    assert all(len(a)==n for a in attention),'Attention token lengths differ.'
    assert np.array_equal(preceding,np.arange(n)),'Attention preceding counts are not generated-response lengths.'
    masses=np.stack([image,prompt,generated]).astype(np.float64)
    assert np.isfinite(masses).all() and (masses>=0).all() and (masses<=1.0001).all()
    total=masses.sum(axis=0)
    assert (total>0).all() and (total<=1.001).all(),'Invalid eligible attention total.'
    if n:assert generated[0]==0,'First prediction must have zero preceding-generated attention.'
    ratio=image.astype(np.float64)/total
    assert np.isfinite(ratio).all() and (ratio>=0).all() and (ratio<=1).all()
    return ratio.astype(np.float32)

def stage_source(record,work):
    source=Path(record['path']);assert source.stat().st_size==record['bytes'],source
    destination=work/record['sha256'][:16]/source.name;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        assert destination.stat().st_size==record['bytes'] and sha256(destination)==record['sha256'],destination
        return destination
    print('Staging and hashing:',source.name,flush=True)
    temporary=destination.with_name(destination.name+'.tmp');h=hashlib.sha256()
    with source.open('rb') as src,temporary.open('wb') as dst:
        for block in iter(lambda:src.read(4*1024**2),b''):h.update(block);dst.write(block)
    assert h.hexdigest()==record['sha256'] and temporary.stat().st_size==record['bytes'],source
    temporary.replace(destination);return destination

def persist_npz(directory,name,arrays,work):
    local=work/name
    with local.open('wb') as f:np.savez_compressed(f,**arrays)
    expected=sha256(local);target=directory/name;temporary=target.with_name(name+'.tmp')
    shutil.copyfile(local,temporary);assert sha256(temporary)==expected;temporary.replace(target)
    local.unlink()
    return {'file_name':name,'bytes':target.stat().st_size,'sha256':expected}

def prepare_dataset(task,label,files,identity,work,expected=None):
    expected=expected or EXPECTATIONS[label];directory=task/'data/prepared'/label;directory.mkdir(parents=True,exist_ok=True)
    manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        saved=read(manifest_path);assert saved['identity']==identity,'Prepared dataset identity changed.'
        for shard in saved['shards']:assert sha256(directory/shard['file_name'])==shard['sha256']
        print(label+': verified existing prepared dataset.',flush=True);return saved
    tables={name:pq.ParquetFile(stage_source(record,work)).read() for name,record in files.items()}
    indices={name:index_table(table) for name,table in tables.items()};keys=set(indices['generations.parquet'])
    assert all(set(index)==keys for index in indices.values()),'Parquet join-key mismatch.'
    assert len(keys)==expected['rows'] and len({k[0] for k in keys})==1
    dataset=next(iter(keys))[0];qids=sorted({k[1] for k in keys});assert len(qids)==expected['questions']
    stats=collections.Counter();models=set();shards=[];locations=[];arrays={};chunk_ids=[];started=time.perf_counter()
    g=tables['generations.parquet'];p=tables['probability_metrics.parquet'];a=tables.get('attention_metrics.parquet')
    for qindex,qid in enumerate(qids):
        assert {k[2] for k in keys if k[1]==qid}==set(range(32))
        rows=[];vectors=collections.defaultdict(list);offsets=[0];question=None;gold=None
        for rid in range(32):
            key=(dataset,qid,rid);gi=indices['generations.parquet'][key];pi=indices['probability_metrics.parquet'][key]
            n=g['n_generated_text_tokens'][gi].as_py();assert isinstance(n,int)
            tokens=array(g,'generated_token_ids',gi,np.int64)
            logp=array(p,'logp',pi,np.float32);kl=array(p,'kl_u_p',pi,np.float32)
            attention=None;eligible={}
            if a is not None:
                ai=indices['attention_metrics.parquet'][key]
                attention=tuple(array(a,name,ai,dtype) for name,dtype in [('img_attn_total',np.float32),('prompt_text_attn_total',np.float32),('generated_text_attn_total',np.float32),('n_generated_text_tokens',np.int64)])
                eligible={name:a[name][ai].as_py() for name in ['n_image_tokens','n_prompt_text_tokens']}
                assert all(isinstance(v,int) and v>0 for v in eligible.values())
            vision=check_vectors(tokens,logp,kl,n,attention)
            text=g['generated_text'][gi].as_py();extracted=g['extracted_answer'][gi].as_py();q=g['question_text'][gi].as_py();ref=g['gold_answer'][gi].as_py()
            model=g['model'][gi].as_py();capped=g['hit_max_new_tokens'][gi].as_py()
            assert isinstance(text,str) and isinstance(q,str) and isinstance(capped,bool)
            if rid==0:question=q;gold=ref
            else:assert q==question and ref==gold,'Question/reference changed across rollouts.'
            if label=='math':assert last_boxed_answer(text)==clean(extracted),('Stored MATH extraction differs from README rule',qid,rid)
            stats['rows']+=1;stats['tokens']+=n;stats['capped']+=int(capped);stats['missing']+=int(clean(extracted) is None);stats['zero_token_rollouts']+=int(n==0);models.add(model)
            rows.append({'rollout_id':rid,'generated_text':text,'extracted_answer':extracted,'n_generated_text_tokens':n,
              'hit_max_new_tokens':capped,'mean_logp':float(logp.astype(np.float64).mean()) if n else None,
              'mean_kl':float(kl.astype(np.float64).mean()) if n else None,
              'mean_vision':float(vision.astype(np.float64).mean()) if n and vision is not None else None,**eligible})
            vectors['token_ids'].append(tokens.astype(np.int32));vectors['logp'].append(logp);vectors['kl'].append(kl)
            if vision is not None:vectors['vision'].append(vision)
            offsets.append(offsets[-1]+n)
        entry={'dataset':dataset,'sample_id':qid,'question_text':question,'gold_answer':gold,'model':model,'rollouts':rows}
        prefix=f'q{qindex%50:02d}';arrays[prefix+'_metadata']=np.frombuffer(json.dumps(entry,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode(),dtype=np.uint8)
        arrays[prefix+'_offsets']=np.asarray(offsets,dtype=np.int64)
        for name,values in vectors.items():arrays[prefix+'_'+name]=np.concatenate(values)
        shard_name=f'shard_{qindex//50:04d}.npz';locations.append({'sample_id':qid,'shard':shard_name,'key':prefix});chunk_ids.append(qid)
        if (qindex+1)%50==0 or qindex+1==len(qids):
            record=persist_npz(directory,shard_name,arrays,work);record['questions']=list(chunk_ids);shards.append(record)
            arrays={};chunk_ids=[];print(f'{label}: {qindex+1}/{len(qids)} questions audited and cached',flush=True)
    stats['questions']=len(qids)
    for name,value in expected.items():assert stats[name]==value,(label,name,stats[name],value)
    assert len(models)==1
    manifest={'identity':identity,'dataset':label,'source_dataset':dataset,'models':sorted(models),'statistics':dict(stats),'shards':shards,'questions':locations,
      'audit_and_cache_seconds':time.perf_counter()-started,'cache_bytes':sum(s['bytes'] for s in shards),'npz_allow_pickle':False,
      'timing_scope':'Token validation/cache construction, excluding source staging and Arrow table loading.'}
    write(manifest_path,manifest);return manifest

def prepare(task,work):
    task=task.resolve();work.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    inventory=read(task/'configs/supplied_data_inventory.json');splits=read(task/'splits/question_splits.json');frozen=read(task/'configs/scorer_frozen.json')
    assert read(task/'results/scorer_preflight.json')['passed']
    assert sha256(task/'configs/scorer_frozen.json')==read(task/'results/scorer_preflight.json')['scorer_sha256']
    for name,version in frozen['packages'].items():assert metadata.version(name)==version,(name,metadata.version(name),version)
    for name,digest_value in frozen['source_sha256'].items():assert sha256(task/'code'/name)==digest_value,name
    asset_root=task/'assets/mathvista'/frozen['mathvista_revision']
    assert sha256(asset_root/'testmini.json')==frozen['mathvista_metadata_sha256']
    assert sha256(asset_root/'calculate_score.py')==frozen['mathvista_upstream_scorer_sha256']
    assert sha256(task/'assets/supplied_README.MD')==frozen['supplied_readme_sha256']
    assert sha256(task/'configs/supplied_data_inventory.json')==frozen['input_inventory_sha256']
    assert sha256(task/'splits/question_splits.json')==frozen['question_splits_sha256']
    ids={label:sorted(s['development']+s['evaluation']) for label,s in splits['datasets'].items()}
    for label,values in ids.items():
        assert len(values)==len(set(values))==1000
        perm=np.random.default_rng(6304).permutation(len(values));expected=[values[int(i)] for i in perm]
        assert splits['datasets'][label]['development']==expected[:200] and splits['datasets'][label]['evaluation']==expected[200:]
    orders=candidate_orders(ids);order_path=task/'splits/candidate_orders.json'
    if order_path.exists():assert read(order_path)==orders,'Candidate orders already exist with different contents.'
    else:write(order_path,orders,True)
    identity={'version':VERSION,'scorer_sha256':sha256(task/'configs/scorer_frozen.json'),'inventory_sha256':sha256(task/'configs/supplied_data_inventory.json'),
      'question_splits_sha256':sha256(task/'splits/question_splits.json'),'candidate_orders_sha256':sha256(order_path),
      'source_sha256':{name:sha256(task/'code'/name) for name in SOURCE_FILES},
      'vision_ratio':'img_attn_total / (img_attn_total + prompt_text_attn_total + generated_text_attn_total); no per-group-size correction',
      'storage':'50 questions per NPZ shard; UTF-8 JSON uint8 metadata; packed token arrays/offsets; allow_pickle=False'}
    recipe_path=task/'configs/preparation_frozen.json'
    if recipe_path.exists():assert read(recipe_path)==identity,'Preparation code/input identity changed.'
    else:write(recipe_path,identity)
    manifests={label:prepare_dataset(task,label,inventory[label]['files'],identity,work) for label in ('math','mathvista')}
    for label in manifests:assert {r['sample_id'] for r in manifests[label]['questions']}==set(ids[label])
    with PreparedPool(task,'math') as pool:
        live=[]
        for qid in splits['datasets']['math']['live_math']:
            q=pool.get(qid)['metadata'];assert qid.startswith('test:')
            live.append({'sample_id':qid,'source_test_row_index':int(qid.split(':')[1]),'question_text':q['question_text']})
    assert len(live)==50
    write(task/'splits/live_math_questions.json',{'rule':'First 50 saved MATH held-out IDs, in the saved permutation order; no gold answers in this generation-input file.','questions':live})
    examples={label:[{'sample_id':qid,'rollout_ids':orders['datasets'][label][qid][0][:2]} for qid in splits['datasets'][label]['evaluation'][:3]] for label in ('math','mathvista')}
    write(task/'splits/trajectory_examples.json',{'rule':'First three saved held-out questions per dataset and first two candidates of saved order 0; fixed before quality/correctness analysis.','datasets':examples})
    result={'passed':True,'identity':identity,'datasets':{label:{'models':m['models'],**m['statistics'],'shards':len(m['shards']),'cache_bytes':m['cache_bytes']} for label,m in manifests.items()},
      'candidate_orders_per_question':10,'live_math_questions':50,'gpu_used':False,'held_out_metrics_computed':False,'elapsed_seconds_this_session':time.perf_counter()-started}
    write(task/'data/prepared/manifest.json',{'identity':identity,'datasets':{label:{'manifest_sha256':sha256(task/'data/prepared'/label/'manifest.json')} for label in manifests}})
    write(task/'results/data_preparation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True);parser.add_argument('--work-dir',type=Path,default=Path('/content/pa3_task3_work/preparation'))
    args=parser.parse_args();prepare(args.task_dir,args.work_dir)
