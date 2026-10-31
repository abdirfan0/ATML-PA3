"""Live generation primitives: no gold-reference loading or answer selection."""
from __future__ import annotations
import hashlib,json,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from task3_live_spec import CALLS,STRATEGIES,MODEL_ID,MODEL_REVISION,seed_for,prompt_record,recipe
from task3_answers import last_boxed_answer

EXECUTION_ORDER=('parallel_0','parallel_1','parallel_2','single_long','revision')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def call_plan(prompts):
    assert len(prompts)==50 and len({p['sample_id'] for p in prompts})==50
    plan=[]
    for index,q in enumerate(prompts):
        assert not any(k in q for k in ('gold_answer','references','solution','correct'))
        for call in EXECUTION_ORDER:
            plan.append({'question_index':index,'sample_id':q['sample_id'],'call':call,'seed':seed_for(q['sample_id'],call),
              'max_new_tokens':CALLS[call],'parent_call':'parallel_0' if call=='revision' else None})
    assert len(plan)==250 and len({p['seed'] for p in plan})==250
    return plan

def stage_model(assets,local):
    local=Path(local);local.mkdir(parents=True,exist_ok=True);source=Path(assets['source_path'])
    started=time.perf_counter();copied=0
    for name,record in assets['files'].items():
        target=local/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==record['bytes'] and sha(target)==record['sha256']:continue
        original=source/name;assert original.stat().st_size==record['bytes'],name
        temp=target.with_name(target.name+'.tmp');h=hashlib.sha256()
        with original.open('rb') as src,temp.open('wb') as dst:
            for block in iter(lambda:src.read(4*1024**2),b''):h.update(block);dst.write(block)
        assert h.hexdigest()==record['sha256'] and temp.stat().st_size==record['bytes'],name
        temp.replace(target);copied+=record['bytes']
        if name.endswith('.safetensors'):print('Staged and verified:',name,flush=True)
    return {'seconds':time.perf_counter()-started,'copied_bytes':copied,'local_path':str(local)}

def eos_ids(model,tokenizer):
    configured=model.generation_config.eos_token_id
    values=([configured] if isinstance(configured,int) else list(configured or []))+[tokenizer.eos_token_id]
    result=sorted({int(v) for v in values if v is not None});assert result
    return result

@torch.inference_mode()
def generate_record(model,tokenizer,question,call,initial_response=None,smoke=False,smoke_cap=None):
    assert call in CALLS
    assert (initial_response is not None)==(call=='revision')
    assert smoke or smoke_cap is None,'A main-run allowance cannot be overridden.'
    cap=CALLS[call] if smoke_cap is None else int(smoke_cap)
    if smoke_cap is not None:assert smoke and 1<=cap<=64
    device=next(model.parameters()).device;cuda=device.type=='cuda'
    if cuda:torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    request=prompt_record(tokenizer,question['sample_id'],question['problem'],initial_response)
    if call!='revision' and 'input_ids' in question:
        assert request['input_ids']==question['input_ids'] and request['prompt']==question['prompt'],'CPU/GPU tokenizer/template mismatch.'
    assert request['input_tokens']+cap<=model.config.max_position_embeddings,'Context limit exceeded; do not truncate.'
    seed=seed_for(question['sample_id'],call);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if cuda:torch.cuda.manual_seed_all(seed)
    ids=torch.tensor([request['input_ids']],device=device,dtype=torch.long);mask=torch.ones_like(ids)
    stops=eos_ids(model,tokenizer);pad=tokenizer.pad_token_id
    if pad is None:pad=stops[0]
    # Explicit settings override the checkpoint's native repetition_penalty=1.05.
    settings={'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.,
      'num_beams':1,'num_return_sequences':1,'use_cache':True,'max_new_tokens':cap,'eos_token_id':stops,'pad_token_id':pad}
    output=model.generate(input_ids=ids,attention_mask=mask,**settings)
    assert output.shape[0]==1 and torch.equal(output[0,:ids.shape[1]],ids[0])
    generated=output[0,ids.shape[1]:].tolist();assert 0<len(generated)<=cap
    ended=generated[-1] in stops;retained=generated[:-1] if ended else generated
    response=tokenizer.decode(retained,skip_special_tokens=False)
    raw_text=tokenizer.decode(generated,skip_special_tokens=False)
    extraction=last_boxed_answer(response)
    if cuda:torch.cuda.synchronize()
    seconds=time.perf_counter()-begin
    return {'sample_id':question['sample_id'],'call':call,'seed':seed,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
      'request':request,'generation_settings':settings,'response':response,'raw_text':raw_text,
      'generated_token_ids':generated,'retained_token_ids':retained,'input_tokens':request['input_tokens'],
      'output_tokens':len(generated),'retained_output_tokens':len(retained),'max_new_tokens':cap,
      'ended_by_eos':ended,'reached_token_cap':len(generated)==cap,'truncated':len(generated)==cap and not ended,
      'extracted_answer':extraction,'extraction_failed':extraction is None,'seconds':seconds,
      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if cuda else 0,
      'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved() if cuda else 0,'smoke':smoke}

def call_path(directory,index,call):return Path(directory)/f'{index:03d}_{call}.json'
def save_call(path,record,identity):
    value={**record,'identity':identity};write(path,value)
    write(path.with_name(path.name+'.complete.json'),{'identity':identity,'sha256':sha(path)})
def load_call(path,identity,expected,parent_hash=None):
    marker=path.with_name(path.name+'.complete.json')
    if not marker.exists():return None
    completed=read(marker);assert completed['identity']==identity and completed['sha256']==sha(path),path
    row=read(path);assert row['identity']==identity and not row['smoke']
    assert row['sample_id']==expected['sample_id'] and row['call']==expected['call'] and row['seed']==expected['seed']
    assert row['max_new_tokens']==expected['max_new_tokens']==CALLS[row['call']]
    assert row['output_tokens']==len(row['generated_token_ids'])<=row['max_new_tokens']
    assert row.get('initial_call_sha256')==parent_hash
    return row

def strategy_costs(calls):
    assert set(calls)==set(CALLS)
    return {strategy:{'calls':keys,'input_tokens':sum(calls[k]['input_tokens'] for k in keys),
      'output_tokens':sum(calls[k]['output_tokens'] for k in keys),
      'retained_output_tokens':sum(calls[k]['retained_output_tokens'] for k in keys),
      'seconds':sum(calls[k]['seconds'] for k in keys),
      'peak_gpu_allocated_bytes':max(calls[k]['peak_gpu_allocated_bytes'] for k in keys)} for strategy,keys in STRATEGIES.items()}
