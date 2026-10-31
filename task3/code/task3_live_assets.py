"""CPU-only pinned dataset/model download and gold-free live-prompt checks."""
import argparse,hashlib,importlib.metadata as metadata,json,os,sys,time
from pathlib import Path
from task3_live_spec import *

def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def original_reference(row):
    from task3_answers import clean,last_boxed_answer
    if 'answer' in row and clean(row['answer']) is not None:return str(row['answer'])
    # This pinned dataset has problem/level/solution/type, not an answer column.
    value=last_boxed_answer(row['solution'])
    assert value is not None,'The selected original solution has no valid last literal boxed answer.'
    return value

def prepare(task):
    started=time.perf_counter();task=task.resolve()
    os.environ.setdefault('HF_HOME','/content/pa3_task3_hf_cache');os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['HF_HUB_OFFLINE']='0';os.environ['HF_DATASETS_OFFLINE']='0'
    offline=read(task/'results/offline/finished.json');assert offline['passed']
    assert offline['policy_sha256']==sha(task/'configs/adaptive_policies_frozen.json')
    scorer=read(task/'configs/scorer_frozen.json');split=read(task/'splits/question_splits.json')['datasets']['math']
    for name,version in scorer['packages'].items():assert metadata.version(name)==version,(name,metadata.version(name),version)
    for record in [scorer,read(task/'configs/preparation_frozen.json'),read(task/'configs/offline_frozen.json')]:
        for name,value in record['source_sha256'].items():assert sha(task/'code'/name)==value,name
    assert scorer['live_math_model']=={'id':MODEL_ID,'revision':MODEL_REVISION}
    selected=read(task/'splits/live_math_questions.json')['questions']
    assert [q['sample_id'] for q in selected]==split['live_math']==split['evaluation'][:50]
    assert len(selected)==50
    from datasets import load_dataset
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoTokenizer
    from task3_pool import PreparedPool
    from task3_answers import math_equivalent

    print('Checking the pinned original MATH test data on CPU...',flush=True)
    original=load_dataset(DATASET_ID,revision=DATASET_REVISION,split='test');assert len(original)==5000
    chosen=sorted(__import__('numpy').random.default_rng(42).choice(5000,size=1000,replace=False).tolist())
    supplied_ids=split['development']+split['evaluation']
    assert sorted(int(qid.split(':')[1]) for qid in supplied_ids)==chosen,'Supplied MATH source indices differ from README sampling.'
    references=[]
    with PreparedPool(task,'math') as pool:
        for q in selected:
            index=q['source_test_row_index'];row=original[index]
            assert q['sample_id']==f'test:{index}' and q['question_text']==row['problem'],q['sample_id']
            prepared=pool.get(q['sample_id'])['metadata'];assert prepared['question_text']==row['problem']
            source_gold=original_reference(row)
            assert math_equivalent(prepared['gold_answer'],source_gold),q['sample_id']
            references.append({'sample_id':q['sample_id'],'source_test_row_index':index,'gold_answer':source_gold,
              'source_field':'answer' if 'answer' in row else 'solution_last_literal_box'})
    # This reference file is for later CPU grading. The GPU generation input below
    # contains only problems/messages/prompts and token IDs, never references.
    write(task/'data/live_math_references.json',{'dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,'references':references})

    api=HfApi();info=api.model_info(MODEL_ID,revision=MODEL_REVISION,files_metadata=True)
    assert info.sha==MODEL_REVISION
    files=sorted(s.rfilename for s in info.siblings if '/' not in s.rfilename and (s.rfilename.endswith(('.json','.safetensors')) or s.rfilename in ('merges.txt','vocab.txt','tokenizer.model','LICENSE','README.md')))
    weight_names=[name for name in files if name.endswith('.safetensors')];assert weight_names
    source=task/'assets/live_model'/MODEL_REVISION
    small=[name for name in files if name not in weight_names]
    snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_dir=str(source),allow_patterns=small,max_workers=4)
    config=read(source/'config.json');assert config['model_type']=='qwen2' and not config.get('quantization_config')
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True,trust_remote_code=False)
    prompts=[prompt_record(tokenizer,q['sample_id'],q['question_text']) for q in selected]
    max_context=int(config['max_position_embeddings']);max_prompt=max(p['input_tokens'] for p in prompts)
    revision_overhead=len(tokenizer(REVISION_INSTRUCTION,add_special_tokens=False)['input_ids'])+64
    assert max_prompt+1024+revision_overhead+2048<=max_context,'Live revision might exceed model context; do not truncate prompts.'
    seeds=[seed_for(q['sample_id'],call) for q in selected for call in CALLS];assert len(set(seeds))==250
    write(task/'data/live_prompts.json',{'model_revision':MODEL_REVISION,'questions':prompts})
    write(task/'configs/live_recipe.json',recipe())

    print('Downloading pinned Qwen2.5-3B weights on CPU; no model forward pass...',flush=True)
    snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_dir=str(source),allow_patterns=files,max_workers=4)
    upstream={s.rfilename:s for s in info.siblings};records={}
    for name in files:
        path=source/name;record={'bytes':path.stat().st_size,'sha256':sha(path)}
        sibling=upstream[name];assert sibling.size is None or record['bytes']==sibling.size,name
        if sibling.lfs is not None:assert record['sha256']==sibling.lfs.sha256,name
        records[name]=record
    packages={name:metadata.version(name) for name in ['torch','transformers','datasets','accelerate','huggingface_hub','tokenizers','numpy','safetensors','math-verify','sympy']}
    frozen={'version':VERSION,'model_id':MODEL_ID,'revision':MODEL_REVISION,'source_path':str(source),'files':records,
      'dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,'cpu_packages':packages,
      'scorer_sha256':sha(task/'configs/scorer_frozen.json'),'offline_finished_sha256':sha(task/'results/offline/finished.json'),
      'question_splits_sha256':sha(task/'splits/question_splits.json'),'selected_questions_sha256':sha(task/'splits/live_math_questions.json'),
      'prompts_sha256':sha(task/'data/live_prompts.json'),'references_sha256':sha(task/'data/live_math_references.json'),
      'recipe_sha256':sha(task/'configs/live_recipe.json'),
      'source_sha256':{name:sha(task/'code'/name) for name in ['task3_live_spec.py','task3_live_assets.py','test_task3_live_cpu.py']}}
    lock=task/'configs/live_assets.json'
    if lock.exists():assert read(lock)==frozen,'Live assets/environment changed; refusing to overwrite the established identity.'
    else:write(lock,frozen)
    result={'passed':True,'questions':50,'original_math_questions_verified':True,'original_math_subset_ids_verified':1000,
      'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'weight_files':len(weight_names),
      'weight_GiB':sum(records[n]['bytes'] for n in weight_names)/2**30,'max_base_input_tokens':max_prompt,'model_context_tokens':max_context,
      'independent_call_seeds':len(seeds),'physical_calls':250,'max_physical_output_tokens':50*sum(CALLS.values()),
      'packages':packages,'model_loaded':False,'gpu_used':False,'seconds_this_session':time.perf_counter()-started,'assets_sha256':sha(lock)}
    write(task/'results/live_assets_preflight.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True);prepare(parser.parse_args().task_dir)
