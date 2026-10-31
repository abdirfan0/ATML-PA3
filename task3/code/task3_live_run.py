"""CPU runner preflight or resumable single-GPU live MATH generation queue."""
from __future__ import annotations
import argparse,importlib.metadata as metadata,json,os,platform,sys,time
from pathlib import Path
import torch
from task3_live_spec import MODEL_ID,MODEL_REVISION,CALLS,recipe,prompt_record
from task3_live_runtime import read,write,sha,call_plan,stage_model,generate_record,call_path,save_call,load_call,strategy_costs

RUNNER_VERSION='pa3-task3-live-runner-v1'
SOURCE_FILES=('task3_live_runtime.py','task3_live_run.py','task3_live_gpu_setup.py','test_task3_live_runner_cpu.py')

def expected_packages(task):
    assets=read(task/'configs/live_assets.json');result=dict(assets['cpu_packages'])
    result.update(read(task/'configs/scorer_frozen.json')['packages']);return result

def identity(task,check_references=False):
    assets=read(task/'configs/live_assets.json');receipt=read(task/'results/live_assets_preflight.json')
    assert receipt['passed'] and receipt['assets_sha256']==sha(task/'configs/live_assets.json')
    assert assets['model_id']==MODEL_ID and assets['revision']==MODEL_REVISION
    for name,value in assets['source_sha256'].items():assert sha(task/'code'/name)==value,name
    for name,value in read(task/'configs/scorer_frozen.json')['source_sha256'].items():assert sha(task/'code'/name)==value,name
    checks={'configs/scorer_frozen.json':'scorer_sha256','results/offline/finished.json':'offline_finished_sha256',
      'splits/question_splits.json':'question_splits_sha256','splits/live_math_questions.json':'selected_questions_sha256',
      'data/live_prompts.json':'prompts_sha256','configs/live_recipe.json':'recipe_sha256'}
    if check_references:checks['data/live_math_references.json']='references_sha256'
    for relative,key in checks.items():assert sha(task/relative)==assets[key],relative
    assert read(task/'configs/live_recipe.json')==recipe()
    prompts=read(task/'data/live_prompts.json')['questions'];plan=call_plan(prompts)
    split=read(task/'splits/question_splits.json')['datasets']['math'];assert [q['sample_id'] for q in prompts]==split['live_math']==split['evaluation'][:50]
    for name,record in assets['files'].items():assert (Path(assets['source_path'])/name).stat().st_size==record['bytes'],name
    return {'version':RUNNER_VERSION,'assets_sha256':sha(task/'configs/live_assets.json'),
      'prompts_sha256':assets['prompts_sha256'],'recipe_sha256':assets['recipe_sha256'],
      'source_sha256':{name:sha(task/'code'/name) for name in SOURCE_FILES},
      'physical_calls':250,'main_call_order':['parallel_0','parallel_1','parallel_2','single_long','revision'],
      'generation_only':True,'timing':'CUDA-synchronized prompt construction/tokenization, input transfer, generation, output decode and balanced-box extraction; exclude loading, smoke and disk persistence'},prompts,plan

def cpu_prepare(task):
    ident,prompts,plan=identity(task,True);expected=expected_packages(task)
    for name,value in expected.items():assert metadata.version(name)==value,(name,metadata.version(name),value)
    lock=task/'configs/live_runner_frozen.json'
    if lock.exists():assert read(lock)==ident,'Live runner identity changed.'
    else:write(lock,ident)
    write(task/'configs/live_call_plan.json',{'identity':ident,'calls':plan})
    report={'passed':True,'identity':ident,'questions':len(prompts),'physical_calls':len(plan),
      'max_physical_output_tokens':sum(c['max_new_tokens'] for c in plan),'shared_initial_reused':True,
      'weights_downloaded':True,'model_loaded':False,'gpu_used':False,'runner_sha256':sha(lock)}
    write(task/'results/live_runner_cpu_preflight.json',report);print(json.dumps(report,indent=2))

def gpu_environment(task):
    assert torch.cuda.is_available() and torch.version.cuda is not None,'Select a CUDA GPU runtime.'
    expected=expected_packages(task)
    for name,value in expected.items():
        actual=metadata.version(name)
        assert (actual.split('+')[0]==value.split('+')[0] if name=='torch' else actual==value),(name,actual,value)
    value={'gpu':torch.cuda.get_device_name(0),'compute_capability':list(torch.cuda.get_device_capability(0)),
      'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'cuda':torch.version.cuda,'python':platform.python_version(),
      'packages':{name:metadata.version(name) for name in expected}}
    path=task/'configs/live_gpu_environment.json'
    if path.exists():assert read(path)==value,'GPU environment changed between live calls/resume sessions.'
    else:write(path,value)
    return value

def smoke(task,model,tokenizer,run_identity):
    path=task/'results/live/smoke.json'
    # Smoke/warmup runs in every new model-loading session, separately from the
    # recorded main calls. It never uses a held-out problem or a gold answer.
    started=time.perf_counter();q={'sample_id':'SMOKE_WARMUP','problem':'Compute 17 times 23. Show your working.'}
    warm=generate_record(model,tokenizer,q,'parallel_0',smoke=True,smoke_cap=16)
    q={'sample_id':'SMOKE_TIMED','problem':'Compute 19 times 27. Show your working.'}
    timed=generate_record(model,tokenizer,q,'parallel_0',smoke=True,smoke_cap=64)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert timed['output_tokens']>0 and timed['seconds']>0
    report={'passed':True,'identity':run_identity,'warmup':warm,'timed':timed,'seconds':time.perf_counter()-started,
      'timed_output_tokens_per_second':timed['output_tokens']/timed['seconds'],'note':'Separate warmup/smoke only, not one of the 250 main calls.'}
    write(path,report);print(f"GPU smoke PASS | {report['timed_output_tokens_per_second']:.1f} tokens/s | peak {timed['peak_gpu_allocated_bytes']/2**30:.2f} GiB",flush=True)
    return report

def run_all(task):
    started=time.perf_counter();ident,prompts,plan=identity(task)
    assert read(task/'configs/live_runner_frozen.json')==ident
    receipt=read(task/'results/live_runner_cpu_preflight.json');assert receipt['passed'] and receipt['identity']==ident
    environment=gpu_environment(task)
    run_identity={'runner_sha256':sha(task/'configs/live_runner_frozen.json'),
      'gpu_environment_sha256':sha(task/'configs/live_gpu_environment.json'),'assets_sha256':ident['assets_sha256']}
    root=task/'results/live';root.mkdir(parents=True,exist_ok=True);done=root/'finished.json'
    if done.exists():assert read(done)['identity']==run_identity;print('Live GPU queue already finished.',flush=True);return
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    assets=read(task/'configs/live_assets.json');local=Path('/content/pa3_task3_live_model')/MODEL_REVISION
    staging=stage_model(assets,local)
    from transformers import AutoTokenizer,AutoModelForCausalLM
    load_start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(str(local),local_files_only=True,trust_remote_code=False)
    model,info=AutoModelForCausalLM.from_pretrained(str(local),local_files_only=True,trust_remote_code=False,
      dtype=torch.float16,attn_implementation='sdpa',device_map={'':0},output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
    assert model.config.model_type=='qwen2' and not getattr(model.config,'quantization_config',None)
    model.eval().requires_grad_(False)
    assert all(p.device.type=='cuda' and p.dtype==torch.float16 for p in model.parameters())
    model.config.use_cache=True
    loading={'seconds':time.perf_counter()-load_start,'parameters':sum(p.numel() for p in model.parameters()),
      'dtype':'float16','quantization':False,'batch_size':1,'attn_implementation':'sdpa','model_revision':MODEL_REVISION}
    warmup=smoke(task,model,tokenizer,run_identity)
    session_path=root/f'session_{time.time_ns()}.json'
    session_base={'identity':run_identity,'staging':staging,'loading':loading,'smoke_seconds':warmup['seconds']}
    write(session_path,{**session_base,'status':'model_loaded_smoke_passed','seconds_to_model_ready':time.perf_counter()-started})
    directory=root/'calls';directory.mkdir(exist_ok=True);all_calls=[];new_calls=0;reused=0;costs=[]
    for question_index,q in enumerate(prompts):
        records={}
        for call in ('parallel_0','parallel_1','parallel_2','single_long','revision'):
            expected=plan[question_index*5+len(records)];assert expected['call']==call
            path=call_path(directory,question_index,call)
            parent_path=call_path(directory,question_index,'parallel_0')
            parent_hash=sha(parent_path) if call=='revision' else None
            saved=load_call(path,run_identity,expected,parent_hash)
            if saved is not None:row=saved;reused+=1
            else:
                initial=records['parallel_0']['response'] if call=='revision' else None
                row=generate_record(model,tokenizer,q,call,initial_response=initial)
                if call=='revision':
                    assert row['request']['messages'][2]['content']==records['parallel_0']['response']
                    row['initial_call_sha256']=parent_hash
                save_call(path,row,run_identity);new_calls+=1
            records[call]=row;all_calls.append(row)
            print(f"Saved/reused {len(all_calls)}/250 | question {question_index+1}/50 | {call} | {row['output_tokens']} tokens | {row['seconds']:.2f}s",flush=True)
        costs.append({'sample_id':q['sample_id'],'strategies':strategy_costs(records)})
    assert len(all_calls)==250
    physical_tokens=sum(c['output_tokens'] for c in all_calls);assert physical_tokens<=409600
    manifest={str(call_path(directory,index,call).relative_to(task)):sha(call_path(directory,index,call)) for index in range(50) for call in CALLS}
    write(root/'call_manifest.json',{'identity':run_identity,'files':manifest})
    write(root/'strategy_costs.json',{'identity':run_identity,'questions':costs,'shared_initial_charged_to_both':True})
    session={**session_base,'status':'queue_finished','new_calls':new_calls,'reused_calls':reused,
      'elapsed_seconds':time.perf_counter()-started}
    write(session_path,session)
    result={'passed':True,'identity':run_identity,'questions':50,'physical_calls':250,'strategy_comparisons':150,
      'physical_output_tokens':physical_tokens,'physical_retained_output_tokens':sum(c['retained_output_tokens'] for c in all_calls),
      'physical_generation_seconds':sum(c['seconds'] for c in all_calls),
      'peak_gpu_allocated_bytes':max(c['peak_gpu_allocated_bytes'] for c in all_calls),
      'peak_gpu_reserved_bytes':max(c['peak_gpu_reserved_bytes'] for c in all_calls),
      'call_manifest_sha256':sha(root/'call_manifest.json'),'strategy_costs_sha256':sha(root/'strategy_costs.json'),
      'session':session,'model_unchanged':True,'gold_answers_loaded_for_generation':False,
      'note':'Required live GPU generation complete. Score responses, create plots/notebook and organize repository on CPU.'}
    write(done,result);print(json.dumps(result,indent=2));print('Task 3 required GPU work finished.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run-all',action='store_true')
    args=parser.parse_args();cpu_prepare(args.task_dir) if args.prepare else run_all(args.task_dir)
