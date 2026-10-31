"""Three-stage FP16 AMP training with complete 250-update recovery states."""
from __future__ import annotations
import argparse,copy,json,math,time
from pathlib import Path
import torch
from task2_runtime import (SEED,STAGES,ENDPOINTS,DEV_UPDATES,read,sha256,digest,json_write,torch_write,
  cpu_tree,seed_all,rng_state,restore_rng,gpu_environment,qwen_source,load_identity)
from task2_vlm import VisualLanguageModel,Connector,LoRALinear,calibrate_connector_scale
from task2_sampler import CurriculumSampler
from task2_features import FeatureStore
from task2_evaluate import development,final_evaluation,greedy

def model_state(model):
    return {'connector':cpu_tree(model.connector.state_dict()),
            'adapters':{name:module.factors() for name,module in model.backbone.named_modules() if isinstance(module,LoRALinear)}}

def load_model_state(model,state):
    model.connector.load_state_dict(state['connector'],strict=True)
    modules={name:module for name,module in model.backbone.named_modules() if isinstance(module,LoRALinear)}
    assert set(modules)==set(state['adapters'])
    with torch.no_grad():
        for name,module in modules.items():
            entry=state['adapters'][name]
            assert entry['scale']==module.alpha/module.rank
            module.A.copy_(entry['A']);module.B.copy_(entry['B'])

def build_model(task,adapters=False,device='cuda'):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    source=qwen_source(task)
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True)
    model,info=AutoModelForCausalLM.from_pretrained(str(source),local_files_only=True,dtype=torch.float16,
      attn_implementation='sdpa',output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
    assert model.config.hidden_size==1024 and model.config.num_hidden_layers==28
    assert sum(p.numel() for p in model.parameters())==596049920
    seed_all(SEED)
    vlm=VisualLanguageModel(model,Connector()).to(device)
    if adapters:
        targets=vlm.enable_lora();assert len(targets)==112
    vlm.backbone.config.use_cache=False
    vlm.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return vlm,tokenizer

def make_optimizer(model,stage):
    connector=list(model.connector.parameters())
    adapters=[p for name,p in model.backbone.named_parameters() if p.requires_grad]
    assert all(p.dtype==torch.float32 for p in connector+adapters)
    groups=[{'params':connector,'lr':STAGES[stage]['connector_lr'],'name':'connector'}]
    if stage>1:
        assert adapters;groups.append({'params':adapters,'lr':STAGES[stage]['adapter_lr'],'name':'lora'})
    else:assert not adapters
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
    count=STAGES[stage]['updates'];warmup=math.ceil(.05*count)
    def factor(step):
        if step<warmup:return step/warmup
        progress=min(1.,(step-warmup)/(count-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    return optimizer,scheduler

def perform_update(model,rows,tokens,store,optimizer,scaler):
    trainable=[p for p in model.parameters() if p.requires_grad]
    skipped=0;begin=time.perf_counter()
    for attempt in range(20):
        optimizer.zero_grad(set_to_none=True);total=0.
        for start in (0,8):
            chunk=rows[start:start+8];features=store.get(chunk,next(model.parameters()).device)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                losses,output=model([tokens[r['example_id']] for r in chunk],features)
                loss=losses.sum()/16
            total+=float(loss.detach());scaler.scale(loss).backward()
            del losses,loss,output,features
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,foreach=False)
        old=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old:
            skipped+=1;continue
        assert math.isfinite(total) and bool(torch.isfinite(norm))
        return {'loss':total,'gradient_norm_before_clip':float(norm),'scale':scaler.get_scale(),
                'skipped_attempts':skipped,'update_seconds':time.perf_counter()-begin}
    raise RuntimeError('20 consecutive AMP overflows; refusing to consume further examples.')

def pools(task):
    grounding=read(task/'splits/grounding_train.json')
    return CurriculumSampler(read(task/'splits/captions_train.json'),[p for p in grounding if p['question_type']=='presence'],
                             [p for p in grounding if p['question_type']=='counting'],read(task/'splits/natural_train.json'))

def initialize_calibration(model,task,store):
    begin=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    calibration=read(task/'splits/calibration_images.json')
    training_images={r['image_id'] for r in read(task/'splits/captions_train.json')}
    assert calibration['image_split']=='train2014' and set(calibration['image_ids'])<=training_images
    rows=[{'image_id':i,'image_split':'train2014'} for i in calibration['image_ids']]
    device=next(model.parameters()).device
    features=store.get(rows,device).float()
    token_ids=torch.tensor(read(task/'data/calibration_text_ids.json'),device=device,dtype=torch.long)
    record=calibrate_connector_scale(model,features,token_ids)
    record['seconds']=time.perf_counter()-begin
    record['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
    del features,token_ids
    return record

def latest(run):
    paths=[p for p in run.glob('update_*/training_state.pt') if p.with_name('complete.json').exists()]
    return max(paths,key=lambda p:int(p.parent.name.split('_')[1])) if paths else None

def save_checkpoint(run,model,optimizer,scheduler,scaler,sampler,stage,stage_update,update,identity,resources,skipped):
    path=run/f'update_{update:04d}/training_state.pt';begin=time.perf_counter()
    state={'identity':identity,'model':model_state(model),'optimizer':cpu_tree(optimizer.state_dict()),
      'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'sampler':sampler.state_dict(),
      'rng':rng_state(),'stage':stage,'stage_update':stage_update,'update':update,
      'examples_seen':update*16,'resources':copy.deepcopy(resources),'skipped_attempts':skipped}
    torch_write(path,state)
    marker={'update':update,'stage':stage,'bytes':path.stat().st_size,'sha256':sha256(path),'seconds':time.perf_counter()-begin}
    json_write(path.with_name('complete.json'),marker)
    print(f'Saved checkpoint {update}: {marker["bytes"]/2**20:.1f} MiB',flush=True)

def smoke(task):
    environment=gpu_environment(task);identity=load_identity(task)
    path=task/'results/gpu_smoke_task2.json'
    if path.exists():assert read(path)['identity']==identity;print('GPU smoke already passed.',flush=True);return
    store=FeatureStore(task,Path('/content/pa3_task2_work/feature_data')/identity['feature_manifest_sha256'][:16])
    model,tokenizer=build_model(task);calibration=initialize_calibration(model,task,store)
    sampler=pools(task);tokens=read(task/'data/tokenized_examples.json');results={}
    scaler=torch.amp.GradScaler('cuda');model.train()
    for stage in (1,2):
        if stage==2:assert len(model.enable_lora())==112
        optimizer,scheduler=make_optimizer(model,stage);torch.cuda.reset_peak_memory_stats();records=[]
        for _ in range(2):
            rows,_=sampler.batch(stage);record=perform_update(model,rows,tokens,store,optimizer,scaler);scheduler.step();records.append(record)
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.connector.parameters())
        assert all(p.grad is None for p in model.backbone.parameters() if not p.requires_grad)
        results[str(stage)]={'updates':records,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
                             'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    model.eval();rows=read(task/'splits/grounding_dev.json')[0]['examples']
    with torch.autocast('cuda',dtype=torch.float16):
        generated=greedy(model,[tokens[r['example_id']] for r in rows],store.get(rows,'cuda'),{tokenizer.eos_token_id},max_new_tokens=4)
    assert len(generated)==2
    json_write(path,{'passed':True,'identity':identity,'environment':environment,'calibration_smoke_only':calibration,'stages':results})
    print('Task 2 GPU smoke PASS.',flush=True)

def train(task):
    gpu_environment(task);identity=load_identity(task)
    assert read(task/'results/gpu_smoke_task2.json')['passed']
    assert read(task/'results/gpu_smoke_task2.json')['identity']==identity
    run=task/'checkpoints/main';run.mkdir(parents=True,exist_ok=True)
    if (run/'finished.json').exists():assert read(run/'finished.json')['identity']==identity;print('Training already complete.',flush=True);return
    seed_all();saved=None;checkpoint=latest(run)
    if checkpoint:
        assert sha256(checkpoint)==read(checkpoint.with_name('complete.json'))['sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['identity']==identity
    start_stage=saved['stage'] if saved else 1
    store=FeatureStore(task,Path('/content/pa3_task2_work/feature_data')/identity['feature_manifest_sha256'][:16])
    model,tokenizer=build_model(task,start_stage>1)
    tokens=read(task/'data/tokenized_examples.json');sampler=pools(task);scaler=torch.amp.GradScaler('cuda')
    resources={};skipped=0;update=0
    if saved:
        load_model_state(model,saved['model']);sampler.load_state_dict(saved['sampler'])
        resources=saved['resources'];skipped=saved['skipped_attempts'];update=saved['update']
        scaler.load_state_dict(saved['scaler'])
    else:
        calibration=initialize_calibration(model,task,store);json_write(run/'calibration.json',calibration)
        torch_write(run/'initial_connector.pt',model_state(model))
        development(task,model,tokens,store,0,identity)
    log_path=run/'training.jsonl'
    if log_path.exists():
        previous=log_path.read_text().splitlines();keep=[]
        for line in previous:
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if row['update']<=update:keep.append(line)
        if len(keep)!=len(previous):
            backup=log_path.with_name(f'training_before_resume_{time.time_ns()}.jsonl');log_path.rename(backup)
            log_path.write_text('\n'.join(keep)+ ('\n' if keep else ''))
    buffered=[]
    def flush():
        if buffered:
            with log_path.open('a') as stream:stream.write(''.join(json.dumps(r)+'\n' for r in buffered))
            buffered.clear()
    for stage in range(start_stage,4):
        if stage==2 and not any(isinstance(m,LoRALinear) for m in model.backbone.modules()):assert len(model.enable_lora())==112
        optimizer,scheduler=make_optimizer(model,stage)
        if saved and stage==start_stage:
            optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
            stage_update=saved['stage_update']
        else:stage_update=0
        key=str(stage)
        resources.setdefault(key,{'training_seconds':0.,'peak_gpu_allocated_bytes':0,'peak_gpu_reserved_bytes':0,
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        model.train();torch.cuda.reset_peak_memory_stats()
        while stage_update<STAGES[stage]['updates']:
            rows,mixture=sampler.batch(stage);lrs={g['name']:g['lr'] for g in optimizer.param_groups}
            record=perform_update(model,rows,tokens,store,optimizer,scaler)
            scheduler.step();stage_update+=1;update+=1;skipped+=record['skipped_attempts']
            resource=resources[key];resource['training_seconds']+=record['update_seconds']
            resource['peak_gpu_allocated_bytes']=max(resource['peak_gpu_allocated_bytes'],torch.cuda.max_memory_allocated())
            resource['peak_gpu_reserved_bytes']=max(resource['peak_gpu_reserved_bytes'],torch.cuda.max_memory_reserved())
            buffered.append({'update':update,'stage':stage,'stage_update':stage_update,'examples_seen':update*16,
              **record,'learning_rates':lrs,'mixture_draws':mixture,'cumulative_mixture_draws':dict(sampler.counts[stage])})
            if update%25==0:
                flush();print(f'Update {update}/4500 | stage {stage} | loss {record["loss"]:.4f} | {record["update_seconds"]:.2f}s',flush=True)
            if update%250==0:
                flush();save_checkpoint(run,model,optimizer,scheduler,scaler,sampler,stage,stage_update,update,identity,resources,skipped)
            if update in DEV_UPDATES:
                development(task,model,tokens,store,update,identity);torch.cuda.reset_peak_memory_stats()
        assert update==ENDPOINTS[stage]
        export=run/f'stage_{stage}_inference.pt';begin=time.perf_counter()
        torch_write(export,{'identity':identity,'stage':stage,'update':update,'model':model_state(model)})
        json_write(run/f'stage_{stage}_resources.json',{**resources[key],'mixture_draws':dict(sampler.counts[stage]),
          'export_bytes':export.stat().st_size,'export_seconds':time.perf_counter()-begin})
        saved=None
    flush();assert update==4500 and sampler.caption_position==36000
    assert all(sampler.draws[s]==STAGES[s]['updates']*8 for s in STAGES)
    result={'passed':True,'identity':identity,'updates':4500,'example_presentations':72000,
      'skipped_attempts':skipped,'stages':{str(s):read(run/f'stage_{s}_resources.json') for s in STAGES},
      'recovery_state_bytes':sum(p.stat().st_size for p in run.glob('update_*/training_state.pt')),
      'checkpoint_save_seconds':sum(read(p)['seconds'] for p in run.glob('update_*/complete.json')),
      'calibration':read(run/'calibration.json')}
    json_write(run/'finished.json',result);print('All three training stages finished.',flush=True)

def evaluate_final(task):
    gpu_environment(task);identity=load_identity(task)
    assert read(task/'checkpoints/main/finished.json')['passed']
    if (task/'results/final/finished.json').exists():
        assert read(task/'results/final/finished.json')['identity']==identity;print('Final evaluation already complete.',flush=True);return
    model,tokenizer=build_model(task,True)
    state=torch.load(task/'checkpoints/main/stage_3_inference.pt',map_location='cpu',weights_only=True)
    assert state['identity']==identity and state['update']==4500
    load_model_state(model,state['model']);del state
    store=FeatureStore(task,Path('/content/pa3_task2_work/feature_data')/identity['feature_manifest_sha256'][:16])
    final_evaluation(task,model,tokenizer,read(task/'data/tokenized_examples.json'),store,identity)
    print('Required final evaluations finished.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-dir',type=Path,required=True)
    action=p.add_mutually_exclusive_group();action.add_argument('--smoke',action='store_true');action.add_argument('--final',action='store_true')
    a=p.parse_args()
    if a.smoke:smoke(a.task_dir)
    elif a.final:evaluate_final(a.task_dir)
    else:train(a.task_dir)
