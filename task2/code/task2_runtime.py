"""Shared CPU-safe file identities, recipes, token preparation and RNG state."""
from __future__ import annotations
import copy, hashlib, importlib.metadata as metadata, json, math, os, random, shutil
from pathlib import Path
import numpy as np
import torch

SEED=6304
STAGES={1:{'updates':2250,'connector_lr':1e-3,'adapter_lr':0.},
        2:{'updates':1500,'connector_lr':1e-4,'adapter_lr':2e-5},
        3:{'updates':750,'connector_lr':5e-5,'adapter_lr':1e-5}}
ENDPOINTS={1:2250,2:3750,3:4500}
DEV_UPDATES={0,500,1000,1500,2000,2250,2500,3000,3500,3750,4000,4500}
BENCH_CONDITIONS=('correct','complementary','mismatched','no_image')
CODE_FILES=('task2_runtime.py','task2_sampler.py','task2_features.py','task2_train.py','task2_evaluate.py','task2_run.py','task2_vlm.py','task2_answers.py','task2_cpu_preflight.py','task2_prepare_images.py')

def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def json_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)
def torch_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');torch.save(value,temp);temp.replace(path)
def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])

def gpu_environment(task):
    assert torch.cuda.is_available() and torch.version.cuda is not None,'Select a CUDA GPU only when ready.'
    expected=read(task/'configs/cpu_environment.json')['packages']
    for name in ('transformers','datasets','accelerate','peft','Pillow','numpy'):
        assert metadata.version(name)==expected[name],f'{name}: expected {expected[name]}, found {metadata.version(name)}'
    extra=read(task/'results/runner_cpu_preflight.json')['packages']
    for name,version in extra.items():assert metadata.version(name)==version,f'{name}: expected {version}, found {metadata.version(name)}'
    assert metadata.version('torch').split('+')[0]==expected['torch'].split('+')[0]
    value={'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
           'packages':{n:metadata.version(n) for n in ('torch','transformers','datasets','accelerate','peft','Pillow','numpy','tokenizers','huggingface_hub')}}
    path=task/'configs/gpu_environment_task2.json'
    if path.exists():assert read(path)==value,'GPU environment changed since the run began.'
    else:json_write(path,value)
    return value

def qwen_source(task):
    revision=read(task/'configs/data_selection_config.json')['tokenizer_revision']
    return task.parent/'task1/assets/base_model'/revision

def recipe():
    return {'seed':SEED,'stages':STAGES,'microbatch':8,'accumulation':2,'effective_batch':16,
      'weight_decay':.01,'betas':[.9,.999],'epsilon':1e-8,'clip_norm':1.,
      'warmup_rule':'ceil(0.05 * stage updates); LambdaLR starts at zero',
      'cosine_final_fraction':.1,'optimizer_policy':'fresh AdamW/scheduler each stage; continue dynamic scaler; decay all trainable tensors including connector biases',
      'stage1_order':'two independently seeded permutations of the 18000 saved captions',
      'stage2_draw_probabilities':{'presence':.5,'counting':.25,'caption':.25},
      'stage3_draw_probabilities':{'natural':.7,'presence':.1,'counting':.1,'caption':.1},
      'draw_size':2,'sampling':'replacement in Stages 2/3; complete pairs or two caption examples per draw, no exact quotas',
      'checkpoint_every':250,'development_updates':sorted(DEV_UPDATES),'generation_cap':16,'generation_batch':8,
      'loss':'mean token CE within each example, then mean over effective batch; EOT supervised',
      'calibration':'128 fixed training images; RMS-match initial connector outputs to embedding RMS of all three associated training captions and their questions; fixed scalar thereafter'}

def code_identity():
    root=Path(__file__).parent
    return {name:sha256(root/name) for name in CODE_FILES}

def load_identity(task):
    config=read(task/'configs/training_recipe.json');assert config==json.loads(json.dumps(recipe()))
    prep=read(task/'results/runner_cpu_preflight.json');assert prep['passed']
    assert prep['code_sha256']==code_identity(),'Runner code changed after CPU preparation.'
    for name,h in prep['input_sha256'].items():assert sha256(task/name)==h,name
    for name,h in read(task/'splits/selection_manifest.json')['files'].items():assert sha256(task/name)==h,name
    for name,record in read(task/'configs/qwen_assets_task2.json')['files'].items():
        assert (qwen_source(task)/name).stat().st_size==record['bytes'],name
    manifest=read(task/'features/patch196/manifest.json')
    return {'recipe_sha256':digest(config),'code_sha256':prep['code_sha256'],
            'inputs_sha256':prep['input_sha256'],'feature_manifest_sha256':sha256(task/'features/patch196/manifest.json')}

def prepare(task):
    from transformers import AutoTokenizer
    from task2_vlm import encode_example
    assert read(task/'results/vlm_cpu_preflight.json')['passed']
    assert read(task/'results/image_preparation.json')['passed']
    selected=read(task/'splits/selection_manifest.json')
    for name,h in selected['files'].items():assert sha256(task/name)==h,name
    tokenizer=AutoTokenizer.from_pretrained(str(qwen_source(task)),local_files_only=True)
    source=qwen_source(task)
    source_names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json']
    source_names.extend(p.name for p in sorted(source.glob('*.safetensors')))
    assert any(name.endswith('.safetensors') for name in source_names)
    qwen_assets={'revision':source.name,'files':{name:{'bytes':(source/name).stat().st_size,'sha256':sha256(source/name)} for name in source_names}}
    asset_path=task/'configs/qwen_assets_task2.json'
    if asset_path.exists():assert read(asset_path)==qwen_assets,'Original Qwen asset identity changed.'
    else:json_write(asset_path,qwen_assets)
    tokens={};all_rows=[];encoded_cache={}
    for name in ('captions_train','captions_dev','grounding_train','grounding_dev','grounding_test','natural_train','natural_dev','natural_test'):
        value=read(task/f'splits/{name}.json')
        rows=value if name.startswith('captions') else [e for p in value for e in p['examples']]
        all_rows.extend(rows)
        for row in rows:
            key=row['example_id']
            kind='caption' if name.startswith('captions') else 'answer'
            cache_key=(row['question'],row['target'],kind)
            if cache_key not in encoded_cache:
                encoded_cache[cache_key]=encode_example(tokenizer,row['question'],row['target'],kind)
            encoded=encoded_cache[cache_key]
            if key in tokens:assert tokens[key]==encoded
            else:tokens[key]=encoded
    destination=task/'data/tokenized_examples.json'
    if destination.exists():assert read(destination)==tokens,'Token preparation changed.'
    else:json_write(destination,tokens)
    calibration=read(task/'splits/calibration_images.json')
    images=set(calibration['image_ids']);train=read(task/'splits/captions_train.json')
    calibration_tokens=[]
    for row in train:
        if row['image_id'] in images:
            calibration_tokens.extend(tokenizer(row['question'],add_special_tokens=False)['input_ids'])
            calibration_tokens.extend(tokens[row['example_id']]['content_ids'])
    assert len(images)==128 and sum(r['image_id'] in images for r in train)==384
    json_write(task/'data/calibration_text_ids.json',calibration_tokens)
    review=read(task/'results/label_review_notes.json')
    flagged={r['pair_id'] for r in review['pairs'] if r['notes']}
    illustrative={}
    for suite in ('grounding','natural'):
        pairs=sorted(read(task/f'splits/{suite}_test.json'),key=lambda r:r['pair_id'])
        illustrative[suite]=[]
        for typ in sorted({p['question_type'] for p in pairs}):
            options=[p for p in pairs if p['question_type']==typ and p['pair_id'] not in flagged]
            assert options
            illustrative[suite].append(options[0]['pair_id'])
    json_write(task/'splits/qualitative_example_ids.json',{'rule':'First pair in sorted pair-ID order per question type, excluding grounding pairs flagged in the pre-training contact-sheet review. Fixed before predictions.', 'pairs':illustrative})
    json_write(task/'configs/training_recipe.json',recipe())
    names=['configs/training_recipe.json','configs/data_selection_config.json','configs/vision_assets.json','configs/qwen_assets_task2.json',
           'configs/cpu_environment.json','results/vlm_cpu_preflight.json',
           'splits/selection_manifest.json','splits/image_controls.json','splits/calibration_images.json',
           'splits/qualitative_example_ids.json',
           'data/tokenized_examples.json','data/calibration_text_ids.json','data/images/manifest.json']
    report={'passed':True,'tokenized_examples':len(tokens),'calibration_images':128,'calibration_caption_examples':384,
      'packages':{name:metadata.version(name) for name in ('tokenizers','huggingface_hub')},
      'calibration_text_tokens':len(calibration_tokens),'code_sha256':code_identity(),
      'input_sha256':{name:sha256(task/name) for name in names},'gpu_used':False,'feature_cache_required_before_training':True}
    json_write(task/'results/runner_cpu_preflight.json',report)
    print(json.dumps(report,indent=2),flush=True)
