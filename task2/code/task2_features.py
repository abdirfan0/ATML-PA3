"""Frozen SigLIP FP16 feature caching and CPU-mapped feature lookup."""
from __future__ import annotations
import argparse, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from task2_runtime import read,sha256,digest,json_write,gpu_environment,seed_all
from task2_prepare_images import ImageArchiveReader
from task2_cpu_preflight import preprocess_image

def stage_file(source,destination,expected_sha):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or sha256(destination)!=expected_sha:
        temp=destination.with_name(destination.name+'.tmp');shutil.copyfile(source,temp)
        assert sha256(temp)==expected_sha;temp.replace(destination)
    return destination

class FeatureStore:
    def __init__(self,task,local_root=None):
        self.manifest=read(task/'features/patch196/manifest.json');self.maps=[];self.lookup={}
        for shard in self.manifest['shards']:
            source=task/'features/patch196'/shard['file_name']
            path=stage_file(source,local_root/shard['file_name'],shard['sha256']) if local_root else source
            if local_root is None:assert sha256(path)==shard['sha256']
            matrix=np.load(path,mmap_mode='r')
            assert matrix.shape==(len(shard['images']),196,768) and matrix.dtype==np.float16
            shard_index=len(self.maps);self.maps.append(matrix)
            for index,row in enumerate(shard['images']):
                key=(row['image_split'],row['image_id']);assert key not in self.lookup
                self.lookup[key]=(shard_index,index)
    def get(self,rows,device='cpu'):
        arrays=[]
        for row in rows:
            shard,index=self.lookup[(row['image_split'],int(row['image_id']))];arrays.append(self.maps[shard][index])
        tensor=torch.from_numpy(np.stack(arrays))
        if str(device).startswith('cuda'):tensor=tensor.pin_memory()
        return tensor.to(device,non_blocking=str(device).startswith('cuda'))

def load_vision(source,device,dtype):
    from transformers import SiglipVisionModel
    model,info=SiglipVisionModel.from_pretrained(str(source),local_files_only=True,dtype=dtype,attn_implementation='sdpa',output_loading_info=True)
    assert not info.get('missing_keys') and not info.get('mismatched_keys') and not info.get('error_msgs'),info
    assert all(k.startswith('text_model.') or k in ('logit_bias','logit_scale') for k in info.get('unexpected_keys',[])),info
    model.use_head=False
    return model.to(device).eval().requires_grad_(False)

def choose_batch(model,pixels,total_memory):
    records=[]
    for size in (32,64,128):
        try:
            torch.cuda.empty_cache();x=pixels.repeat(size,1,1,1)
            with torch.inference_mode():model(pixel_values=x)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(2):
                start=time.perf_counter()
                with torch.inference_mode():model(pixel_values=x)
                torch.cuda.synchronize();times.append(time.perf_counter()-start)
            record={'batch_size':size,'seconds':times,'images_per_second':size/(sum(times)/len(times)),
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'oom':False}
            del x
        except torch.OutOfMemoryError:
            if 'x' in locals():del x
            record={'batch_size':size,'oom':True};torch.cuda.empty_cache()
        records.append(record)
    eligible=[r for r in records if not r['oom'] and r['peak_allocated_bytes']<=.8*total_memory]
    assert eligible,'No feature batch meets the memory headroom rule.'
    chosen=max(eligible,key=lambda r:(r['images_per_second'],-r['batch_size']))['batch_size']
    return chosen,records

def cache_features(task,work):
    environment=gpu_environment(task);seed_all()
    preflight=read(task/'results/vlm_cpu_preflight.json');assert preflight['passed']
    for name,h in preflight['code_sha256'].items():assert sha256(Path(__file__).with_name(name))==h,name
    original_env=preflight['environment']
    for name in ('transformers','Pillow','numpy'):
        assert environment['packages'][name]==original_env[name]
    images=read(task/'data/images/manifest.json');vision=read(task/'configs/vision_assets.json')
    identity={'version':'pa3-task2-feature-v1','image_manifest_sha256':sha256(task/'data/images/manifest.json'),
      'vision_assets_sha256':sha256(task/'configs/vision_assets.json'),'code_sha256':sha256(Path(__file__)),
      'preprocessing_code_sha256':sha256(Path(__file__).with_name('task2_cpu_preflight.py')),
      'image_reader_code_sha256':sha256(Path(__file__).with_name('task2_prepare_images.py')),
      'environment':environment,'dtype':'float16','compute':'FP16 frozen SigLIP; last_hidden_state already final normalized once'}
    destination=task/'features/patch196';destination.mkdir(parents=True,exist_ok=True)
    identity_path=destination/'identity.json'
    if identity_path.exists():assert read(identity_path)==identity,'Feature identity changed.'
    else:json_write(identity_path,identity)
    if (destination/'finished.json').exists():
        manifest=read(destination/'manifest.json');assert manifest['identity']==identity
        for shard in manifest['shards']:assert sha256(destination/shard['file_name'])==shard['sha256']
        print('All 6528 cached features verified; skipping SigLIP.',flush=True);return
    started=time.perf_counter();work=work/digest(identity)[:16]
    image_local=work/'images'
    for shard in images['shards']:stage_file(task/'data/images'/shard['file_name'],image_local/shard['file_name'],shard['sha256'])
    model=load_vision(Path(vision['source_path']),'cuda',torch.float16)
    with ImageArchiveReader(task/'data/images',images,image_local) as reader:
        first=images['shards'][0]['images'][0]
        pixels=torch.from_numpy(preprocess_image(reader.get(first))[None]).to('cuda',dtype=torch.float16)
        benchmark_path=destination/'batch_benchmark.json'
        if benchmark_path.exists():benchmark=read(benchmark_path);batch=benchmark['selected']
        else:
            batch,records=choose_batch(model,pixels,torch.cuda.get_device_properties(0).total_memory)
            benchmark={'selected':batch,'records':records,'workload':'One selected training image repeated; two timed forwards per size; no answer labels; 80% memory cap'}
            json_write(benchmark_path,benchmark)
        del pixels
        torch.cuda.reset_peak_memory_stats();shards=[];forward_seconds=0.;normalization_checked=False
        for index,source_shard in enumerate(images['shards']):
            name=f'features_{index:04d}.npy';path=destination/name;marker_path=path.with_suffix('.json')
            if marker_path.exists():
                record=read(marker_path);assert record['identity_sha256']==digest(identity)
                assert sha256(path)==record['sha256'];assert record['images']==[{k:r[k] for k in ('image_id','image_split')} for r in source_shard['images']]
                normalization_checked=normalization_checked or record.get('final_layer_norm_once',False)
                shards.append(record);continue
            shard_started=time.perf_counter();shard_forward_seconds=0.
            matrix=np.empty((len(source_shard['images']),196,768),dtype=np.float16)
            for start in range(0,len(matrix),batch):
                rows=source_shard['images'][start:start+batch]
                array=np.stack([preprocess_image(reader.get(row)) for row in rows])
                pixels=torch.from_numpy(array).to('cuda',dtype=torch.float16)
                normalized=[];hook=None
                if not normalization_checked:hook=model.post_layernorm.register_forward_hook(lambda m,i,o:normalized.append(o))
                torch.cuda.synchronize();begin=time.perf_counter()
                try:
                    with torch.inference_mode():features=model(pixel_values=pixels).last_hidden_state
                    torch.cuda.synchronize();duration=time.perf_counter()-begin
                    forward_seconds+=duration;shard_forward_seconds+=duration
                finally:
                    if hook is not None:hook.remove()
                assert features.shape==(len(rows),196,768) and features.dtype==torch.float16 and torch.isfinite(features).all()
                if not normalization_checked:
                    assert len(normalized)==1;torch.testing.assert_close(features,normalized[0],rtol=0,atol=0);normalization_checked=True
                matrix[start:start+len(rows)]=features.cpu().numpy();del features,pixels,normalized
            local=work/name;local.parent.mkdir(parents=True,exist_ok=True)
            with local.open('wb') as stream:np.save(stream,matrix,allow_pickle=False)
            checksum=sha256(local);stage_file(local,path,checksum)
            record={'file_name':name,'identity_sha256':digest(identity),'sha256':checksum,'bytes':path.stat().st_size,
                    'final_layer_norm_once':normalization_checked,'vision_forward_seconds':shard_forward_seconds,
                    'elapsed_seconds':time.perf_counter()-shard_started,
                    'images':[{k:r[k] for k in ('image_id','image_split')} for r in source_shard['images']]}
            json_write(marker_path,record);shards.append(record)
            print(f'Cached {index+1}/{len(images["shards"])} feature shards',flush=True)
    assert sum(len(r['images']) for r in shards)==6528
    manifest={'identity':identity,'shards':shards,'images':6528,'shape_per_image':[196,768]}
    json_write(destination/'manifest.json',manifest)
    result={'passed':True,'images':6528,'shape_per_image':[196,768],'dtype':'float16','vision_forward_seconds_this_session':forward_seconds,
      'vision_forward_seconds_all_completed_shards':sum(s['vision_forward_seconds'] for s in shards),
      'completed_shard_seconds_sum':sum(s['elapsed_seconds'] for s in shards),
      'total_seconds_this_session':time.perf_counter()-started,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
      'bytes':sum(r['bytes'] for r in shards),'batch_size':batch,'final_layer_norm_once':normalization_checked,
      'note':'Resume-session timings exclude work from earlier interrupted sessions; shard records remain complete.'}
    json_write(destination/'finished.json',result);json_write(task/'results/feature_caching.json',result)
    del model;torch.cuda.empty_cache();print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-dir',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,default=Path('/content/pa3_task2_work/features'))
    a=p.parse_args();cache_features(a.task_dir,a.work_dir)
