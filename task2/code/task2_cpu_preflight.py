"""CPU checks of saved assets, deterministic image processing and SigLIP features.

Runs one frozen vision forward on one selected training image; no feature cache,
Qwen download, training loop or GPU allocation. Unit tests use a tiny random Qwen.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from task2_prepare_images import ImageArchiveReader

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def preprocess_image(image):
    image=image.convert('RGB').resize((224,224),resample=Image.Resampling.BICUBIC)
    pixels=np.asarray(image,dtype=np.float32)/np.float32(255.)
    pixels=(pixels-np.float32(.5))/np.float32(.5)
    return np.ascontiguousarray(pixels.transpose(2,0,1))

def main(task):
    import torch
    from transformers import AutoTokenizer,SiglipVisionConfig,SiglipVisionModel
    from task2_vlm import Connector,encode_example
    assert metadata.version('transformers')=='5.20.2'
    review_path=Path(__file__).with_name('task2_label_review_notes.json')
    review=json.loads(review_path.read_text())
    for name,digest in review['reviewed_input_hashes'].items():
        assert sha256(task/name)==digest,f'Review inputs differ: {name}'
    (task/'results/label_review_notes.json').write_text(review_path.read_text())
    # CPU device is explicit, even if accidentally called from a CUDA runtime.
    torch.set_num_threads(min(4,torch.get_num_threads()))
    source=json.loads((task/'configs/vision_assets.json').read_text())
    source_dir=Path(source['source_path'])
    for name,record in source['files'].items():
        path=source_dir/name
        assert path.stat().st_size==record['bytes'] and sha256(path)==record['sha256'],name
    raw=json.loads((source_dir/'config.json').read_text())
    config=SiglipVisionConfig(**raw['vision_config'])
    assert (config.image_size,config.patch_size,config.hidden_size)==(224,16,768)
    preprocessing=json.loads((source_dir/'preprocessor_config.json').read_text())
    assert preprocessing['size']=={'height':224,'width':224}
    assert preprocessing['image_mean']==preprocessing['image_std']==[.5,.5,.5]
    assert preprocessing['resample']==3 and preprocessing['do_resize'] and preprocessing['do_rescale'] and preprocessing['do_normalize']
    assert abs(preprocessing['rescale_factor']-1/255)<1e-12
    image_manifest_path=task/'data/images/manifest.json'
    assert sha256(image_manifest_path)==json.loads((task/'results/image_preparation.json').read_text())['manifest_sha256']
    image_manifest=json.loads(image_manifest_path.read_text())
    calibration=json.loads((task/'splits/calibration_images.json').read_text())
    example={'image_split':'train2014','image_id':calibration['image_ids'][0]}
    with ImageArchiveReader(task/'data/images',image_manifest) as reader:image=reader.get(example)
    pixels=preprocess_image(image)
    assert pixels.shape==(3,224,224) and pixels.dtype==np.float32
    assert np.array_equal(pixels,preprocess_image(image))
    assert pixels.min()>=-1 and pixels.max()<=1
    started=time.perf_counter()
    model,loading=SiglipVisionModel.from_pretrained(str(source_dir),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa',output_loading_info=True)
    assert not loading.get('missing_keys'),loading
    assert not loading.get('mismatched_keys'),loading
    assert not loading.get('error_msgs'),loading
    assert all(name.startswith('text_model.') or name in ('logit_bias','logit_scale')
               for name in loading.get('unexpected_keys',[])),loading
    model=model.to('cpu').eval().requires_grad_(False)
    # Pooling is not needed for patch features.
    model.use_head=False
    assert not any(parameter.requires_grad for parameter in model.parameters())
    normalized=[]
    hook=model.post_layernorm.register_forward_hook(lambda module,inputs,output:normalized.append(output.detach()))
    try:
        with torch.inference_mode():features=model(pixel_values=torch.from_numpy(pixels[None])).last_hidden_state
    finally:hook.remove()
    assert features.shape==(1,196,768) and torch.isfinite(features).all()
    assert len(normalized)==1,'Final layer norm must run exactly once.'
    torch.testing.assert_close(features,normalized[0],rtol=0,atol=0)
    connector=Connector()
    projected=connector(features.clone())
    assert projected.shape==(1,196,1024)
    assert all(parameter.dtype==torch.float32 for parameter in connector.parameters())
    qwen_revision=json.loads((task/'configs/data_selection_config.json').read_text())['tokenizer_revision']
    qwen=task.parent/'task1/assets/base_model'/qwen_revision
    tokenizer=AutoTokenizer.from_pretrained(str(qwen),local_files_only=True)
    encoded=encode_example(tokenizer,'Does this image contain a cat?','yes')
    assert encoded['response_ids'][-1]==tokenizer.convert_tokens_to_ids('<|im_end|>')
    prompt=tokenizer.decode(encoded['before_ids']+encoded['after_ids'])
    assert prompt.startswith('<|im_start|>user\n')
    assert '<think>\n\n</think>' in prompt
    assert 'PA3_VISUAL_INSERTION_POINT' not in prompt
    report={'passed':True,'device':'cpu','training_image_checked':example['image_id'],
      'siglip_revision':source['revision'],'patch_shape':list(features.shape),'connector_shape':list(projected.shape),
      'vision_weights_missing_keys':sorted(loading.get('missing_keys',[])),
      'unused_text_tower_keys':len(loading.get('unexpected_keys',[])),
      'connector_parameters':sum(p.numel() for p in connector.parameters()),'final_layer_norm_calls':len(normalized),
      'features_already_final_normalized':True,'image_processing_deterministic':True,
      'label_pairs_reviewed':review['pairs_reviewed'],'label_pairs_flagged_for_ambiguity':review['flagged_pairs'],
      'feature_cache_created':False,'training_started':False,'gpu_used':False,
      'vision_check_seconds':time.perf_counter()-started,
      'preprocessing':{'RGB':True,'resize':[224,224],'PIL_resample':'BICUBIC','rescale':'float32 / 255','normalize':'(x - 0.5) / 0.5','random_augmentation':False},
      'environment':{name:metadata.version(name) for name in ['torch','transformers','numpy','Pillow']},
      'code_sha256':{name:sha256(Path(__file__).with_name(name)) for name in ['task2_cpu_preflight.py','task2_vlm.py']}}
    (task/'results/vlm_cpu_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',required=True,type=Path)
    main(parser.parse_args().task_dir)
