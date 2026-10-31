"""CPU selection of fixed Task 2 captions, natural pairs and annotation pair pools.

No image/model downloads, feature computation, training or model predictions.
Counts and image-isolation requirements follow the ATML-PA3 Task 2 manual.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata as metadata
import itertools
import json
from pathlib import Path
import platform

from task2_answers import VERSION as NORMALIZATION_VERSION, normalize_answer

VERSION='pa3-task2-data-v1'
SEED=6304
NATURAL_QUOTAS={'train':{'yes/no':640,'number':256,'other':640},
                'dev':{'yes/no':12,'number':8,'other':12},
                'test':{'yes/no':36,'number':24,'other':36}}
GROUND_QUOTAS={'train':{'presence':4096,'counting':4096},'dev':{'presence':12,'counting':12},'test':{'presence':32,'counting':32}}
PLURALS={'person':'people','bus':'buses','sheep':'sheep','mouse':'mice','knife':'knives',
         'bench':'benches','sandwich':'sandwiches','toothbrush':'toothbrushes',
         'wine glass':'wine glasses','skis':'skis','scissors':'pairs of scissors','broccoli':'broccoli pieces','tv':'TVs'}

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temp.replace(path)

def records(path,key):
    import ijson
    with Path(path).open('rb') as f:yield from ijson.items(f,key+'.item')

def batches(items,size=512):
    iterator=iter(items)
    while batch:=list(itertools.islice(iterator,size)):yield batch

def tokenizer_lengths(tokenizer,texts):
    return [len(ids) for ids in tokenizer(texts,add_special_tokens=False,return_attention_mask=False)['input_ids']]

def load_captions(path,split,tokenizer):
    print('Reading captions:',split,flush=True)
    images={int(r['id']):{'image_id':int(r['id']),'image_split':split,'file_name':r['file_name'],
                        'width':int(r['width']),'height':int(r['height'])} for r in records(path,'images')}
    captions=defaultdict(list)
    retained=discarded=0
    for batch in batches(records(path,'annotations')):
        texts=[r['caption'] for r in batch]
        for row,length in zip(batch,tokenizer_lengths(tokenizer,texts)):
            if not row['caption'].strip() or length==0 or length>64:
                discarded+=1;continue
            image=int(row['image_id'])
            captions[image].append({'example_id':f"caption:{split}:{row['id']}", 'caption_id':int(row['id']),
              'image_id':image,'image_split':split,'question':'Describe this image.',
              'target':row['caption'],'target_content_tokens':length})
            retained+=1
    for rows in captions.values():rows.sort(key=lambda r:r['caption_id'])
    return images,dict(captions),{'eligible_captions':retained,'empty_or_overlength':discarded}

def load_instances(path):
    print('Reading object counts:',path.name,flush=True)
    categories={int(r['id']):r['name'] for r in records(path,'categories')}
    counts=defaultdict(Counter);crowds=defaultdict(set);present=defaultdict(set)
    for row in records(path,'annotations'):
        image,category=int(row['image_id']),int(row['category_id'])
        present[category].add(image)
        if row.get('iscrowd',0):crowds[category].add(image)
        else:counts[category][image]+=1
    return {'categories':categories,'counts':dict(counts),'crowds':dict(crowds),'present':dict(present)}

def load_natural(directory,split,tokenizer,eligible_images):
    pairs=json.loads((directory/f'v2_mscoco_{split}_complementary_pairs.json').read_text())
    pair_indices={};needed=set()
    for index,ids in enumerate(pairs):
        assert len(ids)==2
        key=tuple(sorted(map(int,ids)))
        if key[0]!=key[1] and key not in pair_indices:
            pair_indices[key]=index;needed.update(key)
    questions={int(r['question_id']):r for r in records(directory/f'v2_OpenEnded_mscoco_{split}_questions.json','questions') if int(r['question_id']) in needed}
    eligible={};stats=Counter({'source_pairs':len(pairs),'distinct_pairs':len(pair_indices)})
    print('Normalizing VQAv2 answers:',split,flush=True)
    for row in records(directory/f'v2_mscoco_{split}_annotations.json','annotations'):
        qid=int(row['question_id']); image=int(row['image_id'])
        if qid not in needed or image not in eligible_images:continue
        q=questions[qid]
        assert int(q['image_id'])==image
        references=[answer['answer'] for answer in row['answers']]
        if len(references)!=10:stats['not_ten_answers']+=1;continue
        normalized=[normalize_answer(answer) for answer in references]
        canonical,agreement=Counter(normalized).most_common(1)[0]
        if not canonical or agreement<8:stats['low_agreement_or_empty']+=1;continue
        # Retain an original string, never the normalized replacement, as target.
        target=next(answer for answer,norm in zip(references,normalized) if norm==canonical)
        eligible[qid]={'example_id':f'vqa:{split}:{qid}','question_id':qid,'image_id':image,'image_split':split,
          'question':q['question'],'target':target,'canonical_answer':canonical,'answers':references,
          'normalized_answers':normalized,'consensus_count':agreement,'answer_type':row['answer_type']}
    accepted=defaultdict(list)
    length_cache={}
    def length(text):
        if text not in length_cache:length_cache[text]=tokenizer_lengths(tokenizer,[text])[0]
        return length_cache[text]
    for (qid1,qid2),index in sorted(pair_indices.items()):
        if qid1 not in eligible or qid2 not in eligible:continue
        first,second=eligible[qid1],eligible[qid2]
        if first['question']!=second['question']:stats['different_question_text']+=1;continue
        if first['image_id']==second['image_id']:stats['same_image']+=1;continue
        if first['canonical_answer']==second['canonical_answer']:stats['same_normalized_consensus']+=1;continue
        typ=first['answer_type']
        if typ!=second['answer_type'] or typ not in ('yes/no','number','other'):
            stats['different_or_unknown_answer_type']+=1;continue
        if length(first['question'])>80 or any(length(e['target'])>16 for e in (first,second)):
            stats['over_content_cap']+=1;continue
        if not length(first['question']) or any(not length(e['target']) for e in (first,second)):
            stats['empty_tokens']+=1;continue
        accepted[typ].append({'pair_id':f'vqav2:{split}:{qid1}:{qid2}','source_pair_index':index,
          'suite':'natural','question_type':typ,'question':first['question'],'examples':[first,second]})
    stats['eligible_by_type']={typ:len(accepted[typ]) for typ in ('yes/no','number','other')}
    print('Eligible natural pairs:',stats['eligible_by_type'],flush=True)
    return dict(accepted),dict(stats)

def choose_natural(candidates,quotas,rng,used_images):
    selected=[]
    # Rarer number pairs first; disjoint images within and across evaluation sets.
    for typ in ('number','yes/no','other'):
        count=0
        for index in rng.permutation(len(candidates[typ])):
            pair=candidates[typ][int(index)]
            images={e['image_id'] for e in pair['examples']}
            if images & used_images:continue
            used_images.update(images);selected.append(pair);count+=1
            if count==quotas[typ]:break
        assert count==quotas[typ],f'Insufficient image-disjoint {typ} pairs: {count}/{quotas[typ]}'
    return [selected[int(i)] for i in rng.permutation(len(selected))]

def question_for(kind,name):
    if kind=='counting':return f'How many {PLURALS.get(name,name+"s")} can you see?'
    noun=name if name=='broccoli' else ('any '+name if name in ('skis','scissors') else ('an ' if name[0].lower() in 'aeiou' else 'a ')+name)
    return f'Does this image contain {noun}?'

def make_grounding(kind,size,split,image_ids,instances,rng,used_images=None):
    categories=instances['categories'];available=set(image_ids)
    pools={}
    for category in sorted(categories):
        if kind=='presence':
            positive=sorted(available & instances['present'].get(category,set()))
            negative=sorted(available - instances['present'].get(category,set()))
            if positive and negative:pools[category]=[positive,negative]
        else:
            grouped=defaultdict(list)
            crowds=instances['crowds'].get(category,set())
            for image,count in instances['counts'].get(category,{}).items():
                if image in available and image not in crowds and 1<=count<=4:grouped[count].append(image)
            if len(grouped)>=2:pools[category]={count:sorted(ids) for count,ids in sorted(grouped.items())}
    assert pools,f'No eligible {kind} categories'
    selected=[];seen=set();attempts=0;category_order=[]
    while len(selected)<size:
        if not category_order:category_order=[int(c) for c in rng.permutation(sorted(pools))]
        category=category_order.pop();attempts+=1
        assert attempts<max(10000,size*100),f'Cannot build {size} {kind} pairs with available images'
        excluded=used_images if used_images is not None else set()
        pool=pools[category]
        if kind=='presence':
            if used_images is None:groups=pool
            else:groups=[[image for image in group if image not in excluded] for group in pool]
            if any(not group for group in groups):continue
            image1=int(rng.choice(groups[0]));image2=int(rng.choice(groups[1]));targets=['yes','no']
        else:
            groups={count:([image for image in ids if image not in excluded] if used_images is not None else ids) for count,ids in pool.items()}
            valid=[count for count,ids in groups.items() if ids]
            if len(valid)<2:continue
            c1,c2=map(int,rng.choice(valid,size=2,replace=False))
            image1=int(rng.choice(groups[c1]));image2=int(rng.choice(groups[c2]));targets=[str(c1),str(c2)]
        key=(category,min(image1,image2),max(image1,image2))
        if key in seen:continue
        seen.add(key)
        pair_id=f'grounding:{split}:{kind}:{category}:{key[1]}:{key[2]}'
        question=question_for(kind,categories[category])
        examples=[{'example_id':f'{pair_id}:{i}','image_id':image,'image_split':split,'question':question,
          'target':target,'canonical_answer':normalize_answer(target),'answers':[target],
          'category_id':category} for i,(image,target) in enumerate(zip([image1,image2],targets))]
        if rng.integers(2):examples.reverse()
        selected.append({'pair_id':pair_id,'suite':'grounding','question_type':kind,
          'category_id':category,'category':categories[category],'question':question,'examples':examples})
        if used_images is not None:used_images.update((image1,image2))
    return selected

def choose_captions(image_ids,captions,count_per_image,rng):
    rows=[]
    for image in image_ids:
        options=captions[image]
        indices=rng.choice(len(options),size=count_per_image,replace=False)
        rows.extend(options[int(index)] for index in indices)
    return rows

def controls(pairs,rng):
    assert len(pairs)>1
    shuffled=[pairs[int(i)] for i in rng.permutation(len(pairs))]
    result=[]
    for index,pair in enumerate(shuffled):
        other=shuffled[(index+1)%len(shuffled)]
        original_ids={e['image_id'] for e in pair['examples']}
        for side,example in enumerate(pair['examples']):
            mismatched=other['examples'][int(rng.integers(2))]
            assert mismatched['image_id'] not in original_ids
            result.append({'example_id':example['example_id'],'pair_id':pair['pair_id'],
              'correct_image_id':example['image_id'],'complementary_image_id':pair['examples'][1-side]['image_id'],
              'mismatched_image_id':mismatched['image_id'],'mismatched_pair_id':other['pair_id'],
              'image_split':'val2014'})
    return result

def images_in(rows,pairs=False):
    return {e['image_id'] for row in rows for e in (row['examples'] if pairs else [row])}

def verify_outputs(outputs,instances,tokenizer):
    train=images_in(outputs['captions_train'])
    assert len(train)==6000 and len(outputs['captions_train'])==18000
    assert len(images_in(outputs['captions_dev']))==len(outputs['captions_dev'])==96
    assert all(value==3 for value in Counter(r['image_id'] for r in outputs['captions_train']).values())
    eval_groups={'captions_dev':images_in(outputs['captions_dev'])}
    for suite,quotas in [('natural',NATURAL_QUOTAS),('grounding',GROUND_QUOTAS)]:
        for split in ('train','dev','test'):
            rows=outputs[f'{suite}_{split}']
            assert len(rows)==sum(quotas[split].values())
            assert Counter(row['question_type'] for row in rows)==Counter(quotas[split])
            assert len({row['pair_id'] for row in rows})==len(rows)
            ids=images_in(rows,True)
            if split=='train':assert ids<=train
            else:
                assert len(ids)==2*len(rows)
                eval_groups[f'{suite}_{split}']=ids
            for row in rows:
                first,second=row['examples']
                assert first['image_id']!=second['image_id']
                assert first['canonical_answer']!=second['canonical_answer']
                assert all(e['question']==row['question'] for e in row['examples'])
                expected_split='train2014' if split=='train' else 'val2014'
                assert all(e['image_split']==expected_split for e in row['examples'])
                if suite=='natural':
                    for e in row['examples']:
                        assert len(e['answers'])==10
                        assert e['normalized_answers']==list(map(normalize_answer,e['answers']))
                        assert e['target'] in e['answers'] and normalize_answer(e['target'])==e['canonical_answer']
                        assert sum(answer==e['canonical_answer'] for answer in e['normalized_answers'])==e['consensus_count']>=8
                else:
                    source=instances[expected_split];category=row['category_id']
                    for e in row['examples']:
                        image=e['image_id']
                        if row['question_type']=='presence':
                            assert (image in source['present'].get(category,set()))==(e['target']=='yes')
                        else:
                            assert image not in source['crowds'].get(category,set())
                            assert source['counts'][category][image]==int(e['target'])
    groups={'training':train,**eval_groups}
    for first,second in itertools.combinations(groups,2):
        assert not groups[first]&groups[second],f'Image leakage: {first}/{second}'
    all_examples=list(outputs['captions_train'])+list(outputs['captions_dev'])
    for name,rows in outputs.items():
        if name.startswith(('natural_','grounding_')):all_examples.extend(e for row in rows for e in row['examples'])
    for batch in batches(all_examples):
        qlengths=tokenizer_lengths(tokenizer,[e['question'] for e in batch])
        tlengths=tokenizer_lengths(tokenizer,[e['target'] for e in batch])
        for e,q,t in zip(batch,qlengths,tlengths):
            assert 0<q<=80
            assert 0<t<=(64 if e['example_id'].startswith('caption:') else 16)
    return {'training_images':len(train),'evaluation_images_by_set':{name:len(ids) for name,ids in eval_groups.items()},
            'total_selected_images':sum(len(ids) for ids in groups.values()),'image_disjoint':True}

def main(task_dir,qwen_source):
    import numpy as np
    from transformers import AutoTokenizer
    source_manifest=task_dir/'configs/metadata_sources.json'
    selection_config={'version':VERSION,'seed':SEED,'metadata_manifest_sha256':sha256(source_manifest),
      'tokenizer_revision':qwen_source.name,'tokenizer_config_sha256':sha256(qwen_source/'tokenizer_config.json'),
      'tokenizer_json_sha256':sha256(qwen_source/'tokenizer.json'),
      'selection_code_sha256':sha256(Path(__file__)),
      'answers_code_sha256':sha256(Path(__file__).with_name('task2_answers.py')),
      'normalization':NORMALIZATION_VERSION,'natural_pair_quotas':NATURAL_QUOTAS,'grounding_pair_quotas':GROUND_QUOTAS,
      'caps':{'question':80,'caption_target':64,'answer_target':16},
      'caption_training_images':6000,'captions_per_training_image':3,'caption_dev_images':96,
      'selection_rule':'One NumPy default_rng(6304). Captions token-filtered first; natural train then val test then val dev; rare number pairs first; unique images per natural split. Caption train includes all natural train images, filled to 6000; grounding train draws only these images. Grounding test then dev are selected before caption dev from remaining val images.',
      'counting_rule':'Non-crowd instances, counts 1..4, excluding any image/category with crowd annotations.',
      'presence_rule':'Positive if any annotation for category, including crowd; negative if no annotation for that category.',
      'content_rule':'Filter over-cap/empty text before selection; preserve original target strings without truncation.',
      'mismatch_rule':'Final suite pair permutation, cyclic next-pair mapping; choose one of that pair images per example; excludes both original pair images.',
      'train_image_reuse':'Natural training images are included in the 6000 caption training images to limit image downloads/feature caching. Training images may recur across stages.',
      'packages':{name:metadata.version(name) for name in ['transformers','tokenizers','numpy','ijson']}}
    manifest_path=task_dir/'splits/selection_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        assert manifest['config']==selection_config,'Selection identity changed; preserve the existing split rather than silently replacing it.'
        for filename,digest in manifest['files'].items():assert sha256(task_dir/filename)==digest,filename
        print('Reusing verified data selection.',flush=True)
        print(json.dumps(json.loads((task_dir/'results/data_selection.json').read_text()),indent=2),flush=True)
        return
    directory=task_dir/'data/annotations'
    metadata_manifest=json.loads(source_manifest.read_text())
    print('Checking source annotation hashes.',flush=True)
    for source in metadata_manifest['sources'].values():
        for row in source['json_files']:
            path=directory/row['filename']
            assert path.stat().st_size==row['bytes'] and sha256(path)==row['sha256'],path
    tokenizer=AutoTokenizer.from_pretrained(str(qwen_source),local_files_only=True)
    rng=np.random.default_rng(SEED)
    images={};captions={};stats={}
    for split in ('train2014','val2014'):
        images[split],captions[split],stats[split+'_captions']=load_captions(directory/f'captions_{split}.json',split,tokenizer)
    eligible_train={image for image,rows in captions['train2014'].items() if len(rows)>=3}
    eligible_val=set(images['val2014'])
    natural_train_candidates,stats['natural_train']=load_natural(directory,'train2014',tokenizer,eligible_train)
    natural_train=choose_natural(natural_train_candidates,NATURAL_QUOTAS['train'],rng,set())
    del natural_train_candidates
    natural_val_candidates,stats['natural_val']=load_natural(directory,'val2014',tokenizer,eligible_val)
    used_val=set()
    natural_test=choose_natural(natural_val_candidates,NATURAL_QUOTAS['test'],rng,used_val)
    natural_dev=choose_natural(natural_val_candidates,NATURAL_QUOTAS['dev'],rng,used_val)
    del natural_val_candidates
    natural_images=images_in(natural_train,True)
    extra=[int(i) for i in rng.choice(sorted(eligible_train-natural_images),size=6000-len(natural_images),replace=False)]
    train_images=sorted(natural_images)+extra
    train_images=[train_images[int(i)] for i in rng.permutation(len(train_images))]
    captions_train=choose_captions(train_images,captions['train2014'],3,rng)
    instances={split:load_instances(directory/f'instances_{split}.json') for split in ('train2014','val2014')}
    outputs={'captions_train':captions_train,'natural_train':natural_train,'natural_dev':natural_dev,'natural_test':natural_test}
    for split in ('train','test','dev'):
        source='train2014' if split=='train' else 'val2014'
        allowed=train_images if split=='train' else sorted(eligible_val)
        used=None if split=='train' else used_val
        outputs['grounding_'+split]=[]
        for kind in ('presence','counting'):
            outputs['grounding_'+split].extend(make_grounding(kind,GROUND_QUOTAS[split][kind],source,allowed,instances[source],rng,used))
    caption_dev_candidates=sorted(set(captions['val2014'])-used_val)
    caption_dev_images=[int(i) for i in rng.choice(caption_dev_candidates,size=96,replace=False)]
    outputs['captions_dev']=choose_captions(caption_dev_images,captions['val2014'],1,rng)
    report=verify_outputs(outputs,instances,tokenizer)
    image_controls={name:controls(outputs[name],rng) for name in ('grounding_test','natural_test')}
    calibration=[int(i) for i in rng.choice(sorted(train_images),size=128,replace=False)]
    selected_images=[]
    for split in ('train2014','val2014'):
        wanted=set(train_images) if split=='train2014' else set().union(*(images_in(outputs[name],name!='captions_dev') for name in ['captions_dev','natural_dev','natural_test','grounding_dev','grounding_test']))
        selected_images.extend(images[split][image] for image in sorted(wanted))
    outputs['image_controls']=image_controls
    outputs['calibration_images']={'image_split':'train2014','image_ids':calibration,'selection':'128 training images; reserved for connector/text embedding magnitude calibration, never evaluation images.'}
    outputs['selected_images']=selected_images
    files={}
    for name,value in outputs.items():
        path=task_dir/f'splits/{name}.json';json_write(path,value);files[str(path.relative_to(task_dir))]=sha256(path)
    json_write(task_dir/'configs/data_selection_config.json',selection_config)
    report.update({'passed':True,'seed':SEED,'caption_train_examples':18000,'caption_dev_examples':96,
      'natural_pairs':{name:len(outputs['natural_'+name]) for name in ('train','dev','test')},
      'natural_pair_types':NATURAL_QUOTAS,'grounding_pairs':{name:len(outputs['grounding_'+name]) for name in ('train','dev','test')},
      'grounding_pair_types':GROUND_QUOTAS,'image_controls_fixed':True,'token_caps_checked':True,
      'normalization':NORMALIZATION_VERSION,'feature_storage_GiB_196_tokens':len(selected_images)*196*768*2/2**30,
      'filter_statistics':stats,'images_downloaded':False,'gpu_used':False})
    json_write(task_dir/'results/data_selection.json',report)
    files['results/data_selection.json']=sha256(task_dir/'results/data_selection.json')
    files['configs/data_selection_config.json']=sha256(task_dir/'configs/data_selection_config.json')
    json_write(manifest_path,{'config':selection_config,'files':files})
    print('\nData selection PASS.',flush=True)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir',required=True,type=Path)
    parser.add_argument('--qwen-source',required=True,type=Path)
    args=parser.parse_args();main(args.task_dir,args.qwen_source)
