"""Development NLL and final greedy image controls, with complete-pair bootstrap."""
from __future__ import annotations
from collections import defaultdict
import hashlib,math,time
import numpy as np
import torch
from task2_runtime import read,json_write,BENCH_CONDITIONS
from task2_answers import normalize_answer,vqa_soft_accuracy

@torch.inference_mode()
def greedy(model,examples,features,eos_ids,max_new_tokens=16):
    """KV-cache decoding; response labels/targets never enter the generation prefix."""
    batch=model.make_batch(examples,features,padding_side='left',include_response=False)
    batch.pop('labels');mask=batch['attention_mask'];position=batch['position_ids']
    tokens=[[] for _ in examples];finished=torch.zeros(len(examples),device=mask.device,dtype=torch.bool)
    past=None;next_ids=None
    for step in range(max_new_tokens):
        if step==0:output=model.backbone(**batch,use_cache=True,return_dict=True)
        else:output=model.backbone(input_ids=next_ids[:,None],attention_mask=mask,position_ids=position,
                                   past_key_values=past,use_cache=True,return_dict=True)
        past=output.past_key_values;next_ids=output.logits[:,-1].argmax(-1)
        values=next_ids.tolist();was_finished=finished.tolist()
        for i,value in enumerate(values):
            if not was_finished[i]:tokens[i].append(value)
        ended=torch.zeros_like(finished)
        for eos in eos_ids:ended|=next_ids==eos
        finished|=ended
        if bool(finished.all()):break
        mask=torch.cat([mask,mask.new_ones((len(examples),1))],dim=1)
        position=(mask.sum(-1)-1)[:,None]
        del output
    return [{'output_token_ids':ids,'output_tokens':len(ids),'ended_by_eos':bool(done),
             'truncated':not bool(done) and len(ids)==max_new_tokens} for ids,done in zip(tokens,finished.tolist())]

def flatten(pairs):return [example for pair in pairs for example in pair['examples']]

def controlled_rows(pairs,condition,mismatches=None):
    rows=[]
    for pair in pairs:
        for side,example in enumerate(pair['examples']):
            row=dict(example)
            if condition=='complementary':row['image_id']=pair['examples'][1-side]['image_id']
            elif condition=='mismatched':row['image_id']=mismatches[example['example_id']]['mismatched_image_id']
            elif condition not in ('correct','no_image'):raise ValueError(condition)
            rows.append(row)
    return rows

def reference_losses(model,rows,tokens,store,condition='correct',content_only=True,batch_size=8):
    values=[];device=next(model.parameters()).device
    with torch.inference_mode(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
        for start in range(0,len(rows),batch_size):
            chunk=rows[start:start+batch_size]
            features=None if condition=='no_image' else store.get(chunk,device)
            losses,output=model([tokens[row['example_id']] for row in chunk],features,content_only_labels=content_only)
            values.extend(losses.float().cpu().tolist());del output,losses,features
    assert len(values)==len(rows) and all(math.isfinite(v) for v in values)
    return values

def development(task,model,tokens,store,update,identity):
    directory=task/'results/development';path=directory/f'update_{update:04d}.json'
    if path.exists():
        saved=read(path);assert saved['identity']==identity;return saved
    training=model.training;model.eval();started=time.perf_counter();result={'update':update,'identity':identity,'suites':{}}
    captions=read(task/'splits/captions_dev.json')
    values=reference_losses(model,captions,tokens,store,content_only=False)
    result['caption_reference_loss']=float(np.mean(values))
    result['caption_examples']=[{'example_id':r['example_id'],'loss':v} for r,v in zip(captions,values)]
    for suite in ('grounding','natural'):
        pairs=read(task/f'splits/{suite}_dev.json');per_condition={}
        types={e['example_id']:p['question_type'] for p in pairs for e in p['examples']}
        for condition in ('correct','complementary','no_image'):
            rows=controlled_rows(pairs,condition)
            values=reference_losses(model,rows,tokens,store,condition,True)
            per_condition[condition]={'mean_answer_nll':float(np.mean(values)),
              'by_type':{typ:float(np.mean([v for r,v in zip(rows,values) if types[r['example_id']]==typ])) for typ in sorted(set(types.values()))},
              'examples':[{'example_id':r['example_id'],'image_id_used':None if condition=='no_image' else r['image_id'],'nll':v} for r,v in zip(rows,values)]}
        result['suites'][suite]={'conditions':per_condition,
          'image_swap_penalty':per_condition['complementary']['mean_answer_nll']-per_condition['correct']['mean_answer_nll']}
    result['seconds']=time.perf_counter()-started
    json_write(path,result);model.train(training)
    print(f'Development {update}: caption={result["caption_reference_loss"]:.4f}; '+
          ' '.join(f'{s} ΔNLL={result["suites"][s]["image_swap_penalty"]:.4f}' for s in result['suites']),flush=True)
    return result

def pair_arrays(records,pair_ids):
    grouped=defaultdict(list)
    for row in records:grouped[row['pair_id']].append(row)
    assert set(grouped)==set(pair_ids)
    out={key:[] for key in ('accuracy','reference_nll','pair_accuracy','identical_answer_rate','empty_answer_rate','partner_answer_rate','generation_seconds_per_answer')}
    for pair in pair_ids:
        rows=grouped[pair];assert len(rows)==2 and len({r['example_id'] for r in rows})==2
        out['accuracy'].append(np.mean([r['score'] for r in rows]))
        out['reference_nll'].append(np.mean([r['reference_nll'] for r in rows]))
        out['pair_accuracy'].append(float(all(r['canonical_exact_match'] for r in rows)))
        out['identical_answer_rate'].append(float(rows[0]['normalized_prediction']==rows[1]['normalized_prediction']))
        out['empty_answer_rate'].append(np.mean([r['normalized_prediction']=='' for r in rows]))
        out['partner_answer_rate'].append(np.mean([r['partner_exact_match'] for r in rows]))
        out['generation_seconds_per_answer'].append(np.mean([r['generation_seconds_amortized'] for r in rows]))
    return {key:np.asarray(values,dtype=float) for key,values in out.items()}

def interval(values,indices):
    boots=values[indices].mean(axis=1)
    return {'mean':float(values.mean()),'ci95':[float(x) for x in np.percentile(boots,[2.5,97.5])]}

def summarize_final(all_records,pairs,suite,resamples=4000,seed=6304):
    result={'bootstrap_resamples':resamples,'unit':'complete question pair','seed':seed,'groups':{}}
    groups={'overall':pairs,**{typ:[p for p in pairs if p['question_type']==typ] for typ in sorted({p['question_type'] for p in pairs})}}
    for name,subset in groups.items():
        ids=sorted(p['pair_id'] for p in subset);allowed=set(ids)
        rng=np.random.default_rng(seed+int.from_bytes(hashlib.sha256((suite+':'+name).encode()).digest()[:4],'little'))
        indices=rng.integers(len(ids),size=(resamples,len(ids)))
        arrays={condition:pair_arrays([r for r in rows if r['pair_id'] in allowed],ids) for condition,rows in all_records.items()}
        table={condition:{metric:interval(values,indices) for metric,values in metrics.items()} for condition,metrics in arrays.items()}
        advantage={'accuracy_correct_minus_complementary':interval(arrays['correct']['accuracy']-arrays['complementary']['accuracy'],indices),
                   'nll_complementary_minus_correct':interval(arrays['complementary']['reference_nll']-arrays['correct']['reference_nll'],indices)}
        result['groups'][name]={'pairs':len(ids),'questions':len(ids)*2,'conditions':table,'correct_image_advantages':advantage}
    return result

def final_evaluation(task,model,tokenizer,tokens,store,identity):
    directory=task/'results/final';directory.mkdir(parents=True,exist_ok=True)
    session_start=time.perf_counter()
    model.eval();controls=read(task/'splits/image_controls.json');device=next(model.parameters()).device
    eos=tokenizer.eos_token_id
    configured=model.backbone.generation_config.eos_token_id
    eos_ids=set(([configured] if isinstance(configured,int) else (configured or []))+[eos,tokenizer.convert_tokens_to_ids('<|im_end|>')]);eos_ids.discard(None)
    for suite in ('grounding','natural'):
        pairs=read(task/f'splits/{suite}_test.json');original=flatten(pairs)
        mismatch={r['example_id']:r for r in controls[suite+'_test']}
        pair_for={e['example_id']:p for p in pairs for e in p['examples']}
        partners={e['example_id']:p['examples'][1-side]['canonical_answer'] for p in pairs for side,e in enumerate(p['examples'])}
        all_records={}
        for condition in BENCH_CONDITIONS:
            path=directory/f'{suite}_{condition}.json'
            if path.exists():
                saved=read(path);assert saved['identity']==identity;all_records[condition]=saved['examples'];continue
            rows=controlled_rows(pairs,condition,mismatch)
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
            condition_start=time.perf_counter()
            nll=reference_losses(model,rows,tokens,store,condition,True)
            results=[]
            for start in range(0,len(rows),8):
                chunk=rows[start:start+8]
                if device.type=='cuda':torch.cuda.synchronize()
                begin=time.perf_counter()
                features=None if condition=='no_image' else store.get(chunk,device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    generated=greedy(model,[tokens[r['example_id']] for r in chunk],features,eos_ids)
                texts=[tokenizer.decode(g['output_token_ids'],skip_special_tokens=True).strip() for g in generated]
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                for offset,(row,generation,text) in enumerate(zip(chunk,generated,texts)):
                    index=start+offset;reference=original[index];assert reference['example_id']==row['example_id']
                    predicted=normalize_answer(text);canonical=reference['canonical_answer'];pair=pair_for[row['example_id']]
                    score=float(bool(predicted) and predicted==canonical) if suite=='grounding' else vqa_soft_accuracy(text,reference['answers'])
                    results.append({'example_id':row['example_id'],'pair_id':pair['pair_id'],'question_type':pair['question_type'],
                      'condition':condition,'question':reference['question'],'original_image_id':reference['image_id'],
                      'image_id_used':None if condition=='no_image' else row['image_id'],'references':reference['answers'],
                      'reference_string':reference['target'],'canonical_answer':canonical,'partner_canonical_answer':partners[row['example_id']],
                      'response':text,'normalized_prediction':predicted,'raw_text':tokenizer.decode(generation['output_token_ids'],skip_special_tokens=False),
                      **generation,'score':score,'canonical_exact_match':bool(predicted) and predicted==canonical,
                      'partner_exact_match':bool(predicted) and predicted==partners[row['example_id']],
                      'reference_nll':nll[index],'generation_seconds_amortized':elapsed/len(chunk),
                      'prompt_before_ids':tokens[row['example_id']]['before_ids'],'prompt_after_ids':tokens[row['example_id']]['after_ids']})
                del features
            assert len(results)==(128 if suite=='grounding' else 192)
            json_write(path,{'identity':identity,'eos_ids':sorted(eos_ids),'examples':results,
              'elapsed_seconds':time.perf_counter()-condition_start,
              'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0})
            all_records[condition]=results
            print(f'Final {suite}/{condition}: {len(results)} questions saved',flush=True)
        summary=summarize_final(all_records,pairs,suite)
        json_write(directory/f'{suite}_summary.json',summary)
    saved=[read(directory/f'{suite}_{condition}.json') for suite in ('grounding','natural') for condition in BENCH_CONDITIONS]
    json_write(directory/'finished.json',{'passed':True,'identity':identity,'suites':2,'conditions_each':4,'generation_answers':1280,
      'seconds_this_session':time.perf_counter()-session_start,
      'peak_gpu_allocated_bytes_all_conditions':max(s['peak_gpu_allocated_bytes'] for s in saved),
      'generation_seconds_all_conditions':sum(r['generation_seconds_amortized'] for s in saved for r in s['examples']),
      'output_tokens_all_conditions':sum(r['output_tokens'] for s in saved for r in s['examples'])})
