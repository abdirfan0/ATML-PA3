"""CPU-only reconstruction of the recorded data preparation; never launches training.

This script reconstructs the notebook preparation rules. It is not the original
executed notebook. Pinned revisions, exact saved source IDs, supervised-token
count and maximum sequence length guard reconstruction before a manifest is saved.
Existing prepared datasets are validated by runner_v1/task1_train.py --prepare.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', str(text)).casefold().split())


def main(task):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm
    config=json.loads((task/'configs/data_config.json').read_text())
    destination=task/'data/metamath_40k'
    if destination.exists():
        raise FileExistsError(f'Prepared data already exists: {destination}. Use the existing CPU --prepare validation.')
    tokenizer=AutoTokenizer.from_pretrained(config['model_id'],revision=config['model_revision'])
    eot=tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert tokenizer.convert_ids_to_tokens(eot)=='<|im_end|>'
    gsm=load_dataset(config['gsm8k_dataset'],'main',revision=config['gsm8k_revision'],split='test')
    excluded={normalize(q) for q in gsm['question']}
    source=load_dataset(config['train_dataset'],revision=config['train_revision'],split='train',streaming=True)
    expected=json.loads((task/'splits/metamath_source_indices.json').read_text())
    rows=[]; counts={'scanned':0,'retained':0,'overlength':0,'empty':0,'overlap':0}
    progress=tqdm(total=40000,desc='Retaining eligible examples')
    for index,record in enumerate(source):
        counts['scanned']+=1
        question=str(record.get('query') or '').strip()
        response=str(record.get('response') or '').strip()
        if not question or not response:
            counts['empty']+=1;continue
        if normalize(question) in excluded or normalize(record.get('original_question') or '') in excluded:
            counts['overlap']+=1;continue
        prompt_ids=tokenizer.apply_chat_template([{'role':'user','content':question}],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        response_ids=tokenizer(response,add_special_tokens=False)['input_ids']+[eot]
        input_ids=prompt_ids+response_ids
        if len(input_ids)>2048:
            counts['overlength']+=1;continue
        assert index==expected[len(rows)], 'Selection differs from the executed preparation; do not train.'
        rows.append({'source_index':index,'input_ids':input_ids,'labels':[-100]*len(prompt_ids)+response_ids,
                     'length':len(input_ids),'response_tokens':len(response_ids)})
        counts['retained']+=1;progress.update(1)
        if len(rows)==40000:break
    progress.close()
    selected=[r['source_index'] for r in rows]
    total=sum(r['response_tokens'] for r in rows)
    assert selected==expected and total==7100056
    assert max(r['length'] for r in rows)==1941
    data=task/'data';data.mkdir(exist_ok=True)
    staging=data/'metamath_40k.rebuilding'
    if staging.exists():raise FileExistsError(staging)
    Dataset.from_list(rows).save_to_disk(str(staging))
    import numpy as np
    manifest={'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
      'counts':counts,'first_source_index':selected[0],'last_source_index':selected[-1],
      'length_percentiles':{str(k):float(v) for k,v in zip([0,25,50,75,95,100],np.percentile([r['length'] for r in rows],[0,25,50,75,95,100]))},
      'total_supervised_tokens':total,'selected_ids_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
      'reconstruction_note':'Created by a reconstructed CPU preparation script; checked against saved source IDs and aggregate token count.'}
    staging.rename(destination)
    (data/'metamath_40k_manifest.json').write_text(json.dumps(manifest,indent=2))
    (task/'results').mkdir(exist_ok=True)
    (task/'results/data_validation.json').write_text(json.dumps({
        'passed':True,'examples':len(rows),'supervised_tokens':total,
        'selected_ids_sha256':manifest['selected_ids_sha256'],
        'validation_scope':'Reconstructed rows; exact saved IDs, response-token total and maximum length checked.'
    },indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir',type=Path,required=True)
    main(parser.parse_args().task_dir)
