"""Download COCO 2014/VQAv2 metadata on CPU. Does not download images or load models.

VQA sources: https://visualqa.org/download.html (current official linked bucket).
COCO source: official images.cocodataset.org S3 bucket via HTTPS path-style URL.
Only caption/instance annotations, VQA questions/answers and pair lists are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

VERSION = 'pa3-task2-metadata-v1'
VQA = 'https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/'
SOURCES = {
 'coco2014': {
  'url':'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',
  'members':['captions_train2014.json','captions_val2014.json','instances_train2014.json','instances_val2014.json'],
  'expected_bytes':252872794,
 },
 'vqa_train_answers': {'url':VQA+'v2_Annotations_Train_mscoco.zip','members':['v2_mscoco_train2014_annotations.json']},
 'vqa_val_answers': {'url':VQA+'v2_Annotations_Val_mscoco.zip','members':['v2_mscoco_val2014_annotations.json']},
 'vqa_train_questions': {'url':VQA+'v2_Questions_Train_mscoco.zip','members':['v2_OpenEnded_mscoco_train2014_questions.json']},
 'vqa_val_questions': {'url':VQA+'v2_Questions_Val_mscoco.zip','members':['v2_OpenEnded_mscoco_val2014_questions.json']},
 'vqa_train_pairs': {'url':VQA+'v2_Complementary_Pairs_Train_mscoco.zip','members':['v2_mscoco_train2014_complementary_pairs.json']},
 'vqa_val_pairs': {'url':VQA+'v2_Complementary_Pairs_Val_mscoco.zip','members':['v2_mscoco_val2014_complementary_pairs.json']},
}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def download(session,url,destination,expected_bytes=None):
    from tqdm.auto import tqdm
    if destination.exists():
        if expected_bytes is not None:
            assert destination.stat().st_size==expected_bytes,f'Existing archive has the wrong size: {destination}'
        print('Reusing:',destination.name,flush=True)
        return
    temporary=destination.with_suffix(destination.suffix+'.part')
    with session.get(url,stream=True,timeout=(20,120)) as response:
        response.raise_for_status()
        total=int(response.headers.get('Content-Length',0))
        with temporary.open('wb') as output,tqdm(total=total or None,unit='B',unit_scale=True,desc=destination.name) as bar:
            for chunk in response.iter_content(chunk_size=1024**2):
                if chunk:output.write(chunk);bar.update(len(chunk))
    if total:assert temporary.stat().st_size==total,(destination,'incomplete download')
    if expected_bytes is not None:assert temporary.stat().st_size==expected_bytes,(destination,'wrong archive size')
    with zipfile.ZipFile(temporary) as archive:
        bad=archive.testzip()
        assert bad is None,f'Archive CRC failed: {bad}'
    temporary.replace(destination)


def extract_selected(archive_path,wanted,destination):
    records=[]
    with zipfile.ZipFile(archive_path) as archive:
        by_basename={}
        for member in archive.infolist():
            if not member.is_dir():by_basename.setdefault(Path(member.filename).name,[]).append(member)
        for name in wanted:
            assert Path(name).name==name
            found=by_basename.get(name,[])
            assert len(found)==1, f'{name}: expected one archive member; found {len(found)}. Members: {archive.namelist()}'
            member=found[0]
            output=destination/name
            if not output.exists():
                temporary=output.with_suffix('.json.part')
                with archive.open(member) as source,temporary.open('wb') as target:
                    shutil.copyfileobj(source,target,length=4*1024**2)
                assert temporary.stat().st_size==member.file_size
                temporary.replace(output)
            assert output.stat().st_size==member.file_size,f'Extracted file size changed: {output}'
            records.append({'filename':name,'bytes':output.stat().st_size,'sha256':sha256(output)})
    return records


def main(task_dir):
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    downloads=task_dir/'data/downloads'
    annotations=task_dir/'data/annotations'
    configs=task_dir/'configs'
    for directory in (downloads,annotations,configs):directory.mkdir(parents=True,exist_ok=True)
    manifest_path=configs/'metadata_sources.json'
    previous=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous is not None:
        assert previous['version']==VERSION
    records={}
    with requests.Session() as session:
        adapter=HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=[429,500,502,503,504]))
        session.mount('https://',adapter)
        for key,spec in SOURCES.items():
            archive_path=downloads/Path(spec['url']).name
            download(session,spec['url'],archive_path,spec.get('expected_bytes'))
            record={'url':spec['url'],'archive':archive_path.name,'bytes':archive_path.stat().st_size,
                    'sha256':sha256(archive_path),
                    'json_files':extract_selected(archive_path,spec['members'],annotations)}
            if previous is not None:
                assert record==previous['sources'][key],f'Metadata identity changed: {key}'
            records[key]=record
            print(f'{key}: {len(record["json_files"])} JSON file(s) ready',flush=True)
    manifest={'version':VERSION,'sources':records}
    atomic_json(manifest_path,manifest)
    report={'passed':True,'archives':len(records),'json_files':sum(len(r['json_files']) for r in records.values()),
            'download_bytes':sum(r['bytes'] for r in records.values()),
            'extracted_bytes':sum(f['bytes'] for r in records.values() for f in r['json_files']),
            'manifest_sha256':sha256(manifest_path),'images_downloaded':False,'gpu_used':False}
    results=task_dir/'results';results.mkdir(exist_ok=True)
    atomic_json(results/'metadata_download.json',report)
    print('\nMetadata preparation complete.',flush=True)
    print(json.dumps(report,indent=2),flush=True)
    print('Next: normalize answers, select pairs and captions, and verify split isolation.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir',type=Path,required=True)
    main(parser.parse_args().task_dir)
