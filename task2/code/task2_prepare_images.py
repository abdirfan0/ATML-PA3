"""CPU download/verification of selected COCO images into resumable TAR shards.

Raw JPEGs are preserved without resizing/re-encoding. Only the selected image
catalog is used; unavailable images cause failure, never replacement/resampling.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import threading
import time

from PIL import Image,ImageDraw,ImageFont,ImageOps

VERSION='pa3-task2-images-v1'
BASE_URL='https://s3.amazonaws.com/images.cocodataset.org'
SHARD_SIZE=256
_thread_state=threading.local()

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def validate_row(row):
    split=row['image_split'];image=int(row['image_id'])
    assert split in ('train2014','val2014') and image>=0
    assert row['file_name']==f'COCO_{split}_{image:012d}.jpg'
    assert int(row['width'])>0 and int(row['height'])>0
    return f"{split}/{row['file_name']}"

def check_jpeg(path,row):
    with Image.open(path) as image:
        assert image.format=='JPEG',f'Expected JPEG: {path}'
        assert image.size==(int(row['width']),int(row['height'])),f'Image dimensions differ: {path}'
        image.verify()
    with Image.open(path) as image:
        image.convert('RGB').load()  # Fully decode, including entropy-coded image data.
    return {'image_id':int(row['image_id']),'image_split':row['image_split'],
            'file_name':row['file_name'],'member':validate_row(row),'width':int(row['width']),
            'height':int(row['height']),'bytes':path.stat().st_size,'sha256':sha256(path)}

def session():
    if not hasattr(_thread_state,'session'):
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        client=requests.Session()
        client.mount('https://',HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,
          status_forcelist=[429,500,502,503,504],allowed_methods=['GET'])))
        _thread_state.session=client
    return _thread_state.session

def fetch_image(row,local_images):
    member=validate_row(row)
    destination=local_images/member
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():return check_jpeg(destination,row)
    temporary=destination.with_name(destination.name+'.part')
    for attempt in range(3):
        try:
            with session().get(f'{BASE_URL}/{member}',stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                expected=int(response.headers.get('Content-Length',0))
                with temporary.open('wb') as output:
                    for chunk in response.iter_content(chunk_size=256*1024):
                        if chunk:output.write(chunk)
            if expected:assert temporary.stat().st_size==expected,f'Incomplete image: {member}'
            result=check_jpeg(temporary,row)
            temporary.replace(destination)
            return result
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)
    raise RuntimeError('Unreachable')

def build_tar(path,rows,local_images):
    temporary=path.with_name(path.name+'.tmp')
    with tarfile.open(temporary,'w',format=tarfile.USTAR_FORMAT) as archive:
        for row in rows:
            member=validate_row(row);source=local_images/member
            info=tarfile.TarInfo(member)
            info.size=source.stat().st_size;info.mtime=0;info.mode=0o644;info.uid=info.gid=0
            info.uname=info.gname=''
            with source.open('rb') as incoming:archive.addfile(info,incoming)
    temporary.replace(path)
    return sha256(path)

def validate_complete_shard(tar_path,marker,rows,identity):
    assert marker['identity']==identity,'Image input identity changed.'
    assert marker['sha256']==sha256(tar_path),f'Archive hash differs: {tar_path}'
    assert marker['bytes']==tar_path.stat().st_size
    assert [r['member'] for r in marker['images']]==[validate_row(r) for r in rows]
    for original,saved in zip(rows,marker['images']):
        for key in ('image_id','image_split','file_name','width','height'):assert original[key]==saved[key]
    return marker

class ImageArchiveReader:
    def __init__(self,image_dir,manifest,local_tar_dir=None):
        self.image_dir=Path(image_dir);self.local_tar_dir=local_tar_dir;self.handles={};self.lookup={}
        for shard in manifest['shards']:
            for row in shard['images']:
                key=(row['image_split'],row['image_id'])
                assert key not in self.lookup
                self.lookup[key]=(shard['file_name'],row)
    def get(self,example):
        name,record=self.lookup[(example['image_split'],int(example['image_id']))]
        if name not in self.handles:
            path=self.image_dir/name
            if self.local_tar_dir is not None:
                local=Path(self.local_tar_dir)/name
                if local.exists():path=local
            self.handles[name]=tarfile.open(path,'r:')
        with self.handles[name].extractfile(record['member']) as stream:
            data=stream.read()
        assert hashlib.sha256(data).hexdigest()==record['sha256'],'Image member hash differs.'
        with Image.open(io.BytesIO(data)) as image:return image.convert('RGB')
    def close(self):
        for archive in self.handles.values():archive.close()
        self.handles.clear()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def review_sheets(task,manifest,local_tar_dir):
    destination=task/'figures/label_review';destination.mkdir(parents=True,exist_ok=True)
    try:
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
        small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    except OSError:
        font=small=ImageFont.load_default()
    pages=[]
    with ImageArchiveReader(task/'data/images',manifest,local_tar_dir) as reader:
        for split in ('dev','test'):
            pairs=sorted(read(task/f'splits/grounding_{split}.json'),key=lambda p:p['pair_id'])
            for offset in range(0,len(pairs),4):
                chunk=pairs[offset:offset+4]
                canvas=Image.new('RGB',(1100,1470),'white');draw=ImageDraw.Draw(canvas)
                draw.text((20,12),f'Grounding {split}: label review, pairs {offset+1}–{offset+len(chunk)}',fill='black',font=font)
                for index,pair in enumerate(chunk):
                    y=48+index*352
                    draw.text((20,y),pair['question'],fill='black',font=font)
                    for side,example in enumerate(pair['examples']):
                        x=20+side*540
                        image=reader.get(example)
                        image=ImageOps.contain(image,(520,260),method=Image.Resampling.LANCZOS)
                        canvas.paste(image,(x+(520-image.width)//2,y+28+(260-image.height)//2))
                        draw.text((x,y+292),f"image {example['image_id']} | annotation target: {example['target']}",fill='black',font=small)
                    draw.text((20,y+320),f"{pair['question_type']} | {pair['category']} | {pair['pair_id']}",fill='#444444',font=small)
                path=destination/f'grounding_{split}_{offset//4+1:02d}.jpg'
                temporary=path.with_name(path.name+'.tmp')
                canvas.save(temporary,format='JPEG',quality=90);temporary.replace(path)
                pages.append({'path':str(path.relative_to(task)),'sha256':sha256(path),'pair_ids':[p['pair_id'] for p in chunk]})
    note={'selection_rule':'All 24 grounding dev pairs and all 64 grounding test pairs, ordered by saved pair ID; four pairs per sheet. No model predictions used.',
          'review_status':'pending human label review','pages':pages,
          'instructions':'Inspect annotation negatives for visible omitted objects and counting labels for ambiguity. Do not alter examples based on model predictions. Save any review findings before training.'}
    atomic_json(task/'results/label_review_index.json',note)
    return len(pages)

def prepare(task,workers,work_dir,make_review=True):
    from tqdm.auto import tqdm
    assert 1<=workers<=16
    selection=read(task/'results/data_selection.json')
    assert selection['passed'] and selection['image_disjoint'] and selection['total_selected_images']==6528
    catalog_path=task/'splits/selected_images.json'
    selection_manifest_path=task/'splits/selection_manifest.json'
    selected_manifest=read(selection_manifest_path)
    assert selected_manifest['files']['splits/selected_images.json']==sha256(catalog_path)
    catalog=read(catalog_path)
    assert len(catalog)==6528 and len({validate_row(r) for r in catalog})==6528
    identity={'version':VERSION,'catalog_sha256':sha256(catalog_path),
              'selection_manifest_sha256':sha256(selection_manifest_path),'source':BASE_URL,'shard_size':SHARD_SIZE}
    image_dir=task/'data/images';image_dir.mkdir(parents=True,exist_ok=True)
    work_dir=work_dir/identity['catalog_sha256'][:16]
    local_images=work_dir/'jpeg';local_tars=work_dir/'tar';local_tars.mkdir(parents=True,exist_ok=True)
    config_path=task/'configs/image_preparation_config.json'
    if config_path.exists():assert read(config_path)==identity,'Image selection/source changed.'
    else:atomic_json(config_path,identity)
    started=time.perf_counter();shards=[];downloaded=0;reused=0
    for start in range(0,len(catalog),SHARD_SIZE):
        index=start//SHARD_SIZE;rows=catalog[start:start+SHARD_SIZE]
        name=f'images_{index:04d}.tar';tar_path=image_dir/name;marker_path=tar_path.with_suffix('.json')
        if marker_path.exists():
            shard=validate_complete_shard(tar_path,read(marker_path),rows,identity)
            shards.append(shard);reused+=len(rows)
            print(f'Reusing shard {index+1}: {len(rows)} verified images',flush=True)
            continue
        ordered={}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(fetch_image,row,local_images):row for row in rows}
            for future in tqdm(as_completed(futures),total=len(futures),desc=f'Image shard {index+1}'):
                record=future.result();ordered[record['member']]=record
        records=[ordered[validate_row(row)] for row in rows]
        local_tar=local_tars/name;digest=build_tar(local_tar,rows,local_images)
        temporary=tar_path.with_name(tar_path.name+'.tmp')
        shutil.copyfile(local_tar,temporary)
        assert sha256(temporary)==digest,'Persistent archive copy failed hash verification.'
        temporary.replace(tar_path)
        shard={'file_name':name,'identity':identity,'bytes':tar_path.stat().st_size,'sha256':digest,'images':records}
        atomic_json(marker_path,shard);shards.append(shard);downloaded+=len(rows)
        print(f'Saved shard {index+1}: {len(rows)} images',flush=True)
    manifest={'identity':identity,'images':len(catalog),'shards':shards}
    manifest_path=image_dir/'manifest.json';atomic_json(manifest_path,manifest)
    pages=review_sheets(task,manifest,local_tars) if make_review else 0
    report={'passed':True,'images':len(catalog),'shards':len(shards),
      'image_bytes':sum(r['bytes'] for s in shards for r in s['images']),
      'archive_bytes':sum(s['bytes'] for s in shards),'verified_existing_images':reused,
      'images_prepared_this_session':downloaded,'elapsed_seconds_this_session':time.perf_counter()-started,
      'manifest_sha256':sha256(manifest_path),'label_review_pages':pages,
      'label_review_status':'pending human review' if make_review else 'not generated',
      'gpu_used':False,'image_policy':'Original JPEG bytes, verified size, dimensions, full RGB decode and SHA256. No resizing/re-encoding of stored input images; no resampling/replacement of unavailable images.'}
    atomic_json(task/'results/image_preparation.json',report)
    print('\nImage preparation PASS.',flush=True);print(json.dumps(report,indent=2),flush=True)
    print('Stay on CPU. Review labels and prepare the frozen vision model before feature caching.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir',required=True,type=Path)
    parser.add_argument('--workers',default=8,type=int)
    parser.add_argument('--work-dir',default=Path(tempfile.gettempdir())/'pa3_task2_image_work',type=Path)
    args=parser.parse_args();prepare(args.task_dir,args.workers,args.work_dir)
