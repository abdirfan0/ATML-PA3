"""Safe, bounded-memory reader for Task 3 CPU-prepared question shards."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

class PreparedPool:
    def __init__(self,task,label,local_dir=None):
        self.root=Path(task)/'data/prepared'/label
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.lookup={x['sample_id']:x for x in self.manifest['questions']}
        self.shards={x['file_name']:x for x in self.manifest['shards']}
        self.local=Path(local_dir) if local_dir is not None else None
        self.current=None;self.archive=None
    def get(self,sample_id):
        row=self.lookup[sample_id];name=row['shard']
        if name!=self.current:
            self.close();path=self.root/name
            assert sha256(path)==self.shards[name]['sha256'],path
            if self.local is not None:
                self.local.mkdir(parents=True,exist_ok=True);destination=self.local/name
                if not destination.exists() or sha256(destination)!=self.shards[name]['sha256']:shutil.copyfile(path,destination)
                assert sha256(destination)==self.shards[name]['sha256'];path=destination
            self.archive=np.load(path,allow_pickle=False);self.current=name
        key=row['key'];metadata=json.loads(self.archive[key+'_metadata'].tobytes().decode())
        offsets=self.archive[key+'_offsets'];assert len(offsets)==33
        arrays={field:self.archive[key+'_'+field] for field in ('token_ids','logp','kl')}
        if key+'_vision' in self.archive:arrays['vision']=self.archive[key+'_vision']
        return {'metadata':metadata,'offsets':offsets,**arrays}
    def close(self):
        if self.archive is not None:self.archive.close()
        self.archive=None;self.current=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
