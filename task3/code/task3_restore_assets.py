"""Restore pinned public MathVista metadata and supplied README on CPU."""
import argparse,hashlib,json,shutil,urllib.request
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def restore(task):
    task=Path(task);cfg=json.loads((task/'configs/scorer_frozen.json').read_text());revision=cfg['mathvista_revision']
    root=task/'assets/mathvista'/revision;root.mkdir(parents=True,exist_ok=True)
    items=[('testmini.json','data/testmini.json',cfg['mathvista_metadata_sha256']),('calculate_score.py','evaluation/calculate_score.py',cfg['mathvista_upstream_scorer_sha256']),('MATHVISTA_LICENSE','LICENSE','da89f9867822be4b8adb1e601d9e9226c195016c6508015eb7593e68ead0c98a')]
    for name,source,digest in items:
        p=root/name
        if not p.exists():
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/lupantech/MathVista/{revision}/{source}',timeout=60) as r: data=r.read()
            assert hashlib.sha256(data).hexdigest()==digest,name
            p.write_bytes(data)
        assert sha(p)==digest,name
    destination=task/'assets/supplied_README.MD'
    if not destination.exists():
        source=task/'data/raw/ATML-PA3-Data/README.MD';assert source.is_file(),source
        assert sha(source)==cfg['supplied_readme_sha256'];shutil.copy2(source,destination)
    assert sha(destination)==cfg['supplied_readme_sha256']
    print('Pinned MathVista assets and supplied README verified. No models or GPU used.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task-dir',type=Path,required=True);restore(p.parse_args().task_dir)
