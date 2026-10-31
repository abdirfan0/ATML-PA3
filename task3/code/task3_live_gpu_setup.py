"""Pin the recorded live environment while preserving Colab's CUDA torch."""
import argparse,importlib.metadata as metadata,json,subprocess,sys
from pathlib import Path

def main(task):
    assert json.loads((task/'results/live_runner_cpu_preflight.json').read_text())['passed']
    assets=json.loads((task/'configs/live_assets.json').read_text())
    expected=dict(assets['cpu_packages']);expected.update(json.loads((task/'configs/scorer_frozen.json').read_text())['packages'])
    current_torch=metadata.version('torch')
    assert current_torch.split('+')[0]==expected['torch'].split('+')[0],f'Expected torch {expected["torch"]}, found {current_torch}'
    subprocess.run([sys.executable,'-c',
      "import torch; assert torch.cuda.is_available() and torch.version.cuda is not None, 'Select a CUDA GPU runtime'; print('GPU:', torch.cuda.get_device_name(0), '| torch:', torch.__version__)"],check=True)
    pins={name:value for name,value in expected.items() if name!='torch'}
    constraints=Path('/content/pa3_task3_live_gpu_constraints.txt')
    constraints.write_text(f'torch=={current_torch}\n'+'\n'.join(f'{n}=={v}' for n,v in pins.items())+'\n')
    subprocess.run([sys.executable,'-m','pip','install','--constraint',str(constraints),*[f'{n}=={v}' for n,v in pins.items()]],check=True)
    # Torch/transformers were not imported into this setup process before pip.
    from task3_live_run import gpu_environment
    print(json.dumps(gpu_environment(task),indent=2));print('GPU environment ready.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True);main(parser.parse_args().task_dir)
