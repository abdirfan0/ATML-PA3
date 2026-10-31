"""Task 2 CPU preparation and sequential GPU cache/smoke/train/final queue."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from task2_runtime import prepare,read,json_write

def run(command,path):
    print('\nLaunching:',' '.join(map(str,command)),flush=True)
    with path.open('a') as stream:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        try:
            for index,line in enumerate(process.stdout):
                print(line,end='',flush=True);stream.write(line)
                if index%25==0 or any(word in line for word in ('Saved','PASS','finished','Development')):stream.flush()
            code=process.wait()
        except BaseException:
            process.terminate();process.wait();raise
        if code:raise subprocess.CalledProcessError(code,command)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--run-all',action='store_true');mode.add_argument('--plan',action='store_true')
    a=p.parse_args();task=a.task_dir
    if a.prepare:prepare(task);return
    if a.plan:
        print('Feature cache:',(task/'features/patch196/finished.json').exists())
        print('GPU smoke:',(task/'results/gpu_smoke_task2.json').exists())
        print('Training:',(task/'checkpoints/main/finished.json').exists())
        print('Final evaluation:',(task/'results/final/finished.json').exists());return
    assert read(task/'results/runner_cpu_preflight.json')['passed']
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    directory=task/'results/queue_logs';directory.mkdir(parents=True,exist_ok=True);code=Path(__file__).parent
    for name,script,flags in [('features','task2_features.py',[]),('smoke','task2_train.py',['--smoke']),('training','task2_train.py',[]),('final','task2_train.py',['--final'])]:
        run([sys.executable,str(code/script),'--task-dir',str(task),*flags],directory/(name+'.log'))
    json_write(task/'results/task2_queue_finished.json',{'passed':True,'features':6528,'training_updates':4500,'training_presentations':72000,'final_generated_answers':1280})
    print('Task 2 required GPU work finished. CPU analysis and repository organization remain.',flush=True)

if __name__=='__main__':main()
