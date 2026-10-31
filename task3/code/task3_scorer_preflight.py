"""CPU-only scorer/environment freeze and supplied MathVista identity checks."""
import argparse,collections,hashlib,importlib.metadata as metadata,json,shutil
from pathlib import Path
import pyarrow.parquet as pq
from task3_answers import VERSION,MATHVISTA_REVISION,NO_ANSWER,mathvista_answer,mathvista_reference

EXPECTED={'math-verify':'0.9.0','latex2sympy2_extended':'1.11.0','antlr4-python3-runtime':'4.13.2','sympy':'1.14.0'}
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def prepare(task):
    for name,version in EXPECTED.items():assert metadata.version(name)==version,(name,metadata.version(name))
    package=read(task/'scorer_package_manifest.json')
    for relative,digest in package['files'].items():assert sha(task/relative)==digest,relative
    assets=task/'assets/mathvista'/MATHVISTA_REVISION
    questions=read(assets/'testmini.json');assert len(questions)==1000
    inventory=read(task/'configs/supplied_data_inventory.json');splits=read(task/'splits/question_splits.json')
    path=Path(inventory['mathvista']['files']['generations.parquet']['path'])
    rows=pq.ParquetFile(path).read(columns=['sample_id','gold_answer']).to_pylist()
    assert len(rows)==32000
    assert {r['sample_id'] for r in rows}==set(questions)
    for row in rows:
        meta=questions[row['sample_id']]
        normalized_gold=mathvista_reference(meta['answer'],meta)
        assert normalized_gold!=NO_ANSWER
        actual_gold=mathvista_reference(row['gold_answer'],meta)
        assert actual_gold==normalized_gold, {'sample_id':row['sample_id'],'supplied_gold':row['gold_answer'],
          'metadata_gold':meta['answer'],'supplied_option':actual_gold,'metadata_option':normalized_gold,'choices':meta['choices']}
    for label,split in splits['datasets'].items():
        assert len(split['development'])==200 and len(split['evaluation'])==800
        assert not set(split['development'])&set(split['evaluation'])
        assert len(set(split['development']+split['evaluation']))==1000
    assert splits['datasets']['math']['live_math']==splits['datasets']['math']['evaluation'][:50]
    versions={name:metadata.version(name) for name in [*EXPECTED,'numpy','pandas','pyarrow','scipy']}
    frozen={'version':VERSION,'packages':versions,'mathvista_revision':MATHVISTA_REVISION,
      'mathvista_metadata_sha256':sha(assets/'testmini.json'),'mathvista_upstream_scorer_sha256':sha(assets/'calculate_score.py'),
      'supplied_readme_sha256':sha(task/'assets/supplied_README.MD'),
      'input_inventory_sha256':sha(task/'configs/supplied_data_inventory.json'),
      'question_splits_sha256':sha(task/'splits/question_splits.json'),
      'source_sha256':{name:sha(task/'code'/name) for name in ['task3_answers.py','task3_scorer_preflight.py','test_task3_answers_cpu.py']},
      'math_policy':{'parser':'Math-Verify LatexExtractionConfig, extracted answer wrapped in math delimiters; no parser text fallback',
        'equivalence':'Exact nonempty stripped string equality, otherwise symmetric strict Math-Verify; same predicate for grading',
        'parse_timeout_seconds':5,'verify_timeout_seconds':5,'float_rounding':6,'numeric_precision':15,
        'groups':'Only revealed predictions; sorted unique strings, deterministic first matching all-member group. Never canonicalize with gold or unseen answers.'},
      'mathvista_policy':'Released prediction types/precision/closest-choice rules; option letters, missing/invalid values NO_ANSWER; gold exact choice text maps directly to its option before letter parsing; no answer extraction from full stored responses',
      'live_math_model':{'id':'Qwen/Qwen2.5-3B-Instruct','revision':'aa8e72537993ba99e69dfaafa59ed015b17504d1'},
      'selection_seed':6304,'supplied_generation_seed':42}
    lock=task/'configs/scorer_frozen.json'
    if lock.exists():assert read(lock)==frozen,'Scorer identity changed; do not overwrite an established evaluation scorer.'
    else:write(lock,frozen)
    result={'passed':True,'scorer_version':VERSION,'mathvista_metadata_questions':1000,
      'mathvista_supplied_gold_rows_checked':32000,'question_splits_verified':True,
      'mathvista_question_types':dict(collections.Counter(q['question_type'] for q in questions.values())),
      'mathvista_answer_types':dict(collections.Counter(q['answer_type'] for q in questions.values())),
      'packages':versions,'gpu_used':False,'held_out_metrics_computed':False,'scorer_sha256':sha(lock)}
    write(task/'results/scorer_preflight.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--task-dir',type=Path,required=True)
    prepare(parser.parse_args().task_dir)
