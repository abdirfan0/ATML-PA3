"""Fixed live-generation recipe and gold-free prompt/seed construction."""
import hashlib

VERSION='pa3-task3-live-v1'
MODEL_ID='Qwen/Qwen2.5-3B-Instruct'
MODEL_REVISION='aa8e72537993ba99e69dfaafa59ed015b17504d1'
DATASET_ID='DigitalLearningGmbH/MATH-lighteval'
DATASET_REVISION='0530c78699ea5e8eb5530600900e1f328b48acad'
SYSTEM='You are a helpful assistant.'
INSTRUCTION=r'Think step by step inside <think> </think> tags, then give your final answer inside \boxed{}.'
REVISION_INSTRUCTION=r'Review the solution above. Identify and correct any errors; if it is correct, retain its answer. Give your final answer inside \boxed{}.'
CALLS={'single_long':3072,'parallel_0':1024,'parallel_1':1024,'parallel_2':1024,'revision':2048}
STRATEGIES={'single_long':['single_long'],'parallel':['parallel_0','parallel_1','parallel_2'],'sequential':['parallel_0','revision']}

def seed_for(sample_id,call):
    assert call in CALLS
    return int.from_bytes(hashlib.sha256(f'6304:{VERSION}:{sample_id}:{call}'.encode()).digest()[:8],'big')%(2**31)

def messages(problem,initial_response=None):
    result=[{'role':'system','content':SYSTEM},{'role':'user','content':problem+'\n\n'+INSTRUCTION}]
    if initial_response is not None:result.extend([{'role':'assistant','content':initial_response},{'role':'user','content':REVISION_INSTRUCTION}])
    return result

def prompt_record(tokenizer,sample_id,problem,initial_response=None):
    chat=messages(problem,initial_response)
    prompt=tokenizer.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)
    ids=list(tokenizer(prompt,add_special_tokens=False)['input_ids'])
    assert ids and all(isinstance(i,int) for i in ids)
    return {'sample_id':sample_id,'problem':problem,'messages':chat,'prompt':prompt,'input_ids':ids,'input_tokens':len(ids)}

def recipe():
    return {'version':VERSION,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,'dataset_id':DATASET_ID,'dataset_revision':DATASET_REVISION,
      'dtype':'float16','quantization':False,'batch_size':1,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.,'do_sample':True,
      'calls':CALLS,'strategies':STRATEGIES,'system':SYSTEM,'instruction':INSTRUCTION,'revision_instruction':REVISION_INSTRUCTION,
      'seed_rule':'SHA256 of 6304:version:sample_id:call, first eight bytes big-endian modulo 2**31; independently reseed each call',
      'shared_initial':'parallel_0 reused verbatim as the initial assistant response for revision; charge its full input/output/runtime to both parallel and sequential strategies',
      'extraction':'Frozen scorer last literal boxed answer, balanced braces; absent/empty/unclosed last box -> missing; no fallback',
      'time_scope':'End-to-end per-call inference with CUDA synchronization; exclude model loading and separate smoke/warmup; count revision prompt and all critique output',
      'cost_scope':'Record complete generated IDs including termination and retained IDs excluding final EOS separately; actual output allowance/cost includes termination tokens',
      'physical_calls_per_question':5,'physical_max_output_tokens_per_question':sum(CALLS.values())}
