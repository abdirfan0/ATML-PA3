"""CPU tests for native generation settings, seeds, stopping flags and resume."""
import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from task3_live_spec import CALLS,MODEL_REVISION,prompt_record,seed_for
from task3_live_runtime import generate_record,save_call,load_call,call_plan,strategy_costs

class TokenizerFixture:
    eos_token_id=255;pad_token_id=0
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return '\n'.join(m['role']+':'+m['content'] for m in messages)+'\nassistant:'
    def __call__(self,text,add_special_tokens):return {'input_ids':list(text.encode())}
    def decode(self,ids,skip_special_tokens):
        return ''.join('<EOS>' if i==255 else chr(i) for i in ids)

class ModelFixture(torch.nn.Module):
    def __init__(self,continuation=None):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False)
        self.config=SimpleNamespace(max_position_embeddings=4096)
        self.generation_config=SimpleNamespace(eos_token_id=[255]);self.settings=None
        self.continuation=continuation if continuation is not None else list(r'<think>2</think> \boxed{2}'.encode())+[255]
    def generate(self,input_ids,attention_mask,**settings):
        self.settings=settings;assert input_ids.shape[0]==1 and torch.equal(attention_mask,torch.ones_like(input_ids))
        ids=self.continuation[:settings['max_new_tokens']]
        return torch.cat([input_ids,input_ids.new_tensor([ids])],dim=1)

class RunnerTests(unittest.TestCase):
    def question(self):return prompt_record(TokenizerFixture(),'test:1','What is 1+1?')
    def test_prescribed_generate_settings_and_costs(self):
        model=ModelFixture();row=generate_record(model,TokenizerFixture(),self.question(),'parallel_0')
        self.assertEqual(row['extracted_answer'],'2');self.assertEqual(row['retained_output_tokens']+1,row['output_tokens'])
        self.assertTrue(row['ended_by_eos']);self.assertFalse(row['truncated'])
        self.assertEqual(model.settings['repetition_penalty'],1.);self.assertEqual(model.settings['max_new_tokens'],1024)
        self.assertEqual(model.settings['num_beams'],1);self.assertEqual(model.settings['num_return_sequences'],1)
    def test_empty_retained_response_is_saved(self):
        row=generate_record(ModelFixture([255]),TokenizerFixture(),self.question(),'parallel_0')
        self.assertEqual(row['response'],'');self.assertEqual(row['output_tokens'],1);self.assertIsNone(row['extracted_answer'])
    def test_cap_and_eos_at_cap_are_distinct(self):
        row=generate_record(ModelFixture(),TokenizerFixture(),self.question(),'parallel_0',smoke=True,smoke_cap=4)
        self.assertTrue(row['truncated']);self.assertTrue(row['reached_token_cap']);self.assertFalse(row['ended_by_eos'])
        row=generate_record(ModelFixture([65,255]),TokenizerFixture(),self.question(),'parallel_0',smoke=True,smoke_cap=2)
        self.assertFalse(row['truncated']);self.assertTrue(row['reached_token_cap']);self.assertTrue(row['ended_by_eos'])
    def test_main_budget_cannot_be_overridden(self):
        with self.assertRaises(AssertionError):generate_record(ModelFixture(),TokenizerFixture(),self.question(),'parallel_0',smoke_cap=4)
    def test_revision_uses_initial_verbatim(self):
        initial='  <think>work</think> \\boxed{2}\n'
        row=generate_record(ModelFixture(),TokenizerFixture(),self.question(),'revision',initial_response=initial)
        self.assertEqual(row['request']['messages'][2]['content'],initial);self.assertEqual(row['max_new_tokens'],2048)
        self.assertGreater(row['input_tokens'],self.question()['input_tokens'])
    def test_base_prompt_mismatch_rejected(self):
        q=self.question();q['input_ids']=[1]
        with self.assertRaises(AssertionError):generate_record(ModelFixture(),TokenizerFixture(),q,'parallel_0')
    def test_call_plan_and_no_gold_inputs(self):
        questions=[prompt_record(TokenizerFixture(),f'test:{i}','Question?') for i in range(50)]
        plan=call_plan(questions);self.assertEqual(len(plan),250);self.assertEqual(sum(p['max_new_tokens'] for p in plan),409600)
        questions[0]['gold_answer']='LEAK'
        with self.assertRaises(AssertionError):call_plan(questions)
    def test_resume_hash_and_seed_contract(self):
        q=self.question();record=generate_record(ModelFixture(),TokenizerFixture(),q,'parallel_0')
        expected={'sample_id':q['sample_id'],'call':'parallel_0','seed':seed_for(q['sample_id'],'parallel_0'),'max_new_tokens':1024}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'call.json';identity={'test':'fixture'}
            self.assertIsNone(load_call(path,identity,expected));save_call(path,record,identity)
            self.assertEqual(load_call(path,identity,expected)['response'],record['response'])
            path.write_text(path.read_text()+' ')
            with self.assertRaises(AssertionError):load_call(path,identity,expected)
    def test_shared_cost_is_charged_to_both_strategies(self):
        calls={name:{'input_tokens':10,'output_tokens':20,'retained_output_tokens':19,'seconds':2.,'peak_gpu_allocated_bytes':100} for name in CALLS}
        calls['revision']['input_tokens']=40;cost=strategy_costs(calls)
        self.assertEqual(cost['parallel']['output_tokens'],60);self.assertEqual(cost['sequential']['output_tokens'],40)
        self.assertEqual(cost['sequential']['input_tokens'],50);self.assertEqual(cost['sequential']['seconds'],4.)
    def test_real_tiny_qwen2_generation_seed(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM
        torch.manual_seed(7)
        cfg=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096,eos_token_id=255,pad_token_id=0,bos_token_id=1)
        model=Qwen2ForCausalLM(cfg).eval().requires_grad_(False)
        kwargs={'smoke':True,'smoke_cap':4}
        a=generate_record(model,TokenizerFixture(),self.question(),'parallel_0',**kwargs)
        b=generate_record(model,TokenizerFixture(),self.question(),'parallel_0',**kwargs)
        self.assertEqual(a['generated_token_ids'],b['generated_token_ids'])
    def test_loading_api_fp16_device_map(self):
        from transformers import Qwen2Config,Qwen2ForCausalLM,AutoModelForCausalLM
        cfg=Qwen2Config(vocab_size=256,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=4096)
        with tempfile.TemporaryDirectory() as temporary:
            Qwen2ForCausalLM(cfg).save_pretrained(temporary)
            loaded,info=AutoModelForCausalLM.from_pretrained(temporary,local_files_only=True,trust_remote_code=False,
              dtype=torch.float16,attn_implementation='sdpa',device_map={'':'cpu'},output_loading_info=True)
            self.assertFalse(info.get('missing_keys'));self.assertFalse(info.get('mismatched_keys'))
            self.assertTrue(all(p.dtype==torch.float16 for p in loaded.parameters()))
    def test_cpu_prepare_never_probes_cuda(self):
        from unittest.mock import patch
        from task3_live_run import cpu_prepare
        from task3_live_runtime import write,sha
        from task3_live_spec import recipe
        from task3_live_run import SOURCE_FILES
        with tempfile.TemporaryDirectory() as temporary:
            task=Path(temporary);(task/'code').mkdir();(task/'assets/model').mkdir(parents=True)
            for name in SOURCE_FILES:(task/'code'/name).write_text('test fixture source\n')
            (task/'assets/model/model.safetensors').write_bytes(b'fixture')
            questions=[prompt_record(TokenizerFixture(),f'test:{i}','Question?') for i in range(50)]
            write(task/'configs/scorer_frozen.json',{'packages':{},'source_sha256':{}})
            write(task/'results/offline/finished.json',{'passed':True})
            write(task/'splits/question_splits.json',{'datasets':{'math':{'live_math':[q['sample_id'] for q in questions],'evaluation':[q['sample_id'] for q in questions]}}})
            write(task/'splits/live_math_questions.json',{'fixture':True})
            write(task/'data/live_prompts.json',{'questions':questions});write(task/'data/live_math_references.json',{'fixture':True})
            write(task/'configs/live_recipe.json',recipe())
            asset={'model_id':'Qwen/Qwen2.5-3B-Instruct','revision':MODEL_REVISION,'source_sha256':{},'cpu_packages':{},
              'source_path':str(task/'assets/model'),'files':{'model.safetensors':{'bytes':7}},
              'scorer_sha256':sha(task/'configs/scorer_frozen.json'),'offline_finished_sha256':sha(task/'results/offline/finished.json'),
              'question_splits_sha256':sha(task/'splits/question_splits.json'),'selected_questions_sha256':sha(task/'splits/live_math_questions.json'),
              'prompts_sha256':sha(task/'data/live_prompts.json'),'recipe_sha256':sha(task/'configs/live_recipe.json'),
              'references_sha256':sha(task/'data/live_math_references.json')}
            write(task/'configs/live_assets.json',asset)
            write(task/'results/live_assets_preflight.json',{'passed':True,'assets_sha256':sha(task/'configs/live_assets.json')})
            import contextlib,io
            with patch('torch.cuda.is_available',side_effect=AssertionError('CPU preparation must not probe CUDA')),contextlib.redirect_stdout(io.StringIO()):
                cpu_prepare(task);cpu_prepare(task)
            self.assertEqual(json.loads((task/'results/live_runner_cpu_preflight.json').read_text())['physical_calls'],250)

if __name__=='__main__':unittest.main()
