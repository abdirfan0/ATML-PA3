"""CPU tests for live prompts, fixed allowances and reproducible independent seeds."""
import unittest
from task3_live_spec import *
class TokenizerFixture:
    def apply_chat_template(self,chat,tokenize,add_generation_prompt):
        assert tokenize is False and add_generation_prompt
        return '\n'.join(x['role']+':'+x['content'] for x in chat)+'\nassistant:'
    def __call__(self,text,add_special_tokens):return {'input_ids':list(text.encode())}
class LiveTests(unittest.TestCase):
    def test_system_and_original_problem(self):
        problem='Original problem\n  Preserve spacing.';chat=messages(problem)
        self.assertEqual(chat[0],{'role':'system','content':'You are a helpful assistant.'})
        self.assertEqual(chat[1]['content'],problem+'\n\n'+INSTRUCTION)
    def test_native_template_returns_ids(self):
        r=prompt_record(TokenizerFixture(),'test:1','Question?')
        self.assertEqual(r['input_tokens'],len(r['input_ids']));self.assertEqual(len(r['messages']),2)
        self.assertNotIn('gold_answer',r)
    def test_revision_reuses_initial_verbatim(self):
        initial='<think>work</think> \\boxed{2}'
        chat=messages('Question?',initial);self.assertEqual(chat[2],{'role':'assistant','content':initial})
        self.assertEqual(chat[3]['content'],REVISION_INSTRUCTION)
        self.assertEqual(chat[:2],messages('Question?'))
    def test_strategy_allowances(self):
        self.assertTrue(all(sum(CALLS[c] for c in calls)==3072 for calls in STRATEGIES.values()))
    def test_shared_initial_is_not_generated_twice(self):
        self.assertIn('parallel_0',STRATEGIES['parallel']);self.assertIn('parallel_0',STRATEGIES['sequential'])
        self.assertEqual(len(CALLS),5);self.assertEqual(sum(CALLS.values()),8192)
    def test_independent_reproducible_seeds(self):
        first=[seed_for(f'test:{q}',c) for q in range(50) for c in CALLS]
        self.assertEqual(first,[seed_for(f'test:{q}',c) for q in range(50) for c in CALLS])
        self.assertEqual(len(set(first)),250)
    def test_original_pinned_model(self):
        self.assertEqual(MODEL_ID,'Qwen/Qwen2.5-3B-Instruct')
        self.assertEqual(MODEL_REVISION,'aa8e72537993ba99e69dfaafa59ed015b17504d1')
    def test_prescribed_sampling(self):
        r=recipe();self.assertTrue(r['do_sample']);self.assertEqual(r['batch_size'],1)
        self.assertEqual((r['temperature'],r['top_p'],r['top_k'],r['repetition_penalty']),(.7,.8,20,1.))
        self.assertEqual(r['dtype'],'float16');self.assertFalse(r['quantization'])
    def test_pinned_dataset_solution_reference(self):
        from task3_live_assets import original_reference
        self.assertEqual(original_reference({'solution':r'Work and \boxed{\frac{1}{2}}'}),r'\frac{1}{2}')
        with self.assertRaises(AssertionError):original_reference({'solution':'No boxed answer'})
if __name__=='__main__':unittest.main()
