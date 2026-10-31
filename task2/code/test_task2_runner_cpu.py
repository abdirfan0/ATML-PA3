import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from task2_sampler import CurriculumSampler
from task2_runtime import rng_state,restore_rng,torch_write
from task2_train import model_state,load_model_state,make_optimizer,latest
from task2_evaluate import greedy,controlled_rows,pair_arrays,summarize_final
from task2_features import FeatureStore
from test_task2_vlm_cpu import tiny,example

class SamplerTests(unittest.TestCase):
    def sampler(self):
        captions=[{'example_id':f'c{i}'} for i in range(24)]
        def pairs(prefix):return [{'examples':[{'example_id':f'{prefix}{i}a'},{'example_id':f'{prefix}{i}b'}]} for i in range(10)]
        return CurriculumSampler(captions,pairs('p'),pairs('k'),pairs('n'))
    def test_stage1_two_passes_and_recovery(self):
        sampler=self.sampler();all_rows=[]
        for _ in range(3):rows,_=sampler.batch(1);all_rows+=rows
        self.assertEqual(sampler.caption_position,48)
        self.assertTrue(all(sum(r['example_id']==f'c{i}' for r in all_rows)==2 for i in range(24)))
        with self.assertRaises(AssertionError):sampler.batch(1)
    def test_mixtures_pair_integrity_and_exact_sampler_resume(self):
        sampler=self.sampler();sampler.batch(2);state=sampler.state_dict()
        expected=[sampler.batch(3) for _ in range(8)]
        restored=self.sampler();restored.load_state_dict(state)
        self.assertEqual(expected,[restored.batch(3) for _ in range(8)])
        for rows,counts in expected:
            self.assertEqual(sum(counts.values()),8)
            for i in range(0,16,2):
                if not rows[i]['example_id'].startswith('c'):
                    self.assertEqual(rows[i]['example_id'][:-1],rows[i+1]['example_id'][:-1])
    def test_probabilities_are_draws_not_exact_batch_quotas(self):
        sampler=self.sampler();counts=[]
        for _ in range(20):_,mixture=sampler.batch(2);counts.append(tuple(sorted(mixture.items())))
        self.assertGreater(len(set(counts)),1)

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(min(4,torch.get_num_threads()))
    def test_trainable_export_optimizer_scheduler_and_rng_resume(self):
        model=tiny();model.enable_lora();model.train()
        optimizer,scheduler=make_optimizer(model,2)
        features=torch.randn(2,196,768,generator=torch.Generator().manual_seed(42));examples=[example(),example(answer=(4,))]
        def step(m,opt,sch):
            opt.zero_grad(set_to_none=True);losses,_=m(examples,features);losses.mean().backward();opt.step();sch.step()
        step(model,optimizer,scheduler);step(model,optimizer,scheduler)
        state={'model':model_state(model),'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),'rng':rng_state()}
        step(model,optimizer,scheduler);expected=model_state(model)
        fresh=tiny();fresh.enable_lora();fresh.train();new_opt,new_scheduler=make_optimizer(fresh,2)
        load_model_state(fresh,state['model']);new_opt.load_state_dict(state['optimizer']);new_scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
        step(fresh,new_opt,new_scheduler);actual=model_state(fresh)
        for key in expected['connector']:torch.testing.assert_close(expected['connector'][key],actual['connector'][key],rtol=0,atol=0)
        for name in expected['adapters']:
            for factor in ('A','B'):torch.testing.assert_close(expected['adapters'][name][factor],actual['adapters'][name][factor],rtol=0,atol=0)
        self.assertEqual(scheduler.state_dict(),new_scheduler.state_dict())
    def test_latest_ignores_incomplete_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for update in (250,500):
                path=root/f'update_{update:04d}/training_state.pt';torch_write(path,{})
            (root/'update_0250/complete.json').write_text('{}')
            self.assertEqual(latest(root).parent.name,'update_0250')
    def test_feature_store_image_mapping(self):
        from task2_runtime import sha256
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'features/patch196';folder.mkdir(parents=True)
            values=np.stack([np.full((196,768),3,dtype=np.float16),np.full((196,768),7,dtype=np.float16)])
            np.save(folder/'f.npy',values)
            (folder/'manifest.json').write_text(json.dumps({'shards':[{'file_name':'f.npy','sha256':sha256(folder/'f.npy'),'images':[{'image_split':'train2014','image_id':1},{'image_split':'val2014','image_id':2}]}]}))
            store=FeatureStore(root)
            selected=store.get([{'image_split':'val2014','image_id':2},{'image_split':'train2014','image_id':1}])
            self.assertEqual(selected[:,0,0].tolist(),[7,3])

class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(min(4,torch.get_num_threads()))
    def test_cached_greedy_matches_full_prefix_and_batch_padding(self):
        model=tiny().eval();examples=[example(after=3),example(after=1)];features=torch.randn(2,196,768)
        cached=greedy(model,examples,features,set(),4)
        with torch.inference_mode():
            batch=model.make_batch(examples,features,padding_side='left',include_response=False);batch.pop('labels');expected=[[],[]]
            for _ in range(4):
                output=model.backbone(**batch,use_cache=False);next_ids=output.logits[:,-1].argmax(-1)
                for row,value in zip(expected,next_ids.tolist()):row.append(value)
                batch['inputs_embeds']=torch.cat([batch['inputs_embeds'],model.backbone.get_input_embeddings()(next_ids)[:,None]],dim=1)
                batch['attention_mask']=torch.cat([batch['attention_mask'],batch['attention_mask'].new_ones((2,1))],dim=1)
                batch['position_ids']=(batch['attention_mask'].cumsum(-1)-1).clamp_min(0).masked_fill(batch['attention_mask']==0,0)
        self.assertEqual([r['output_token_ids'] for r in cached],expected)
        self.assertTrue(all(r['truncated'] for r in cached))
        for index,e in enumerate(examples):
            single=greedy(model,[e],features[index:index+1],set(),4)
            self.assertEqual(single[0]['output_token_ids'],cached[index]['output_token_ids'])
    def test_finished_rows_stop_but_other_rows_continue(self):
        class Scripted(nn.Module):
            def __init__(self):super().__init__();self.emb=nn.Embedding(10,32);self.config=SimpleNamespace(hidden_size=32)
            def get_input_embeddings(self):return self.emb
            def forward(self,inputs_embeds=None,input_ids=None,past_key_values=None,**kwargs):
                batch=inputs_embeds.shape[0] if inputs_embeds is not None else input_ids.shape[0]
                length=inputs_embeds.shape[1] if inputs_embeds is not None else 1
                logits=torch.zeros(batch,length,10);logits[0,-1,3]=1;logits[1,-1,4 if past_key_values is None else 3]=1
                return SimpleNamespace(logits=logits,past_key_values=1)
        from task2_vlm import VisualLanguageModel,Connector
        model=VisualLanguageModel(Scripted(),Connector(language_dim=32)).eval()
        result=greedy(model,[example(),example()],None,{3},16)
        self.assertEqual([r['output_tokens'] for r in result],[1,2]);self.assertTrue(all(r['ended_by_eos'] for r in result))
    def test_controls_preserve_targets(self):
        pairs=[{'examples':[{'example_id':'a','image_id':1,'target':'yes'},{'example_id':'b','image_id':2,'target':'no'}]}]
        swapped=controlled_rows(pairs,'complementary')
        self.assertEqual([r['image_id'] for r in swapped],[2,1]);self.assertEqual([r['target'] for r in swapped],['yes','no'])
        self.assertEqual(pairs[0]['examples'][0]['image_id'],1)
    def test_pair_bootstrap_uses_shared_indices_and_keeps_empties(self):
        pairs=[{'pair_id':'p0','question_type':'presence'},{'pair_id':'p1','question_type':'presence'}]
        rows=[]
        for pair in range(2):
            for side in range(2):
                rows.append({'pair_id':f'p{pair}','example_id':f'{pair}:{side}','score':1-pair,'reference_nll':float(pair+1),
                  'canonical_exact_match':pair==0,'normalized_prediction':str(side) if pair==0 else '',
                  'partner_exact_match':False,'generation_seconds_amortized':.1})
        all_records={condition:copy.deepcopy(rows) for condition in ('correct','complementary','mismatched','no_image')}
        for row in all_records['complementary']:row['reference_nll']+=1
        summary=summarize_final(all_records,pairs,'grounding',resamples=4000)
        overall=summary['groups']['overall']
        self.assertEqual(overall['conditions']['correct']['accuracy']['mean'],.5)
        self.assertEqual(overall['conditions']['correct']['empty_answer_rate']['mean'],.5)
        self.assertEqual(overall['conditions']['correct']['pair_accuracy']['mean'],.5)
        self.assertEqual(overall['correct_image_advantages']['accuracy_correct_minus_complementary']['ci95'],[0.,0.])
        self.assertEqual(overall['correct_image_advantages']['nll_complementary_minus_correct']['ci95'],[1.,1.])

if __name__=='__main__':unittest.main()
