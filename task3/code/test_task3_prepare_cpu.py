"""Small CPU tests for joins, token alignment, normalization and cache reuse."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from task3_prepare import check_vectors,index_table,candidate_orders,prepare_dataset
from task3_pool import sha256,PreparedPool

class PreparationTests(unittest.TestCase):
    def vectors(self):return np.array([10,11]),np.array([-1.,-2.]),np.array([.3,.4])
    def test_probability_alignment(self):
        t,l,k=self.vectors()
        self.assertIsNone(check_vectors(t,l,k,2))
        with self.assertRaises(AssertionError):check_vectors(t,l[:1],k,2)
    def test_nonfinite_probability(self):
        t,l,k=self.vectors();k[0]=np.nan
        with self.assertRaises(AssertionError):check_vectors(t,l,k,2)
    def test_negative_kl(self):
        t,l,k=self.vectors();k[0]=-.1
        with self.assertRaises(AssertionError):check_vectors(t,l,k,2)
    def test_normalized_attention_not_raw_mass(self):
        t,l,k=self.vectors();a=(np.array([.2,.2]),np.array([.3,.2]),np.array([0.,.1]),np.array([0,1]))
        self.assertTrue(np.allclose(check_vectors(t,l,k,2,a),[.4,.4]))
    def test_preceding_counts_not_scalar_length(self):
        t,l,k=self.vectors();a=(np.array([.2,.2]),np.array([.3,.2]),np.array([0.,.1]),np.array([2,2]))
        with self.assertRaises(AssertionError):check_vectors(t,l,k,2,a)
    def test_zero_attention_denominator(self):
        t,l,k=self.vectors();a=(np.zeros(2),np.zeros(2),np.zeros(2),np.arange(2))
        with self.assertRaises(AssertionError):check_vectors(t,l,k,2,a)
    def test_empty_rollout_retained(self):
        z=np.array([]);self.assertIsNone(check_vectors(z,z,z,0))
    def test_duplicate_join_key(self):
        table=pa.Table.from_pylist([{'dataset':'x','sample_id':'1','rollout_id':0}]*2)
        with self.assertRaises(AssertionError):index_table(table)
    def test_orders_repeatable_and_complete(self):
        ids={'math':['test:2','test:10'],'mathvista':['2','10']}
        a=candidate_orders(ids);self.assertEqual(a,candidate_orders(ids))
        for questions in a['datasets'].values():
            for orders in questions.values():
                self.assertEqual(len(orders),10)
                self.assertEqual(len({tuple(v) for v in orders}),10)
                self.assertTrue(all(sorted(v)==list(range(32)) for v in orders))
    def fixture(self,root,label):
        g=[];p=[];a=[]
        for qid in ['1','2']:
            for rid in range(32):
                key={'dataset':label,'sample_id':qid,'rollout_id':rid}
                g.append({**key,'model':'unit-test-fixture','question_text':'What is 1+1?',
                  'generated_text':'work \\boxed{2}','generated_token_ids':[10,11],'n_generated_text_tokens':2,
                  'hit_max_new_tokens':False,'gold_answer':'2','extracted_answer':'2'})
                p.append({**key,'logp':[-1.,-2.],'kl_u_p':[.3+rid/100,.4]})
                a.append({**key,'n_image_tokens':4,'n_prompt_text_tokens':6,'img_attn_total':[.2,.2],
                  'prompt_text_attn_total':[.3,.2],'generated_text_attn_total':[0.,.1],'n_generated_text_tokens':[0,1]})
        sources=root/'raw';sources.mkdir();files={}
        data={'generations.parquet':g,'probability_metrics.parquet':list(reversed(p))}
        if label=='mathvista':data['attention_metrics.parquet']=a[::2]+a[1::2]
        for name,rows in data.items():
            path=sources/name;pq.write_table(pa.Table.from_pylist(rows),path)
            files[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':sha256(path)}
        return files
    def test_shuffled_join_and_safe_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            task=Path(d);work=task/'work';work.mkdir();files=self.fixture(task,'mathvista')
            expected={'questions':2,'rows':64,'tokens':128,'capped':0,'missing':0};identity={'fixture':True}
            first=prepare_dataset(task,'mathvista',files,identity,work,expected)
            with PreparedPool(task,'mathvista') as pool:
                q=pool.get('1');self.assertEqual(q['offsets'].tolist(),list(range(0,65,2)))
                self.assertTrue(np.allclose(q['vision'],.4))
                self.assertAlmostEqual(float(q['kl'][20]),.4,places=6) # rollout 10's first value
                self.assertEqual(q['metadata']['rollouts'][10]['rollout_id'],10)
            second=prepare_dataset(task,'mathvista',files,identity,work,expected);self.assertEqual(first,second)
    def test_math_extraction_and_totals(self):
        with tempfile.TemporaryDirectory() as d:
            task=Path(d);work=task/'work';work.mkdir();files=self.fixture(task,'math')
            result=prepare_dataset(task,'math',files,{'fixture':True},work,{'questions':2,'rows':64,'tokens':128,'capped':0,'missing':0})
            self.assertEqual(result['statistics']['rows'],64)

if __name__=='__main__':unittest.main()
