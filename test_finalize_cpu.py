"""CPU-only checks for corrupt/incomplete per-example export rejection."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from finalize_task1 import verify_samples

class SampleAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'samples.jsonl'
        self.addCleanup(self.temp.cleanup)
    def write(self,rows):self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def test_likelihood_scores_and_coverage(self):
        self.write([{'doc_id':0,'filter':'none','acc':1,'acc_norm':0},{'doc_id':1,'filter':'none','acc':0,'acc_norm':1}])
        with patch.dict('finalize_task1.TASKS',{'hellaswag':2}):
            result=verify_samples(self.path,{'acc,none':.5,'acc_norm,none':.5},'hellaswag')
        self.assertEqual(result['rows'],2)
    def test_duplicate_rejected(self):
        self.write([{'doc_id':0,'filter':'none','acc':1}]*2)
        with patch.dict('finalize_task1.TASKS',{'boolq':2}),self.assertRaises(AssertionError):
            verify_samples(self.path,{'acc,none':1},'boolq')
    def test_missing_example_rejected(self):
        self.write([{'doc_id':0,'filter':'none','acc':1}])
        with patch.dict('finalize_task1.TASKS',{'boolq':2}),self.assertRaises(AssertionError):
            verify_samples(self.path,{'acc,none':1},'boolq')
    def test_score_mismatch_rejected(self):
        self.write([{'doc_id':0,'filter':'none','acc':1}])
        with patch.dict('finalize_task1.TASKS',{'boolq':1}),self.assertRaises(AssertionError):
            verify_samples(self.path,{'acc,none':0},'boolq')
    def test_gsm_filters_and_generation_rates(self):
        rows=[]
        for filt in ('strict-match','flexible-extract'):
            for i in (0,1):
                rows.append({'doc_id':i,'filter':filt,'exact_match':int(i==0 and filt=='flexible-extract'),
                             'generation':{'truncated':i==1},'filtered_resps':['2' if i==0 else '[invalid]']})
        self.write(rows)
        summary={'questions':2,'truncation_rate':.5,'missing_extraction_rate':.5}
        with patch.dict('finalize_task1.TASKS',{'gsm8k_cot':2}):
            result=verify_samples(self.path,{'exact_match,strict-match':0,'exact_match,flexible-extract':.5},'gsm8k_cot',summary)
        self.assertEqual(result['rows'],4)
    def test_extraction_rate_mismatch_rejected(self):
        self.write([{'doc_id':0,'filter':f,'exact_match':0,'generation':{'truncated':False},'filtered_resps':['[invalid]']} for f in ('strict-match','flexible-extract')])
        with patch.dict('finalize_task1.TASKS',{'gsm8k_cot':1}),self.assertRaises(AssertionError):
            verify_samples(self.path,{'exact_match,strict-match':0,'exact_match,flexible-extract':0},'gsm8k_cot',{'questions':1,'truncation_rate':0,'missing_extraction_rate':0})

if __name__=='__main__':unittest.main()
