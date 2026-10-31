"""CPU checks for leakage boundaries, ties, costs, calibration and policies."""
import copy,unittest
import numpy as np
from task3_answers import NO_ANSWER
from task3_selection import vote,rwc,visual_scores,select_prefix,stop_at,choose_policy
from task3_offline import question_analysis,outcome,select_policies,summarize

class OfflineTests(unittest.TestCase):
    def payload(self,label='mathvista',gold='2'):
        rows=[];kl=[];vision=[]
        for i in range(32):
            k=[.1+i*.01,.2+i*.01];v=[.2,.4];kl.extend(k);vision.extend(v)
            rows.append({'rollout_id':i,'extracted_answer':'1' if i<4 else '2','n_generated_text_tokens':2,
              'mean_logp':-1.-i*.01,'mean_kl':float(np.mean(k)),'hit_max_new_tokens':False,'generated_text':'test response'})
        q={'metadata':{'sample_id':'fixture','question_text':'Fixture?','gold_answer':gold,'rollouts':rows},
          'offsets':np.arange(0,65,2),'kl':np.array(kl,dtype=np.float32),'vision':np.array(vision,dtype=np.float32)}
        meta={'question_type':'free_form','answer_type':'integer','precision':None}
        return (label,q,[list(range(32)) for _ in range(10)],meta if label=='mathvista' else None,[0])
    def test_sc_earliest_support(self):
        self.assertEqual(vote(['B','A','A','B'])['group'],'B')
    def test_rwc_average_ranks(self):
        result=rwc(['A','B','B'],[3,2,2]);self.assertEqual(result['votes'],{'A':4,'B':2})
        self.assertAlmostEqual(result['margin'],1/3)
    def test_rwc_exact_vote_tie(self):
        result=rwc(['A','B','B','A'],[4,3,2,1])
        self.assertEqual(result['group'],'A');self.assertFalse(result['unique']);self.assertEqual(result['margin'],0)
    def test_equal_scores_reduce_to_sc(self):
        self.assertEqual(rwc(['B','A','A'],[1,1,1])['group'],'A')
    def test_visconf_token_covariance(self):
        score=visual_scores([np.array([0.,10.])]*2,[np.array([0.,1.])]*2)
        self.assertTrue(np.allclose(score['visconf'],.25));self.assertTrue(np.allclose(score['tokenwise'],.5))
        self.assertTrue(np.allclose(score['tokenwise']-score['visconf'],score['covariance']))
    def test_constant_kl_is_finite(self):
        score=visual_scores([np.ones(2)]*2,[np.array([.1,.9])]*2)
        self.assertTrue(np.array_equal(score['visconf'],[0.,0.]))
    def test_future_scores_do_not_change_prefix(self):
        first=self.payload();second=copy.deepcopy(first);second[1]['kl'][8:]=100000
        for row in second[1]['metadata']['rollouts'][4:]:row['mean_kl']=100000
        a=question_analysis(first);b=question_analysis(second)
        for n in ('1','2','4'):
            self.assertEqual(a['orders'][0][n]['choices'],b['orders'][0][n]['choices'])
            self.assertEqual(a['orders'][0][n]['visconf_percentiles'],b['orders'][0][n]['visconf_percentiles'])
    def test_gold_does_not_change_choices_or_stopping(self):
        for label in ('math','mathvista'):
            a=question_analysis(self.payload(label,'1'));b=question_analysis(self.payload(label,'2'))
            for oa,ob in zip(a['orders'],b['orders']):
                self.assertEqual(oa,ob) # every selector/stopping input is gold-free
                for n in oa:
                    self.assertEqual(oa[n]['choices'],ob[n]['choices']);self.assertEqual(oa[n]['sc'],ob[n]['sc']);self.assertEqual(oa[n]['rwc'],ob[n]['rwc'])
                self.assertEqual(stop_at(oa,'quality',.75,.1),stop_at(ob,'quality',.75,.1))
    def test_no_answer_leader_does_not_stop(self):
        x={'sc':{'unique':True,'group':NO_ANSWER,'support':1.},'rwc':{'group':NO_ANSWER,'margin':1.}}
        prefixes={str(n):x for n in (4,8,16,32)}
        self.assertEqual(stop_at(prefixes,'agreement',.5),32)
        self.assertEqual(stop_at(prefixes,'quality',.5,0),32)
    def test_first_answer_consumes_only_first(self):
        q=question_analysis(self.payload());p=q['orders'][0]['32']
        self.assertEqual(outcome(q,p,'FirstAns')['token_cost'],2);self.assertEqual(outcome(q,p,'SC')['token_cost'],64)
        self.assertEqual(outcome(q,p,'FirstAns')['rollouts'],1)
    def test_threshold_boundary_and_ties(self):
        def row(acc,cost,tau,gamma):return {'accuracy':acc,'token_cost':cost,'setting':{'family':'quality','tau':tau,'gamma':gamma}}
        rows=[row(.79,100,.75,.25),row(.79,100,.5,.25),row(.79,100,.5,.1),row(.8,200,.5,0)]
        chosen=choose_policy(rows,.8);self.assertEqual(chosen['setting']['tau'],.5);self.assertEqual(chosen['setting']['gamma'],.1)
    def test_order_averaged_reports_and_recoverable_stops(self):
        a=question_analysis(self.payload());b=copy.deepcopy(a);b['sample_id']='fixture2'
        policies=select_policies([a,b]);summary,raw=summarize('mathvista',[a,b],policies)
        self.assertEqual(len(summary['fixed']),72)
        for row in summary['fixed']:
            self.assertAlmostEqual(row['sampling_failure']+row['selection_failure'],1-row['accuracy'])
        for family in ('agreement','quality'):
            self.assertEqual(sum(summary['adaptive'][0]['stop_counts'].values()),20)
            self.assertEqual(len(raw[family]),2)

if __name__=='__main__':unittest.main()
