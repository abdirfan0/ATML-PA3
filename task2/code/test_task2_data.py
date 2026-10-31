import copy
import unittest
import numpy as np
from task2_answers import normalize_answer,vqa_soft_accuracy
from task2_select_data import choose_natural,controls,make_grounding,question_for,choose_captions

class NormalizationTests(unittest.TestCase):
    def test_normalization(self):
        for original,expected in [(' The two cats! ','2 cats'),('1,000','1000'),('dont','don\'t'),('3.14.','3.14'),('A cat','cat'),('', '')]:
            self.assertEqual(normalize_answer(original),expected)
    def test_soft_consensus_is_leave_one_out(self):
        for matches,expected in [(0,0),(1,.3),(2,.6),(3,.9),(4,1),(10,1)]:
            self.assertAlmostEqual(vqa_soft_accuracy('yes',['yes']*matches+['no']*(10-matches)),expected)
        self.assertEqual(vqa_soft_accuracy('', ['']*10),0)

class SelectionTests(unittest.TestCase):
    def candidates(self):
        result={}
        for t,typ in enumerate(('yes/no','number','other')):
            result[typ]=[{'pair_id':f'{typ}:{i}','question_type':typ,'examples':[{'image_id':100*t+2*i},{'image_id':100*t+2*i+1}]} for i in range(8)]
        return result
    def test_natural_determinism_quotas_and_isolation(self):
        selected=choose_natural(self.candidates(),dict.fromkeys(('yes/no','number','other'),2),np.random.default_rng(6304),set())
        repeated=choose_natural(self.candidates(),dict.fromkeys(('yes/no','number','other'),2),np.random.default_rng(6304),set())
        self.assertEqual(selected,repeated)
        used={e['image_id'] for p in selected for e in p['examples']}
        second=choose_natural(self.candidates(),dict.fromkeys(('yes/no','number','other'),2),np.random.default_rng(3),used)
        self.assertFalse({e['image_id'] for p in selected for e in p['examples']} & {e['image_id'] for p in second for e in p['examples']})
    def test_insufficient_pairs_rejected(self):
        with self.assertRaises(AssertionError):choose_natural(self.candidates(),dict.fromkeys(('yes/no','number','other'),9),np.random.default_rng(1),set())
    def instances(self):
        return {'categories':{1:'person',2:'dog'},'present':{1:set(range(8)),2:set(range(4,12))},
                'counts':{1:{i:i%4+1 for i in range(8)},2:{i:i%3+1 for i in range(4,12)}},'crowds':{1:{3},2:{7}}}
    def test_presence_and_counting_annotations(self):
        source=self.instances()
        for kind in ('presence','counting'):
            pairs=make_grounding(kind,20,'train2014',list(range(16)),source,np.random.default_rng(6304))
            self.assertEqual(len({p['pair_id'] for p in pairs}),20)
            for pair in pairs:
                a,b=pair['examples'];cat=pair['category_id']
                self.assertNotEqual(a['target'],b['target'])
                for example in pair['examples']:
                    image=example['image_id']
                    if kind=='presence':self.assertEqual(example['target']=='yes',image in source['present'][cat])
                    else:
                        self.assertNotIn(image,source['crowds'][cat]);self.assertEqual(int(example['target']),source['counts'][cat][image])
    def test_grounding_excluded_images(self):
        used={0,1,2,3,12,13}
        initial=used.copy()
        pairs=make_grounding('presence',2,'val2014',list(range(16)),self.instances(),np.random.default_rng(5),used)
        chosen=[e['image_id'] for p in pairs for e in p['examples']]
        self.assertEqual(len(chosen),len(set(chosen)))
        self.assertFalse(initial & set(chosen))
    def test_mismatched_maps_fixed_and_exclude_original_pair(self):
        pairs=choose_natural(self.candidates(),dict.fromkeys(('yes/no','number','other'),2),np.random.default_rng(6304),set())
        for p in pairs:
            for i,e in enumerate(p['examples']):e['example_id']=p['pair_id']+':'+str(i)
        first=controls(pairs,np.random.default_rng(8));second=controls(pairs,np.random.default_rng(8))
        self.assertEqual(first,second)
        for row in first:self.assertNotIn(row['mismatched_image_id'],[row['correct_image_id'],row['complementary_image_id']])
    def test_caption_sampling_is_exact_without_replacement(self):
        captions={i:[{'caption_id':10*i+j} for j in range(5)] for i in range(4)}
        selected=choose_captions(list(range(4)),captions,3,np.random.default_rng(6304))
        self.assertEqual(len(selected),12)
        self.assertEqual(len({r['caption_id'] for r in selected}),12)
    def test_template_questions(self):
        self.assertEqual(question_for('counting','person'),'How many people can you see?')
        self.assertEqual(question_for('presence','airplane'),'Does this image contain an airplane?')

if __name__=='__main__':unittest.main()
