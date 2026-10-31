import unittest
import numpy as np
from task3_live_score import select_parallel, interval

class ScoringTests(unittest.TestCase):
    def test_equivalent_votes(self):
        self.assertEqual(select_parallel(['1/2',r'\frac{1}{2}','7'])[0],0)
    def test_ties_use_first_sample(self):
        self.assertEqual(select_parallel(['9','2','4'])[0],0)
        self.assertEqual(select_parallel(['4','9','2'])[0],0)
    def test_missing_votes_are_retained(self):
        self.assertEqual(select_parallel(['7',None,None])[0],1)
        self.assertEqual(select_parallel([None,'7','8'])[0],0)
    def test_majority_not_first(self):
        self.assertEqual(select_parallel(['3','7','7'])[0],1)
    def test_paired_identical_has_zero_difference(self):
        draws=np.random.default_rng(6304).integers(0,50,(1000,50))
        self.assertEqual(interval(np.zeros(50),draws),{'mean':0.,'ci95':[0.,0.]})

if __name__=='__main__':unittest.main()
