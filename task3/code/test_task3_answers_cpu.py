"""CPU scorer tests: missing values, equivalence, gold-free groups and boxes."""
import unittest
from task3_answers import *

class ScorerTests(unittest.TestCase):
    def test_nested_last_box(self):
        self.assertEqual(last_boxed_answer(r'old \boxed{2}; new \boxed{\frac{1}{2}}'),r'\frac{1}{2}')
    def test_bad_last_box_never_falls_back(self):
        for text in [r'\boxed{2} then \boxed{',r'\boxed{2} then \boxed{}','answer: 2']:
            self.assertIsNone(last_boxed_answer(text))
    def test_math_equivalence(self):
        for a,b in [(r'\frac{1}{2}','0.5'),(r'\sqrt{4}','2'),('x=2','2'),(r'\{1,2\}',r'\{2,1\}')]:
            self.assertTrue(math_equivalent(a,b),(a,b))
    def test_math_inequality(self):
        self.assertFalse(math_equivalent('2','3'));self.assertFalse(math_equivalent(None,None))
        self.assertFalse(math_equivalent('x+1','y+1'))
    def test_missing_group(self):
        self.assertEqual(math_groups([None,'',float('nan')]),[NO_ANSWER]*3)
    def test_groups_order_independent(self):
        values=['0.5',r'\frac{1}{2}','2','',None];groups=math_groups(values)
        self.assertEqual(groups[0],groups[1]);self.assertNotEqual(groups[0],groups[2])
        self.assertEqual(math_groups(list(reversed(values))),list(reversed(groups)))
    def test_choice_rules(self):
        meta={'question_type':'multi_choice','answer_type':'text','choices':['red','blue','green']}
        for value in ['B','(b) blue','blue','blu']:self.assertEqual(mathvista_answer(value,meta),'B')
    def test_empty_choice_not_guessed(self):
        meta={'question_type':'multi_choice','answer_type':'text','choices':['red','blue']}
        for value in [None,'','  ',float('nan')]:self.assertEqual(mathvista_answer(value,meta),NO_ANSWER)
    def test_integer_float_and_list(self):
        integer={'question_type':'free_form','answer_type':'integer'}
        self.assertEqual(mathvista_answer('3.0',integer),'3');self.assertEqual(mathvista_answer('3.9',integer),'3')
        meta={'question_type':'free_form','answer_type':'float','precision':2}
        self.assertEqual(mathvista_answer('1.236',meta),'1.24')
        self.assertTrue(mathvista_correct('1.236','1.24',meta))
        self.assertEqual(mathvista_answer('NaN',meta),NO_ANSWER)
        meta={'question_type':'free_form','answer_type':'list'}
        self.assertEqual(mathvista_answer('[1, 2]',meta),'[1, 2]')
    def test_choice_uses_no_gold(self):
        meta={'question_type':'multi_choice','answer_type':'text','choices':['red','blue'],'answer':'red'}
        first=mathvista_answer('blue',meta);meta['answer']='NOT_THE_GOLD'
        self.assertEqual(first,mathvista_answer('blue',meta))
    def test_reference_literal_parenthesized_letter(self):
        meta={'question_type':'multi_choice','answer_type':'text','choices':['(c)','(a)','(e)','(d)','(b)']}
        self.assertEqual(mathvista_reference('(b)',meta),'E')
        self.assertEqual(mathvista_reference('E',meta),'E')
        self.assertEqual(mathvista_answer('(b)',meta),'B')
        self.assertTrue(mathvista_correct('E','(b)',meta))
        self.assertFalse(mathvista_correct('B','(b)',meta))

if __name__=='__main__':unittest.main()
