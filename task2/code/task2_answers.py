"""Consistent VQA answer normalization and local leave-one-out consensus scoring.

Normalization adapted from GT-Vision-Lab/VQA vqaEval.py, author aagrawal,
commit a013f0043c1e2cdc995922dfe257f7149aa9af06.
Upstream SHA256: f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8.
https://github.com/GT-Vision-Lab/VQA/blob/a013f0043c1e2cdc995922dfe257f7149aa9af06/PythonEvaluationTools/vqaEvaluation/vqaEval.py

Unlike the old evaluator's conditional call, normalization is always applied to
predictions and every reference, as required by this assignment. Keep raw answers.
"""
import re
VERSION = 'pa3-task2-vqa-normalization-v1'
CONTRACTIONS = {'aint': "ain't", 'arent': "aren't", 'cant': "can't", 'couldve': "could've", 'couldnt': "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've", 'didnt': "didn't", 'doesnt': "doesn't", 'dont': "don't", 'hadnt': "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", 'hasnt': "hasn't", 'havent': "haven't", 'hed': "he'd", "hed've": "he'd've", "he'dve": "he'd've", 'hes': "he's", 'howd': "how'd", 'howll': "how'll", 'hows': "how's", "Id've": "I'd've", "I'dve": "I'd've", 'Im': "I'm", 'Ive': "I've", 'isnt': "isn't", 'itd': "it'd", "itd've": "it'd've", "it'dve": "it'd've", 'itll': "it'll", "let's": "let's", 'maam': "ma'am", 'mightnt': "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", 'mightve': "might've", 'mustnt': "mustn't", 'mustve': "must've", 'neednt': "needn't", 'notve': "not've", 'oclock': "o'clock", 'oughtnt': "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", 'shant': "shan't", "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", 'shouldve': "should've", 'shouldnt': "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": 'somebodyd', "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", 'somebodyll': "somebody'll", 'somebodys': "somebody's", 'someoned': "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", 'someonell': "someone'll", 'someones': "someone's", 'somethingd': "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", 'somethingll': "something'll", 'thats': "that's", 'thered': "there'd", "thered've": "there'd've", "there'dve": "there'd've", 'therere': "there're", 'theres': "there's", 'theyd': "they'd", "theyd've": "they'd've", "they'dve": "they'd've", 'theyll': "they'll", 'theyre': "they're", 'theyve': "they've", 'twas': "'twas", 'wasnt': "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 'weve': "we've", 'werent': "weren't", 'whatll': "what'll", 'whatre': "what're", 'whats': "what's", 'whatve': "what've", 'whens': "when's", 'whered': "where'd", 'wheres': "where's", 'whereve': "where've", 'whod': "who'd", "whod've": "who'd've", "who'dve": "who'd've", 'wholl': "who'll", 'whos': "who's", 'whove': "who've", 'whyll': "why'll", 'whyre': "why're", 'whys': "why's", 'wont': "won't", 'wouldve': "would've", 'wouldnt': "wouldn't", "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", 'yall': "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", 'youd': "you'd", "youd've": "you'd've", "you'dve": "you'd've", 'youll': "you'll", 'youre': "you're", 'youve': "you've"}

NUMBER_WORDS = dict(zip(['none','zero','one','two','three','four','five','six','seven','eight','nine','ten'], ['0','0','1','2','3','4','5','6','7','8','9','10']))
ARTICLES = {'a','an','the'}
PUNCTUATION = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
PERIOD = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA = re.compile(r"(\d)(,)(\d)")

def normalize_answer(value):
    original = str(value).replace('\n',' ').replace('\t',' ').strip()
    text = original
    for punctuation in PUNCTUATION:
        if punctuation+' ' in original or ' '+punctuation in original or COMMA.search(original):
            text = text.replace(punctuation,'')
        else:
            text = text.replace(punctuation,' ')
    text = PERIOD.sub('', text)
    words = [NUMBER_WORDS.get(word,word) for word in text.lower().split()]
    return ' '.join(CONTRACTIONS.get(word,word) for word in words if word not in ARTICLES)

def vqa_soft_accuracy(prediction, references):
    assert len(references)==10
    predicted = normalize_answer(prediction)
    if not predicted:return 0.0
    answers = [normalize_answer(answer) for answer in references]
    matches = sum(answer==predicted for answer in answers)
    return sum(min(1.0,(matches-int(answer==predicted))/3.0) for answer in answers)/10.0
