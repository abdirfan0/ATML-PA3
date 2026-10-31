"""Frozen PA3 Task 3 answer parsing, gold-free grouping, and grading.

MathVista normalization follows lupantech/MathVista evaluation/calculate_score.py
at c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb. Its option-text result is
represented by an option letter as required by PA3. Missing/invalid extracted
answers are not guessed from full responses. No model or GPU is used.
"""
from __future__ import annotations
from functools import lru_cache
import math,re
from math_verify import parse, verify, LatexExtractionConfig

VERSION='pa3-task3-answer-scorer-v2'
NO_ANSWER='NO_ANSWER'
MATHVISTA_REVISION='c5acd37d75a8c8f9756cb66c78fe2e54fa2c31fb'

def clean(value):
    if value is None or isinstance(value,float) and math.isnan(value):return None
    value=str(value).strip()
    return value or None

def last_boxed_answer(text):
    """Last literal box only; no fallback if that box is absent/empty/unclosed."""
    marker='\\boxed{';start=text.rfind(marker)
    if start<0:return None
    start+=len(marker);depth=1
    for pos in range(start,len(text)):
        if text[pos]=='{':depth+=1
        elif text[pos]=='}':
            depth-=1
            if depth==0:return clean(text[start:pos])
    return None

@lru_cache(maxsize=50000)
def parse_math(value):
    value=clean(value)
    if value is None:return ()
    if value.startswith('$$') and value.endswith('$$'):value=value[2:-2]
    elif value.startswith('$') and value.endswith('$'):value=value[1:-1]
    elif value.startswith('\\(') and value.endswith('\\)'):value=value[2:-2]
    elif value.startswith('\\[') and value.endswith('\\]'):value=value[2:-2]
    return tuple(parse('$'+value+'$',extraction_config=[LatexExtractionConfig()],
                       fallback_mode='no_fallback',extraction_mode='first_match',parsing_timeout=5))

@lru_cache(maxsize=200000)
def _equivalent_sorted(a,b):
    if a==b:return True
    left=parse_math(a);right=parse_math(b)
    if not left or not right:return False
    # A symmetric predicate is used for both grouping and grading, with strict
    # variable matching. Never use a gold answer to canonicalize candidates.
    return bool(verify(list(left),list(right),strict=True,timeout_seconds=5)
                and verify(list(right),list(left),strict=True,timeout_seconds=5))

def math_equivalent(a,b):
    a=clean(a);b=clean(b)
    if a is None or b is None:return False
    return _equivalent_sorted(*sorted((a,b)))

def math_groups(extracted_answers):
    """Stable groups from predictions only; independent of reveal order and gold.

    Sort unique extracted strings, then admit an answer to the first group only
    when it matches every existing member. This deterministic all-member rule
    avoids merging through a nontransitive numeric-tolerance bridge. The group
    number is just an identifier, never a vote/score tie breaker.
    """
    values=[clean(a) for a in extracted_answers];groups=[];lookup={}
    for value in sorted(set(values)-{None}):
        for index,members in enumerate(groups):
            if all(math_equivalent(value,other) for other in members):break
        else:index=len(groups);groups.append([])
        groups[index].append(value);lookup[value]=f'MATH_GROUP_{index:02d}'
    return [NO_ANSWER if v is None else lookup[v] for v in values]

def edit_distance(a,b):
    """Levenshtein distance; no extra binary dependency for option matching."""
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def mathvista_answer(extraction,metadata):
    """Released type/precision/choice rules, without reading metadata['answer'].

    Missing predictions stay NO_ANSWER before any closest-option mapping.
    Closest-option ties use the first listed choice, as in the released scorer.
    """
    value=clean(extraction)
    if value is None:return NO_ANSWER
    question_type=metadata['question_type'];answer_type=metadata['answer_type']
    if question_type=='multi_choice':
        choices=metadata['choices'];assert choices and all(isinstance(x,str) for x in choices)
        letters=[chr(65+i) for i in range(len(choices))]
        match=re.findall(r'\(([a-zA-Z])\)',value)
        if match:value=match[0].upper()
        if value in letters:return value
        distance=[edit_distance(value,c) for c in choices]
        return letters[distance.index(min(distance))]
    try:
        if answer_type=='integer':return str(int(float(value)))
        if answer_type=='float':
            number=float(value)
            if not math.isfinite(number):return NO_ANSWER
            return str(round(number,int(metadata['precision'])))
        if answer_type=='list':return str(value)
    except (ValueError,OverflowError,TypeError):return NO_ANSWER
    raise ValueError(f'Unsupported MathVista type: {question_type}/{answer_type}')

def mathvista_reference(gold,metadata):
    """Decode a reference field, whose exact choice text is authoritative.

    For example, question 168's reference '(b)' is choice E, not option B.
    Predictions still use the released extractor-normalization rules above.
    Reference fields may instead already contain an option letter.
    """
    value=clean(gold)
    if value is None:return NO_ANSWER
    if metadata['question_type']=='multi_choice' and value in metadata['choices']:
        return chr(65+metadata['choices'].index(value))
    return mathvista_answer(gold,metadata)

def mathvista_correct(extraction,gold,metadata):
    candidate=mathvista_answer(extraction,metadata)
    return candidate!=NO_ANSWER and candidate==mathvista_reference(gold,metadata)
