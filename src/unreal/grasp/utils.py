

import csv
import math
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Union

import numpy as np
import tqdm

from .attribute import CustomAttribute, _default_attr2text
from .augmented_text import AugmentedText
from .default_attributes import DEFAULT_ATTRIBUTES

if TYPE_CHECKING:
    from .pattern import Pattern


def entropy_binary(count_pos: int, count_neg: int) -> float:
    n_total = count_pos + count_neg
    if n_total == 0:
        return None
    p_pos = count_pos / n_total
    p_neg = count_neg / n_total
    if p_pos == 0 or p_neg == 0:
        return 0.0
    else:
        return -(p_pos * math.log2(p_pos) + p_neg * math.log2(p_neg))

# ========== Simplify pattern set ==========
def _is_specialized(p1:'Pattern', p2:'Pattern') -> bool: 
    """Return True if p1 is a specialization of p2 including the case where p1_labels == p2_labels"""
    p1_labels = [True if v else False for v in p1.pos_example_labels + p1.neg_example_labels]
    p2_labels = [True if v else False for v in p2.pos_example_labels + p2.neg_example_labels]
    for idx, v in enumerate(p1_labels):
        if v:
            if not p2_labels[idx]:
                return False
    return True

def remove_specialized_patterns(patterns: List['Pattern'],
                                mode: int = 1,
                                metric: Optional[Union[Callable, str]] = None
    ) -> List['Pattern']:
    # Mode = 1: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 and metric of p2 is lower than p1
    # Mode = 2: Remove pattern p2 if there exists p1 in the patterns set such that p2 is a specialization of p1 regardless of the metric value of p1 and p2
    
    # Find the metric function
    if callable(metric):
        metric_func = metric
    elif metric is None:
        metric_func = lambda x: x.metric
    elif metric == 'global':
        metric_func = lambda x: x.global_information_gain 
    elif metric == 'local':
        metric_func = lambda x: x.information_gain 
    elif metric == 'relative':
        metric_func = lambda x: x.relative_information_gain
    elif metric.startswith('F_'):
        try:
            beta = float(metric[2:])
        except:
            assert False, f"Invalid metric: {metric}"
        metric_func = lambda x: (1+beta**2) * (x.precision * x.recall) / ((x.precision*beta**2) + x.recall) if (x.precision is not None) else 0.0
    else:
        assert False, f"Invalid metric {metric}"

    # Remove patterns
    the_list = patterns
    to_remove = set() 
    for idx1, p in enumerate(the_list):
        for idx2, q in enumerate(the_list):
            if idx1 == idx2:
                continue
            if mode == 1 and metric_func(p) >= metric_func(q) and _is_specialized(q, p):
                to_remove.add(idx2)
            elif mode == 2 and _is_specialized(q, p):
                to_remove.add(idx2)
    return [p for idx, p in enumerate(the_list) if idx not in to_remove]

# ========== Translation ==========
def attr2score(attr: str) -> float:
    # Return a score used to rank the attribute during translation
    attr_type = attr.split(':')[0]
    if attr_type == 'SPACY':
        attr_type = attr.split('-')[0]

    SCORES = {
        'TEXT': 0,
        'LEMMA': 5,
        'SPACY:NER': 10,
        'SPACY:POS': 20,
        'SENTIMENT': 30,        
        'HYPERNYM': 40,
        'SPACY:DEP': 50,
    }

    return SCORES.get(attr_type, 100)


def _attr2text(attr: str, p: 'Pattern', is_complement: bool = False) -> str:
    attr_type = attr.split(':')[0]
    if attr_type in p.all_attributes_dict:
        return p.all_attributes_dict[attr_type].translation_function(attr, is_complement)
    else:
        return _default_attr2text(attr, is_complement)

def _slot2text(s: Set[str], p: 'Pattern') -> str:
    s = list(s)
    s.sort(key = lambda x: attr2score(x))
    ans = ''
    for aidx, a in enumerate(s):
        if aidx == 0:
            ans += _attr2text(a, p, is_complement = False)
        elif aidx == 1:
            ans += f' which is also {_attr2text(a, p, is_complement = True)}'
        elif aidx == len(s)-1:
            if aidx == 2:
                ans += f' and {_attr2text(a, p, is_complement = True)}'
            else:
                ans += f', and {_attr2text(a, p, is_complement = True)}'
        else:
            ans += f', {_attr2text(a, p, is_complement = True)}'
    return ans


def pattern2text(p: 'Pattern') -> str:
    has_gap = p.window_size > len(p.pattern)
    connector = 'immediately' if not has_gap else 'closely'
    ans = ''
    for sidx, slot in enumerate(p.pattern):
        if sidx == 0:
            ans += _slot2text(slot, p)
        elif sidx == 1:
            ans += f', {connector} followed by {_slot2text(slot, p)}'
        else:
            # ans += f', and then {connector} followed by {slot2text(slot, p)}'
            ans += f', and then by {_slot2text(slot, p)}'
    return ans[0].upper() + ans[1:]

# ========== Feature extraction ==========

def extract_features(texts: List[str], 
                     patterns: List['Pattern'],
                     polarity: bool = False, # If True, features will be [1, 0, -1] where 1 means positive rule match and -1 means negative rule match
                     include_standard: List[str] = DEFAULT_ATTRIBUTES, #['TEXT', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'], 
                     include_custom: List[CustomAttribute] = []): # Return numpy array [len(texts), len(patterns)]
    if polarity:
        for p in patterns:
            assert p.support_class in ["Positive", "Negative"], "Cannot set Polarity = True as some of the patterns do not have a valid support_class"

    ans = []
    for t in tqdm(texts):
        vector = []
        augtext = AugmentedText(t, None, include_standard, include_custom)
        for p in patterns:
            if p.is_match(augtext)[0]:
                if polarity:
                    value = 1 if p.support_class == 'Positive' else -1
                    vector.append(value)
                else:
                    vector.append(1)
            else:
                vector.append(0)
        ans.append(vector)
    return np.array(ans)            

# ========== Export ==========
def patterns2csv(patterns: List['Pattern'],
                 filepath: str):

    data = []
    for idx, p in enumerate(patterns):
        data.append({
            'index': idx,
            'pattern': p.get_pattern_id(),
            'meaning': pattern2text(p),
            '#pos': p.pos_example_labels.count(True), 
            '#neg': p.neg_example_labels.count(True),
            'score': p.metric,
            'coverage': p.coverage, 
            'precision': p.precision, 
            'recall': p.recall, 
            'F1': (2 * p.precision * p.recall) / (p.precision + p.recall) if (p.precision is not None) else None
            })
    with open(filepath, 'w', newline='\n', encoding='utf-8') as csv_file:
        fieldnames = ['index', 'pattern', 'meaning', '#pos', '#neg', 'score', 'coverage', 'precision', 'recall', 'F1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f'Successful: {len(data)} rules have been written to {filepath}')