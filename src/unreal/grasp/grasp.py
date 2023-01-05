import json
from collections import Counter
from typing import Callable, List, Optional, Sequence, Union

from tqdm import tqdm

from .attribute import CustomAttribute
from .augmented_text import AugmentedText
from .default_attributes import (DEFAULT_ATTRIBUTES, HypernymAttribute,
                                 LemmaAttribute, SentimentAttribute,
                                 SpacyAttribute, TextAttribute)
from .pattern import Pattern
from .utils import pattern2text, patterns2csv


class GrASP():
    
    def __init__(self,
                 min_freq_threshold: float = 0.005, # t1
                 correlation_threshold: float = 0.5, # t2
                 alphabet_size: int = 100, # k1
                 num_patterns: int = 100, # k2
                 max_len: int = 5, # maxLen
                 window_size: Optional[int] = 10, # w
                 gaps_allowed: Optional[int] = None, # If gaps allowed is not None, it overrules the window size
                 gain_criteria: Union[str, Callable[[Pattern], float]] = 'global', # 'F_beta', 'global', 'local', or 'relative'
                 min_coverage_threshold: Optional[float] = None, # float: Proportion of examples
                 print_examples: Union[int, Sequence[int]] = 2, 
                 include_standard: List[str] = DEFAULT_ATTRIBUTES, # Available options include ['TEXT', 'LEMMA', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT']
                 include_custom: List[CustomAttribute] = []) -> None:
        
        # Standard hyperparameters: minimal frequency threshold t1=0.005, correlation threshold t2 =0.5, size of the alphabet k1 =100, number of patterns in the output k2 = 100, maximal pattern length maxLen=5, and window size w=10
        
        self.min_freq_threshold = min_freq_threshold
        self.correlation_threshold = correlation_threshold
        self.alphabet_size = alphabet_size
        self.num_patterns = num_patterns
        self.max_len = max_len
        
        # Gaps VS Window size
        if gaps_allowed is not None and gaps_allowed >= 0:
            self.gaps_allowed = gaps_allowed
            self.window_size = None
        elif gaps_allowed is not None and gaps_allowed < 0:
            raise Exception(f'Gaps allowed should not be less than 0, {gaps_allowed} given')
        else:
            self.gaps_allowed = None
            self.window_size = window_size
        
        # Gain criteria
        assert callable(gain_criteria) or (gain_criteria in ['global', 'local', 'relative']) or\
            (gain_criteria.startswith('F_')), f"Gain criterial must be callable or 'F_beta' (beta is a float number), 'global', 'local', or 'relative', but {gain_criteria} is given"
        self.gain_criteria = gain_criteria
        
        if callable(self.gain_criteria):
            self.sort_key = self.gain_criteria
        elif self.gain_criteria == 'global':
            self.sort_key = lambda x: x.global_information_gain 
        elif self.gain_criteria == 'local':
            self.sort_key = lambda x: x.information_gain 
        elif self.gain_criteria == 'relative':
            self.sort_key = lambda x: x.relative_information_gain
        elif self.gain_criteria.startswith('F_'):
            try:
                beta = float(self.gain_criteria[2:])
            except:
                assert False, f"Invalid gain criteria: {self.gain_criteria}"
            self.sort_key = lambda x: (1+beta**2) * (x.precision * x.recall) / ((x.precision*beta**2) + x.recall) if (x.precision is not None) else 0.0
        else:
            assert False, f"Invalid gain criteria {self.gain_criteria}"

        # Minimum coverage
        self.min_coverage_threshold = min_coverage_threshold
        
        self.include_standard = include_standard
        self.include_custom = include_custom

        self.all_attributes_dict = dict()
        # Standard attributes
        if 'TEXT' in self.include_standard:
            self.all_attributes_dict['TEXT'] = TextAttribute
        if 'LEMMA' in self.include_standard:
            self.all_attributes_dict['LEMMA'] = LemmaAttribute
        if set(include_standard).intersection(set(['POS', 'DEP', 'NER'])):
            self.all_attributes_dict['SPACY'] = SpacyAttribute
        if 'HYPERNYM' in self.include_standard:
            self.all_attributes_dict['HYPERNYM'] = HypernymAttribute
        if 'SENTIMENT' in self.include_standard:
            self.all_attributes_dict['SENTIMENT'] = SentimentAttribute
        for attr_class in self.include_custom:
            self.all_attributes_dict[attr_class.name] = attr_class
        
        # For printing patterns
        if isinstance(print_examples, list) or isinstance(print_examples, tuple):
            assert len(print_examples) == 2 and all(i>=0 for i in print_examples)
            self.print_examples = print_examples
        else:
            assert print_examples >= 0
            self.print_examples = (print_examples, print_examples)
            
        self.candidate_alphabet = None # After removing non-frequent attributes
        self.alphabet = None
        self.positives = None
        self.negatives = None
    
    def _remove_nonfrequent_attributes(self) -> List[str]:
        assert self.pos_augmented is not None
        assert self.neg_augmented is not None
        
        all_augmented = self.pos_augmented + self.neg_augmented
        all_attributes = []
        for augtext in all_augmented:
            all_attributes += list(augtext.all_unique_features)
        the_counter = Counter(all_attributes)
        min_freq = self.min_freq_threshold * len(all_augmented)
        candidate_alphabet = []
        for the_attr, freq in the_counter.most_common():
            if freq < min_freq:
                break
            else:
                candidate_alphabet.append(the_attr)
            
        # Remove non-frequent attributes from all_features of augmented texts
        for augtext in self.pos_augmented:
            augtext.keep_only_features(candidate_alphabet)
        for augtext in self.neg_augmented:
            augtext.keep_only_features(candidate_alphabet)
            
        return candidate_alphabet
    
    def _find_top_k_patterns(self, patterns: List[Pattern], k: int, use_coverage_threshold: bool = True) -> List[Pattern]:
        patterns.sort(key = lambda x: x.metric, reverse = True) # Sort by information gain descendingly
        
        ans = []
        for p in patterns:
            # Do not keep a pattern if it has too low coverage
            if use_coverage_threshold and self.min_coverage_threshold is not None and p.coverage < self.min_coverage_threshold:
                continue
                
            # Do not select if no gain at all
            if p.metric <= 0:
                continue
                
            is_correlated = False
            for a in ans:
                if p.global_normalized_mutual_information(a) > self.correlation_threshold:
                    is_correlated = True
                    break
            if not is_correlated:
                ans.append(p)
                if len(ans) % int(k/10) == 0:
                    print(f"Finding top k: {len(ans)} / {k}")
                if len(ans) == k:
                    break     
        return ans
        
    def _select_alphabet_remove_others(self) -> [List[str], List[Pattern]]:
        assert self.candidate_alphabet is not None
        
        w_size = self.gaps_allowed if self.gaps_allowed is not None else self.window_size
        self.root_pattern = Pattern([], w_size, None, self)
        
        # Find information gain of each candidate
        canndidate_alphabet_patterns = []
        for c in tqdm(self.candidate_alphabet):
            w_size = self.gaps_allowed + 1 if self.gaps_allowed is not None else self.window_size
            the_candidate = Pattern([set([c])], w_size, self.root_pattern, self)
            canndidate_alphabet_patterns.append(the_candidate)
        
        # Find top k1 attributes to be the alphabet while removing correlated attributes
        alphabet_patterns = self._find_top_k_patterns(canndidate_alphabet_patterns, k = self.alphabet_size)
        alphabet = [list(p.pattern[0])[0] for p in alphabet_patterns]
        
        # Remove non-alphabet attributes from all_features of augmented texts
        for augtext in self.pos_augmented:
            augtext.keep_only_features(alphabet)
        for augtext in self.neg_augmented:
            augtext.keep_only_features(alphabet)
            
        return alphabet, alphabet_patterns
        
    
    def fit_transform(self, positives: List[Union[str, AugmentedText]], negatives: List[Union[str, AugmentedText]]) -> List[Pattern]:
        # Fit = Find a set of alphabet
        # 1. Create augmented texts
        print("Step 1: Create augmented texts")
        self.positives, self.negatives, self.pos_augmented, self.neg_augmented = [], [], [], []
        for is_neg, input_list in enumerate([positives, negatives]):
            for t in tqdm(input_list):
                if isinstance(t, str):
                    if not is_neg:
                        self.pos_augmented.append(AugmentedText(t, True, self.include_standard, self.include_custom))
                        self.positives.append(t)
                    else:
                        self.neg_augmented.append(AugmentedText(t, False, self.include_standard, self.include_custom))
                        self.negatives.append(t)
                else: # t is an AugmentedText
                    assert self.include_standard == t.include_standard
                    assert self.include_custom == t.include_custom
                    if not is_neg:
                        self.pos_augmented.append(t)
                        self.positives.append(t.text)
                    else:
                        self.neg_augmented.append(t)
                        self.negatives.append(t.text)
        
        # 2. Find frequent attributes (according to min_freq_threshold)
        print("Step 2: Find frequent attributes")
        self.candidate_alphabet = self._remove_nonfrequent_attributes()
        print(f"Total number of candidate alphabet = {len(self.candidate_alphabet)}, such as {self.candidate_alphabet[:5]}")
        
        # 3. Find alphabet set (according to alphabet_size and correlation_threshold)
        print("Step 3: Find alphabet set")
        self.alphabet, self.seed_patterns = self._select_alphabet_remove_others()
        print(f"Total number of alphabet = {len(self.alphabet)}")
        #print(self.alphabet)
        #print("Example of patterns")
        #for p in self.seed_patterns[:5]:
        #    print(p)
        
        # 4. Grow and selecct patterns
        print("Step 4: Grow patterns")
        current_patterns = list(self.seed_patterns)
        last = list(current_patterns)
        visited = set([p.get_pattern_id() for p in current_patterns])
        for length in range(2, self.max_len+1):
            new_candidates = []
            for p in tqdm(last):
                for a in self.alphabet:
                    # Grow right
                    grow_right_candidate = p.pattern + [set([a])]
                    if Pattern.pattern_list2str(grow_right_candidate) not in visited:
                        w_size = self.gaps_allowed + len(grow_right_candidate) if self.gaps_allowed is not None else self.window_size
                        new_candidates.append(Pattern(grow_right_candidate, w_size, p, self))
                        visited.add(Pattern.pattern_list2str(grow_right_candidate))
                    # Grow inside
                    grow_inside_candidate = p.pattern[:-1] + [set([a]).union(p.pattern[-1])]
                    if Pattern.pattern_list2str(grow_inside_candidate) not in visited:
                        w_size = self.gaps_allowed + len(grow_inside_candidate) if self.gaps_allowed is not None else self.window_size
                        new_candidates.append(Pattern(grow_inside_candidate, w_size, p, self))
                        visited.add(Pattern.pattern_list2str(grow_inside_candidate))
            print(f'Length {length} / {self.max_len}; New candidates = {len(new_candidates)}')
            if len(new_candidates) == 0:
                break
            current_patterns = self._find_top_k_patterns(current_patterns + new_candidates, k = self.num_patterns)
            last = [p for p in current_patterns if p in new_candidates] # last is recently added patterns
            #print("Example of current patterns")
            #for p in current_patterns[:5]:
            #    print(p)
        self.extracted_patterns = current_patterns
        return current_patterns

    def to_json(self, filepath: str, comment: str = '', patterns: Optional[List[Pattern]] = None):
        # Rules
        if patterns is None:
            patterns = self.extracted_patterns

        rules = []
        for idx, p in enumerate(patterns):
            rules.append({
                'index': idx,
                'pattern': p.get_pattern_id(),
                'meaning': pattern2text(p),
                'class': 'pos' if p.pos_example_labels.count(True) > p.neg_example_labels.count(True) else 'neg',
                '#pos': p.pos_example_labels.count(True), 
                '#neg': p.neg_example_labels.count(True),
                'score': p.metric,
                'coverage': p.coverage, 
                'precision': p.precision, 
                'recall': p.recall, 
                'F1': (2 * p.precision * p.recall) / (p.precision + p.recall) if (p.precision is not None) else None,
                'pos_example_labels': [False if v is None else v for v in p.pos_example_labels], 
                'neg_example_labels': [False if v is None else v for v in p.neg_example_labels],
                })

        # Configuration
        configuration = {
            'min_freq_threshold': self.min_freq_threshold,
            'correlation_threshold': self.correlation_threshold,
            'alphabet_size': self.alphabet_size,
            'num_patterns': self.num_patterns,
            'max_len': self.max_len,
            'window_size': self.window_size,
            'gaps_allowed': self.gaps_allowed,
            'gain_criteria': str(self.gain_criteria),
            'min_coverage_threshold': self.min_coverage_threshold,
            'include_standard': self.include_standard, 
            'include_custom': [attr.name for attr in self.include_custom],
            'comment': comment
        } 

        # Dataset
        # Positive examples and negative examples
        
        for c in ['pos', 'neg']:
            the_exs = [{'idx': eidx, 'text': t.text, 'tokens': t.tokens, 'label': c, 'rules': [[] for x in t.tokens], 'class': [[] for x in t.tokens]} for eidx, t in enumerate(eval(f'self.{c}_augmented'))]
            for idx, p in enumerate(tqdm(patterns)):
                assert len(eval(f'p.{c}_example_labels')) == len(eval(f'self.{c}_augmented'))
                for eidx, v in enumerate(eval(f'p.{c}_example_labels')):
                    if v: # match
                        is_match, match_indices = p.is_match(eval(f'self.{c}_augmented[eidx]'))
                        assert is_match and isinstance(match_indices, list)
                        for match in match_indices:
                            the_exs[eidx]['rules'][match].append(idx)
                            if p.support_class == 'Positive':
                                the_exs[eidx]['class'][match].append(1)
                            else:
                                the_exs[eidx]['class'][match].append(-1)
            if c == 'pos': 
                pos_exs = the_exs
            else:
                neg_exs = the_exs

        dataset = {
            'info': {'total': len(self.pos_augmented) + len(self.neg_augmented), 
                     '#pos': len(self.pos_augmented),
                     '#neg': len(self.neg_augmented)
                    },
            'pos_exs': pos_exs,
            'neg_exs': neg_exs
        }

        main_obj = {
            'configuration': configuration,
            'alphabet': self.alphabet,
            'rules': rules,
            'dataset': dataset
        }
        
        with open(filepath, "w") as f: # Write a JSON file
            json.dump(main_obj, f)

        print(f'Successfully dump the results to {filepath}')

    def to_csv(self, filepath: str, patterns: Optional[List[Pattern]] = None):
        if patterns is None:
            patterns = self.extracted_patterns
        patterns2csv(patterns, filepath)


    