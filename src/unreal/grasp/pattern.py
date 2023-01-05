import random
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sklearn.metrics import normalized_mutual_info_score
from termcolor import colored

from .augmented_text import AugmentedText
from .utils import entropy_binary

if TYPE_CHECKING:
    from .grasp import GrASP

# ========== Patterns ==========

class Pattern():
    
    def __init__(self,
                 pattern: List[Set[str]],
                 window_size: Optional[int], # None means no window size (can match an arbitrary length)
                 parent: Optional['Pattern']=None,
                 grasp: Optional['GrASP']=None,
                ) -> None:
        
        self.pattern = pattern
        self.parent = parent
        # self.grasp = grasp
        self.window_size = window_size
        self.print_examples = grasp.print_examples
        self.pos_augmented = grasp.pos_augmented
        self.neg_augmented = grasp.neg_augmented
        self.sort_key = grasp.sort_key
        self.gain_criteria = grasp.gain_criteria
        self.all_attributes_dict = grasp.all_attributes_dict
        
        # ----- Check pattern matching
        if self.pattern == []: # Root node matches everything
            assert self.parent is None, "Root node cannot have any parent"
            self.pos_example_labels = [True for item in self.pos_augmented]
            self.neg_example_labels = [True for item in self.neg_augmented]
             # True (Match parent and here) / False (Match parent but not here) / None (Not match in parent)
        else:
            if self.parent is None:
                self.pos_example_labels = [self.is_match(augtext)[0] for augtext in self.pos_augmented]
                self.neg_example_labels = [self.is_match(augtext)[0] for augtext in self.neg_augmented]
            else:
                self.pos_example_labels = [self.is_match(augtext)[0] if val else None for augtext, val in zip(self.pos_augmented, self.parent.pos_example_labels)]
                self.neg_example_labels = [self.is_match(augtext)[0] if val else None for augtext, val in zip(self.neg_augmented, self.parent.neg_example_labels)]
        
        # ----- Count match and notmatch
        pos_match, neg_match = self.pos_example_labels.count(True), self.neg_example_labels.count(True)
        pos_notmatch, neg_notmatch = self.pos_example_labels.count(False), self.neg_example_labels.count(False)
        pos_none, neg_none = self.pos_example_labels.count(None), self.neg_example_labels.count(None)
        
        self.num_total_match = pos_match + neg_match
        self.num_total_notmatch = pos_notmatch + neg_notmatch
        self.num_total_all = self.num_total_match + self.num_total_notmatch
        if self.parent is not None:
            assert self.num_total_all == self.parent.num_total_match
        self.prob_match = self.num_total_match / self.num_total_all
        self.prob_notmatch = self.num_total_notmatch / self.num_total_all
           
        # ----- Calculate entropy and information gain
        self.entropy_match = entropy_binary(pos_match, neg_match)
        self.entropy_notmatch = entropy_binary(pos_notmatch, neg_notmatch)
        if self.parent is None:
            self.information_gain = None
            self.relative_information_gain = None
        else:
            # print(self.parent.entropy_match, self.prob_match, self.entropy_match, self.prob_notmatch, self.entropy_notmatch)
            self.information_gain = self.parent.entropy_match
            if self.prob_match > 0:
                self.information_gain -= self.prob_match * self.entropy_match
            if self.prob_notmatch > 0:
                self.information_gain -= self.prob_notmatch * self.entropy_notmatch
            self.relative_information_gain = self.information_gain / self.parent.entropy_match if self.parent.entropy_match != 0 else 0
                
        # ----- Calculate global weighted entropy and information gain (Consider None as False)
        self.root_entropy = entropy_binary(len(self.pos_augmented), len(self.neg_augmented))
        self.global_entropy_match = self.entropy_match
        self.global_entropy_notmatch = entropy_binary(pos_notmatch+pos_none, neg_notmatch+neg_none)
        self.global_num_total_all = len(self.pos_augmented) + len(self.neg_augmented)
        self.global_prob_match = self.num_total_match / self.global_num_total_all
        self.global_prob_notmatch = 1 - self.global_prob_match
        if self.parent is None:
            self.global_information_gain = None
        else:
            self.global_information_gain = self.root_entropy
            if self.global_prob_match > 0:
                self.global_information_gain -= self.global_prob_match * self.global_entropy_match
            if self.global_prob_notmatch > 0:
                self.global_information_gain -= self.global_prob_notmatch * self.global_entropy_notmatch
        
        # ----- Calculate precision, recall, and coverage of the pattern
        self.support_class = "Positive" if pos_match >= neg_match else "Negative"
        self.precision = max(pos_match, neg_match) / self.num_total_match if self.num_total_match > 0 else None
        self.recall = pos_match / len(self.pos_example_labels) if pos_match >= neg_match else neg_match / len(self.neg_example_labels)
        self.coverage = self.global_prob_match
        try:
            self.metric = self.sort_key(self)
        except:
            print(self.precision, self.recall)
            raise Exception()
    
    def normalized_mutual_information(self, another_pattern: 'Pattern') -> float:
        labels_1 = self.get_all_labels()
        labels_2 = another_pattern.get_all_labels()
        assert len(labels_1) == len(labels_2)
        assert self.parent == another_pattern.parent
        
        f_labels_1, f_labels_2 = [], []
        for l1, l2 in zip(labels_1, labels_2):
            assert (l1 is None) == (l2 is None)
            if l1 is not None:
                f_labels_1.append(int(l1))
                f_labels_2.append(int(l2))
        return normalized_mutual_info_score(f_labels_1, f_labels_2)
    
    def global_normalized_mutual_information(self, another_pattern: 'Pattern') -> float:
        labels_1 = self.global_get_all_labels()
        labels_2 = another_pattern.global_get_all_labels()
        assert len(labels_1) == len(labels_2)
        return normalized_mutual_info_score(labels_1, labels_2)
        
    @staticmethod
    def _is_token_match(pattern_attributes: Set[str], token_attributes: Set[str]) -> bool:
        return pattern_attributes.issubset(token_attributes)
    
        
    def _is_match_recursive(self, pattern: List[Set[str]], attribute_list: List[Set[str]], start: int, end: Optional[int]) -> List[int]:
        # Base case
        if pattern == []:
            return []
        if start == len(attribute_list):
            return False
        
        # Recursive case
        if end is None:
            stop_match = len(attribute_list)
        else:
            assert end <= len(attribute_list)
            stop_match = end
        for idx in range(start, stop_match):
            if Pattern._is_token_match(pattern[0], attribute_list[idx]):
                if self.window_size is None: # No window size
                    match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, len(attribute_list))
                else: # Has window size
                    if end is None: # Match the first token (The end point has not been fixed)
                        match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, min(idx+self.window_size, len(attribute_list)))
                    else: # The end point has been fixed
                        match_indices = self._is_match_recursive(pattern[1:], attribute_list, idx+1, end)
                if isinstance(match_indices, list):
                    return [idx] + match_indices
        return False
                            
    def is_match(self, augtext: AugmentedText) -> [bool, Union[List[int], bool]]:
        match_indices = self._is_match_recursive(self.pattern, augtext.all_features, 0, None)
        return isinstance(match_indices, list), match_indices        
    
    def get_all_labels(self) -> List[Optional[bool]]:
        return self.pos_example_labels + self.neg_example_labels
    
    def global_get_all_labels(self) -> List[bool]:
        ans = self.pos_example_labels + self.neg_example_labels
        ans = [item if item is not None else False for item in ans] # Change None to False
        return ans
    
    @staticmethod
    def pattern_list2str(pattern: List[Set[str]]) -> str:
        ans = [sorted(list(a_set)) for a_set in pattern]
        return str(ans)
    
    def get_pattern_id(self) -> str:
        return Pattern.pattern_list2str(self.pattern)
    
    def _print_match_text(self, augtext: AugmentedText) -> str:
        is_match, match_indices = self.is_match(augtext)
        if not is_match:
            return colored('[NOT MATCH]', 'red', attrs=['blink']) + ': ' + ' '.join(augtext.tokens)
        else:
            ans = colored('[MATCH]', 'green', attrs=['blink']) + ': '
            assert isinstance(match_indices, list)
            for idx, t in enumerate(augtext.tokens):
                if idx not in match_indices:
                    ans += t + ' '
                else:
                    ans += colored(f'{t}:{list(self.pattern[match_indices.index(idx)])}', 'cyan', attrs=['reverse']) + ' '
            return ans
    def get_example_string(self, num_examples: int = 2) -> str:
        num_print_ex = num_examples
        example_string = ''
        if num_print_ex > 0:
            example_string = '\n' + colored('Examples', 'green', attrs=['reverse', 'blink']) + f' ~ Class {self.support_class}:\n'
            examples = []
            if self.support_class == 'Positive':
                ids = [idx for idx, v in enumerate(self.pos_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_ex]:
                    examples.append(self._print_match_text(self.pos_augmented[id]))
                    examples.append('-'*25)
            else:
                ids = [idx for idx, v in enumerate(self.neg_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_ex]:
                    examples.append(self._print_match_text(self.neg_augmented[id]))
                    examples.append('-'*25)
            example_string += '\n'.join(examples)
        return example_string
    
    def get_counterexample_string(self, num_examples: int = 2) -> str:
        num_print_counterex = num_examples
        counterexample_string = ''
        if num_print_counterex > 0:
            counterexample_string = '\n' + colored('Counterexamples', 'red', attrs=['reverse', 'blink']) + f' ~ Not class {self.support_class}:\n'
            counterexamples = []
            if self.support_class == 'Positive':
                ids = [idx for idx, v in enumerate(self.neg_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_counterex]:
                    counterexamples.append(self._print_match_text(self.neg_augmented[id]))
                    counterexamples.append('-'*25)
            else:
                ids = [idx for idx, v in enumerate(self.pos_example_labels) if v]
                random.shuffle(ids)
                for id in ids[:num_print_counterex]:
                    counterexamples.append(self._print_match_text(self.pos_augmented[id]))
                    counterexamples.append('-'*25)
            counterexample_string += '\n'.join(counterexamples)
        return counterexample_string
    
    def print_examples(self, num_examples: int = 2) -> None:
        print(self.get_example_string(num_examples))
        
    def print_counterexamples(self, num_examples: int = 2) -> None:
        print(self.get_counterexample_string(num_examples))
    
    def __str__(self) -> str:                   
        ans_list = [f'Pattern: {self.get_pattern_id()}',
                    f'Window size: {self.window_size}',
                    f'Class: {self.support_class}',
                    f'Precision: {self.precision:.3f}',
                    f'Match: {self.num_total_match} ({self.coverage*100:.1f}%)',
                    f'Gain = {self.global_information_gain:.3f}',
                   ]
        metric_name = '' if callable(self.gain_criteria) else f'({self.gain_criteria}) '
        ans_list.append(f'Metric {metric_name}= {self.metric:.3f}')
        example_string = self.get_example_string(num_examples = self.print_examples[0])
        counterexample_string = self.get_counterexample_string(num_examples = self.print_examples[1])
        return '\n'.join(ans_list) + example_string + counterexample_string + '\n' + ('='*50)