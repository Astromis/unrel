from typing import  Iterable, List

from .attribute import CustomAttribute
from .default_attributes import (DEFAULT_ATTRIBUTES, HypernymAttribute,
                                 LemmaAttribute, SentimentAttribute,
                                 SpacyAttribute, TextAttribute, tokenizer)

# ========== Augmented Text ==========

class AugmentedText():
    
    def __init__(self, 
                 text: str,
                 is_positive: bool = False, 
                 include_standard: List[str] = DEFAULT_ATTRIBUTES, #['TEXT', 'LEMMA', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'], 
                 include_custom: List[CustomAttribute] = []) -> None:
        self.text = text
        self.include_standard = include_standard
        self.include_custom = include_custom
        tokenized_text = tokenizer(self.text)
        self.tokens = [t.text for t in tokenized_text]
        self.lemmas = [t.lemma_ for t in tokenized_text]
        self.features = dict()
        self.attributes = []
        
        # Standard attributes
        if 'TEXT' in self.include_standard:
            self.attributes.append(TextAttribute)
        if 'LEMMA' in self.include_standard:
            self.attributes.append(LemmaAttribute)
        if set(include_standard).intersection(set(['POS', 'DEP', 'NER'])):
            self.attributes.append(SpacyAttribute)
        if 'HYPERNYM' in self.include_standard:
            self.attributes.append(HypernymAttribute)
        if 'SENTIMENT' in self.include_standard:
            self.attributes.append(SentimentAttribute)
            
        # Custom attributes
        self.attributes = self.attributes + self.include_custom
        attribute_names = [attr.name for attr in self.attributes]
        assert len(attribute_names) == len(set(attribute_names)), "Attribute names are not unique. Please do not use TEXT, POS, DEP, HYPERNYM, NER, SENTIMENT as custom attribute names"
        
        # Extraction
        for attr in self.attributes:
            # Filter out 'POS', 'DEP', 'NER' if they are excluded
            if attr.name == 'SPACY':
                ans = attr.extract(self.text, self.tokens)
                for i in range(len(ans)):
                    filtered_ans = []
                    for attr_type in ['POS', 'DEP', 'NER']:
                        if attr_type in self.include_standard:
                            filtered_ans += [item for item in ans[i] if item.startswith(f'SPACY:{attr_type}')]
                    ans[i] = set(filtered_ans)
                self.features[attr.name] = ans
            elif attr.name == 'LEMMA':
                self.features[attr.name] = attr.extract(self.text, self.lemmas)
            else:
                self.features[attr.name] = attr.extract(self.text, self.tokens)
            assert len(self.features[attr.name]) == len(self.tokens), f"The number of tokens returned by the extraction function of {attr.name} is not equal to tokens from Spacy."
            
        # Merge features from all attributes
        self.all_features = []
        for i in range(len(self.tokens)):
            t_ans = set()
            for attr_name, f in self.features.items():
                t_ans.update(f[i])
            self.all_features.append(t_ans)
        assert len(self.all_features) == len(self.tokens)
        
        # A set of all unique features in this text
        self.all_unique_features = set([item for t_ans in self.all_features for item in t_ans])
        
    def __str__(self) -> str:
        lines = [f'{t}: {f}' for t, f in zip(self.tokens, self.all_features)]
        return self.text + '\n' + '\n'.join(lines)
    
    def keep_only_features(self, features_to_keep: Iterable[str]) -> None:
        for i in range(len(self.all_features)):
            self.all_features[i] = self.all_features[i].intersection(features_to_keep)

