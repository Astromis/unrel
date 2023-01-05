import os
from typing import List, Set

import nltk
import spacy
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk

from .attribute import Attribute

nlp = spacy.load('en_core_web_sm')#ru_core_news_sm
tokenizer = spacy.load('en_core_web_sm', disable = ['parser', 'ner', 'textcat']) # Use only tokenizer

DEFAULT_ATTRIBUTES =  ['TEXT', 'LEMMA', 'POS', 'DEP', 'NER', 'HYPERNYM', 'SENTIMENT'] # ['TEXT', 'POS', 'NER', 'HYPERNYM', 'SENTIMENT']


# ----- Text attribute -----        
def _text_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    tokens = map(str.lower, tokens)
    return [set([t]) for t in tokens]

def _text_translation(attr:str, 
                      is_complement:bool = False) -> str:
    word = attr[5:]
    return f'the word "{word}"'

TextAttribute = Attribute(name = 'TEXT', extraction_function = _text_extraction, translation_function = _text_translation)

# ----- Lemma attribute -----        
def _lemma_extraction(text: str, lemmas: List[str]) -> List[Set[str]]:
    lemmas = map(str.lower, lemmas)
    return [set([t]) for t in lemmas]

def _lemma_translation(attr:str, 
                      is_complement:bool = False) -> str:
    lemma = attr[6:]
    return f'a form of "{lemma}"'

LemmaAttribute = Attribute(name = 'LEMMA', extraction_function = _lemma_extraction, translation_function = _lemma_translation)


# ----- Spacy attribute (POS, DEP, NER) -----
def _spacy_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    ans = []
    for t in nlp(text):
        t_ans = []
        t_ans.append(f'POS-{t.pos_}') # Universal dependency tag https://universaldependencies.org/u/pos/
        # t_ans.append(f'POS-{t.tag_}') # Penn tree bank Part-of-speech https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        t_ans.append(f'DEP-{t.dep_}') # Dependency parsing tag
        if t.ent_type_ != "":
            t_ans.append(f'NER-{t.ent_type_}') # Named-entity recognition
        ans.append(set(t_ans))
    return ans

def _spacy_translation(attr:str, 
                      is_complement:bool = False) -> str:
    subtype = attr[6:9]
    value = attr[10:]

    if subtype == 'POS':
        mapping = {
            'ADJ': 'an adjective (describes a noun)',
            'ADP': 'a preposition',
            'ADV': 'an adverb (describes a verb)',
            'AUX': 'an auxiliary (a function word for verbs)',
            'CCONJ': 'a coordinating conjunction (links parts of the sentence)',
            'DET': 'a determiner (expresses a reference of a noun)',
            'INTJ': 'an interjection (expresses an emotional reaction)',
            'NOUN': 'a noun',
            'NUM': 'a number',
            'PART': 'a function word of type particle',
            'PRON': 'a pronoun',
            'PROPN': 'a proper noun (a name of a specific individual, place, or object)',
            'PUNCT': 'a punctuation mark',
            'SCONJ': 'a subordinating conjunction (defines a relation between sentence parts)',
            'SYM': 'a symbol',
            'VERB': 'a verb',
            'X': 'a word of category misc.',
            'SPACE': 'a space',
            'EOL': 'an end of line (EOL)'
        }
        return mapping[value]

    elif subtype == 'DEP':
        if is_complement:
            return f'having the dependency type "{value}"'
        else:
            return f'a word with the dependency type "{value}"'

    elif subtype == 'NER':
        mapping = {
            "PERSON": "a named person",
            "NORP": "a nationality or a religious or political group",
            "FACILITY": "a facility (such as a building, an airport, etc.)",
            "FAC": "a facility (such as a building, an airport, etc.)",
            "ORG": "an organization (such as a company, an agency, etc.)",
            "GPE": "a country, city, or state",
            "LOC": "a non-geopolitical location (such as a mountain range, a body of water)",
            "PRODUCT": "a named product",
            "EVENT": "a named event (such as a hurricane, a battle, etc.)",
            "WORK_OF_ART": "a title of a work of art (such as a book, a song, etc.)",
            "LAW": "a named document made into laws",
            "LANGUAGE": "a named language",
            "DATE": "an absolute or relative date or period",
            "TIME": "a time smaller than a day",
            "PERCENT": 'a percentage (including "%")',
            "MONEY": "a monetary value",
            "QUANTITY": "a measurement",
            "ORDINAL": 'an ordinal number (such as "first", "second", etc.)',
            "CARDINAL": 'a cardinal number (such as one, two, etc.)',
        }
        return mapping[value]

    else:
        assert False, f"Invalid spacy attribute {attr}"

SpacyAttribute = Attribute(name = 'SPACY', extraction_function = _spacy_extraction, translation_function = _spacy_translation)

# ----- Hypernym attribute -----
HYPERNYM_DICT = dict()

def _get_anvr_pos(penn_tag: str): # Return Literal['a', 'n', 'v', 'r', None]
    if penn_tag.startswith('JJ'): # Adjective
        return 'a'
    elif penn_tag.startswith('NN'): # Noun
        return 'n'
    elif penn_tag.startswith('VB'): # Verb
        return 'v'
    elif penn_tag.startswith('RB'): # Adverb
        return 'r'
    else: # Invalid wordnet type
        return None 
    
def _get_all_hypernyms(synset: nltk.corpus.reader.wordnet.Synset) -> Set[nltk.corpus.reader.wordnet.Synset]:
    if str(synset) not in HYPERNYM_DICT:
        ans = set()
        direct_hypernyms = synset.hypernyms() # type: List[nltk.corpus.reader.wordnet.Synset]
        for ss in direct_hypernyms:
            ans.update(_get_all_hypernyms(ss))
            ans.update(set([ss]))
        HYPERNYM_DICT[str(synset)] = ans
    return HYPERNYM_DICT[str(synset)]
    
def _hypernym_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    ans = []
    for t in nlp(text):
        pos = _get_anvr_pos(t.tag_)
        if pos is not None:
            synset = lesk(tokens, t.text, pos)
            if synset is not None:
                all_hypernyms = set([synset]) # This version of hypernym extraction includes synset of the word itself
                all_hypernyms.update(_get_all_hypernyms(synset))
                ans.append(set([str(ss)[8:-2] for ss in all_hypernyms]))
            else:
                ans.append(set([]))
        else:
            ans.append(set([]))
    return ans

def _hypernym_translation(attr:str, 
                      is_complement:bool = False) -> str:
    synset = attr.split(":")[1].split('.')
    word, pos, idx = synset[0], synset[1], synset[2]
    if pos == 'a': pos = 'adj'
    if pos == 'r': pos = 'adv' 
    return f'a type of {word} ({pos})'

# ----- Hypernym-N attribute -----
def _get_all_hypernyms_above(synset: nltk.corpus.reader.wordnet.Synset, above: int) -> Set[nltk.corpus.reader.wordnet.Synset]:
    if above == 0:
        return set()
    
    if (str(synset), above) not in HYPERNYM_DICT:
        ans = set()
        direct_hypernyms = synset.hypernyms() # type: List[nltk.corpus.reader.wordnet.Synset]
        for ss in direct_hypernyms:
            ans.update(_get_all_hypernyms_above(ss, above-1))
            ans.update(set([ss]))
        HYPERNYM_DICT[(str(synset), above)] = ans
    return HYPERNYM_DICT[(str(synset), above)]

def get_custom_hypernym_extraction_function(above: int = 3, wsd: str = 'lesk'):
    assert wsd in ['lesk', 'first'], f"Invalid word sense disambiguation mode: {wsd}"
    def _custom_hypernym_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
        wsd_mode = wsd
        ans = []
        for t in nlp(text):
            pos = _get_anvr_pos(t.tag_)
            if pos is not None:
                if wsd == 'lesk':
                    synset = lesk(tokens, t.text, pos)
                elif wsd == 'first':
                    all_synsets = wn.synsets(t.text, pos=pos)
                    synset = all_synsets[0] if all_synsets else None

                if synset is not None:
                    all_hypernyms = set([synset]) # This version of hypernym extraction includes synset of the word itself
                    all_hypernyms.update(_get_all_hypernyms_above(synset, above))
                    ans.append(set([str(ss)[8:-2] for ss in all_hypernyms]))
                else:
                    ans.append(set([]))
            else:
                ans.append(set([]))
        return ans
    return _custom_hypernym_extraction

HypernymAttribute = Attribute(name = 'HYPERNYM', extraction_function = get_custom_hypernym_extraction_function(above = 3, wsd = 'lesk'), translation_function = _hypernym_translation)

# ----- Sentiment attribute -----        
# Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In International Conference on Knowledge Discovery and Data Mining, KDD’04, pages 168–177. (https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon)

POS_LEXICON_FILENAME = os.path.join(os.path.dirname(__file__), 'resources/opinion-lexicon-English/positive-words.txt')
NEG_LEXICON_FILENAME = os.path.join(os.path.dirname(__file__), 'resources/opinion-lexicon-English/negative-words.txt')
POSITIVE_LEXICON = [line.strip().lower() for line in open(POS_LEXICON_FILENAME, encoding="iso-8859-1") if line.strip() != '' and line[0] != ';']
NEGATIVE_LEXICON = [line.strip().lower() for line in open(NEG_LEXICON_FILENAME, encoding="iso-8859-1") if line.strip() != '' and line[0] != ';']

def _sentiment_extraction(text: str, tokens: List[str]) -> List[Set[str]]:
    tokens = map(str.lower, tokens)
    ans = []
    for t in tokens:
        t_ans = []
        if t.lower() in POSITIVE_LEXICON:
            t_ans.append('pos')
        if t.lower() in NEGATIVE_LEXICON:
            t_ans.append('neg')
        ans.append(set(t_ans))
    return ans

def _sentiment_translation(attr:str, 
                      is_complement:bool = False) -> str:
    sentiment = attr.split(":")[1]
    sentiment = 'positive' if sentiment == 'pos' else 'negative'
    if is_complement:
        return f'bearing a {sentiment} sentiment'
    else:
        return f'a {sentiment}-sentiment word'
    
SentimentAttribute = Attribute(name = 'SENTIMENT', extraction_function = _sentiment_extraction, translation_function = _sentiment_translation, values = ['pos', 'neg'])

# ALL_ATTRIBUTES = {attr.name:attr for attr in [TextAttribute, SpacyAttribute, HypernymAttribute, SentimentAttribute]}