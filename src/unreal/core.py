import warnings
from collections import Counter
from math import log
from string import punctuation
from typing import List, Union

import numpy as np
import pandas as pd
from nltk import ngrams, sent_tokenize, word_tokenize
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.tokenize import SyllableTokenizer
from rusyll.rusyll import token_to_syllables
from scipy.stats import hypergeom


def tokenize_from_list_strings(string_list:List[str], stop_words=None, ngram=1):
    tokens = word_tokenize(' '.join(string_list))
    tokens = list(filter(lambda x: x not in punctuation, tokens))
    if stop_words:
        tokens = list(filter(lambda x: x not in stop_words, tokens))
    return tokens

def get_ngrams(tokens: List[str], n, join_with=None):
    ngrams_ = ngrams(tokens, n)
    if join_with:
        ngrams_ = [join_with.join(x) for x in ngrams_]
    return ngrams_


def get_trigram_assisiations(word_tokens: List[str], top_n=100):
    trigram_measures = TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(word_tokens)
    return finder.nbest(trigram_measures.pmi, top_n)
        

def calculate_idf(tokenized_texts: List[List[str]]):
    idf = Counter()
    for t in tokenized_texts:
        for w in set(t):
            idf[w] += 1
    for w in idf:
        idf[w] = log(len(tokenized_texts)/idf[w], 2)
    return idf

def heaviside(x: Union[int, float]):
    if x< 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        return x

def calculate_htz(text_inclass: list, text_outclass: list):
    metric = []
    t1 = calculate_idf(text_inclass)
    t2 = calculate_idf(text_outclass)
    for w in t1:
        delta = t2[w] - t1[w]
        metric.append( (w, delta * heaviside(delta)) )
    metric.sort(key = lambda x: x[1], reverse=True)

    return metric

def get_htz(df, top_k=20):
    texts = df.text.to_list()
    texts = [word_tokenize(t) for t in texts]
    texts = np.array(texts)

    lexicon = {l:[] for l in df.label.to_list()}
    for label in lexicon:
        in_class = texts[df[df.label == label].index]
        out_class = texts[df[df.label != label].index]

        lexicon[label] = [x[0] for x in calculate_htz(in_class, out_class)][:top_k]
    #FIXME: check that all word lists are equal
    return pd.DataFrame(lexicon)

def calculate_rtd(tokens_df1_freq: np.array, tokens_df2_freq: np.array, alpha=0.3):

    unnormilized_divergance = np.abs(1 / (tokens_df1_freq ** alpha) - 1 / (tokens_df2_freq ** alpha)) ** (1 / (alpha+1))
    norm_tern_1 = (alpha+1) / alpha * np.sum( np.abs( 1/(tokens_df1_freq ** alpha) - 1/ ( ( tokens_df1_freq.shape[0] + 0.5 *  tokens_df2_freq.shape[0]) ** alpha)  ) **(1/(alpha+1)) )
    norm_tern_2 = (alpha+1) / alpha * np.sum( np.abs( 1/ ( ( tokens_df2_freq.shape[0] + 0.5 *  tokens_df1_freq.shape[0]) ** alpha) - 1/(tokens_df2_freq ** alpha) ) **(1/(alpha+1)) )
    norm = norm_tern_1 + norm_tern_2
    rde =  1 / norm * (alpha + 1) / alpha * unnormilized_divergance
    return rde

def get_common_words_intersect(lst1: list, lst2:list ):
    c1 = Counter(lst1)
    c2 = Counter(lst2)
    return Counter({ k: c1[k] + c2[k] for k in c1.keys() & c2.keys() }), c1, c2

def get_rtd(corp_tokens: List[str], ref_corp_tokens: List[str]):

    common_words, corp1_freq, corp2_freq  = get_common_words_intersect(corp_tokens,ref_corp_tokens)
    common_words = common_words.most_common()
    common_words = [x[0] for x in common_words]
    tokens_df1_freq = []
    tokens_df2_freq = []

    corp1_freq = [x[0] for x in corp1_freq.most_common()]
    corp2_freq = [x[0] for x in corp2_freq.most_common()] 
    for w in common_words:
        try:
            tokens_df1_freq.append(corp1_freq.index(w) +1)
            tokens_df2_freq.append(corp2_freq.index(w) + 1)
        except:
            print(w)
            continue
    tokens_df1_freq = np.array(tokens_df1_freq)
    tokens_df2_freq = np.array(tokens_df2_freq)
    rde = calculate_rtd(tokens_df1_freq, tokens_df2_freq)
    
    t = list(zip(common_words,rde))
    t.sort(key=lambda x: x[1], reverse=True)
    top_rtd = []
    for i, (w, d) in enumerate(t):
        top_rtd.append((i, w, d*10000, corp1_freq.index(w), corp2_freq.index(w)))
        
    return pd.DataFrame(top_rtd, columns=["index", "word", "rtd", "rank in our corpus", "rank in common one"])

def get_flash_readibility(text:str, lang, type)-> float:
    if lang not in ["english", "russian"]:
        raise NotImplementedError(lang)
    tokens = word_tokenize(text, lang)
    sents = sent_tokenize(text, lang)
    mean_sent_len = np.mean([len(x.split(' ')) for x in sents])
    if lang == "russian":
        mean_word_len = np.mean([len(token_to_syllables(x)) for x in tokens])
        if type == "flash":
            readibility = 206.836-60.1*mean_word_len-1.3*mean_sent_len
        elif type == "flash-kinside":
            readibility = 0.5*mean_sent_len + 8.4*mean_word_len - 15.59

    else:
        SSP = SyllableTokenizer()
        mean_word_len = np.mean([len(SSP.tokenize(x)) for x in tokens])
        if type == "flash":
            readibility = 206.836-84.6*mean_word_len-1.015*mean_sent_len
        elif type == "flash-kinside":
            readibility = 0.39*mean_sent_len + 11.8*mean_word_len - 15.59
    return readibility

def fog_index(text: str, lang) -> float:
    if lang not in ["english", "russian"]:
        raise NotImplementedError(lang)
    tokens = word_tokenize(text, lang)
    if len(tokens) < 100:
        warnings.warn(f"Text should contain at least 100 tokens. You have {len(tokens)}")
    middle = len(tokens) // 2
    selected_tokens = tokens[middle-50:middle] + tokens[middle:middle+50]
    sents = sent_tokenize(' '.join(selected_tokens), lang)
    mean_sent_len = np.mean([len(x.split(' ')) for x in sents])
    if lang == "russian":
        word_syll = [len(token_to_syllables(x)) for x in tokens]
    else:
        SSP = SyllableTokenizer()
        word_syll = [len(SSP.tokenize(x)) for x in tokens]
    complex_words = list(filter(lambda x: x > 2, word_syll))
    complex_words_frac = len(complex_words) / len(selected_tokens)
    if lang == "english":
        fog_index = (mean_sent_len + complex_words_frac) * 0.4
    else:
        fog_index = (1.25*mean_sent_len + 0.24*complex_words_frac) * 0.4
    return fog_index

def get_ttr(tokens: List[str]):
    return len(set(tokens)) / len(tokens)

def get_hd_d(tokens: List[str], sample_size=42):
    token_counter = Counter(tokens)
    len_all_tokens = len(tokens)
    proba = []
    for word in token_counter:
        hd = hypergeom(len_all_tokens, token_counter[word], sample_size )
        proba.append(1-hd.pmf(0))
    return sum(proba) / sample_size

def calculate_mtld(tokens, stabilization_point):
    start = 0
    factor_counter = 0
    rest_factor = 0
    for i in range(1,len(tokens)):
        curr_ttr = get_ttr(tokens[start:i])
        if  curr_ttr <= stabilization_point:
            factor_counter += 1
            start = i
    if curr_ttr > stabilization_point:
        rest_factor = 1 - ( (curr_ttr - stabilization_point) / (1 - curr_ttr))
    return len(tokens) / (factor_counter + rest_factor)

def get_mtld(tokens: List[str], stabilization_point=0.72):
    forward = calculate_mtld(tokens, stabilization_point)
    backward = calculate_mtld(tokens[::-1], stabilization_point)
    return (forward + backward) /2