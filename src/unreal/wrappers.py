
from typing import List

import gensim
import pandas as pd
from gsdmm import MovieGroupProcess
from scipy.stats import chi2_contingency


def get_chi2(df_, var_name1:str, var_name2:str):
    return chi2_contingency(pd.crosstab(df_[var_name1], df_[var_name2]))

def cluster_by_mgp(unique_text_tokens: List[List[str]], k: int, vocab_size=20000, scores=False):
    mgp = MovieGroupProcess(K=k, alpha=0.1, beta=0.1, n_iters=30)
    labels = mgp.fit(unique_text_tokens, vocab_size)
    #if scores:
    #    return labels, 
    return labels

def get_gensim_lda(corpus: List[List[str]], num_topics, supplumentary=False):
    dic = gensim.corpora.Dictionary(corpus, )
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = num_topics, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    if supplumentary:
        return lda_model, dic, bow_corpus
    else: 
        return lda_model